package kalmf

import (
	"errors"
	"sync"

	"gonum.org/v1/gonum/mat"
)

var (
	ErrMatrixDimNotMatch = errors.New("matrix dimension not match with states")
)

type KalmanFilter interface {
	Predict(*mat.Dense, *mat.Dense, *mat.Dense) (mat.Dense, mat.Dense, error)
	Update([]float64, *mat.Dense) (mat.Dense, mat.Dense, error)
}

type BaseKalmanFilter struct {
	predictNoiseCovrs *mat.BandDense // Q, only diagonal elements, n * n
	measureNoiseCovrs *mat.BandDense // R, only diagonal elements, n * n

	states   *mat.Dense // may be predict or fused, n * 1
	errCovrs *mat.Dense // may be predict or fused, n * n

	lock sync.RWMutex
}

func NewBaseKalmanFilter(initStates []float64, initErrCovrsMat *mat.Dense, predictNoiseVar, measureNoiseVar []float64) (KalmanFilter, error) {
	f := &BaseKalmanFilter{}
	nStates := len(initStates)
	f.states = mat.NewDense(nStates, 1, initStates)
	if r, c := initErrCovrsMat.Dims(); r != nStates || c != nStates {
		return nil, ErrMatrixDimNotMatch
	}
	f.errCovrs = initErrCovrsMat
	if len(predictNoiseVar) != nStates || len(measureNoiseVar) != nStates {
		return nil, ErrMatrixDimNotMatch
	}
	f.predictNoiseCovrs = mat.NewDiagonalRect(nStates, nStates, predictNoiseVar)
	f.measureNoiseCovrs = mat.NewDiagonalRect(nStates, nStates, measureNoiseVar)
	return f, nil
}

// Predict ctrlTransMat * ctrlMat must be n * 1
func (f *BaseKalmanFilter) Predict(statesTrans *mat.Dense, ctrlTrans *mat.Dense, ctrl *mat.Dense) (states, errCovrs mat.Dense, err error) {
	nStates, _ := f.states.Dims()
	if r, c := statesTrans.Dims(); r != nStates || c != nStates {
		err = ErrMatrixDimNotMatch
		return
	}

	if ctr, _ := ctrlTrans.Dims(); ctr != nStates {
		err = ErrMatrixDimNotMatch
		return
	}

	if _, cc := ctrl.Dims(); cc != 1 {
		err = ErrMatrixDimNotMatch
		return
	}

	transedStates := mat.NewDense(nStates, 1, nil)
	transedCtrl := mat.NewDense(nStates, 1, nil)
	covrTrans := mat.NewDense(nStates, nStates, nil)

	f.lock.Lock()
	defer f.lock.Unlock()

	transedStates.Mul(statesTrans, f.states)
	transedCtrl.Mul(ctrlTrans, ctrl)
	f.states.Add(transedStates, transedCtrl)

	covrTrans.Product(statesTrans, f.errCovrs, statesTrans.T())
	f.errCovrs.Add(covrTrans, f.predictNoiseCovrs)

	states = *f.states
	errCovrs = *f.errCovrs

	return
}

// Update correct states using measurements
func (f *BaseKalmanFilter) Update(measures []float64, measureTrans *mat.Dense) (states, errCovrs mat.Dense, err error) {
	nStates, _ := f.states.Dims()
	if len(measures) != nStates {
		err = ErrMatrixDimNotMatch
		return
	}
	if r, c := measureTrans.Dims(); r != nStates || c != nStates {
		err = ErrMatrixDimNotMatch
		return
	}

	kg := mat.NewDense(nStates, nStates, nil)
	kg1 := mat.NewDense(nStates, nStates, nil)
	kg2 := mat.NewDense(nStates, nStates, nil)
	kg3 := mat.NewDense(nStates, nStates, nil)
	kg4 := mat.NewDense(nStates, nStates, nil)
	transedStates := mat.NewDense(nStates, 1, nil)
	measuresMat := mat.NewDense(nStates, 1, measures)
	measurePredictErr := mat.NewDense(nStates, 1, nil)
	weightedStates := mat.NewDense(nStates, 1, nil)
	covrs := mat.NewDense(nStates, nStates, nil)

	f.lock.Lock()
	defer f.lock.Unlock()

	// kalman gain
	kg1.Mul(f.errCovrs, measureTrans.T())
	kg2.Product(measureTrans, f.errCovrs, measureTrans.T())
	kg3.Add(kg2, f.measureNoiseCovrs)
	err = kg4.Inverse(kg3)
	if err != nil {
		return
	}
	kg.Mul(kg1, kg4)

	// new states
	transedStates.Mul(measureTrans, f.states)
	measurePredictErr.Sub(measuresMat, transedStates)
	weightedStates.Mul(kg, measurePredictErr)
	f.states.Add(f.states, weightedStates)

	// new err covrs
	covrs.Product(kg, measureTrans, f.errCovrs)
	f.errCovrs.Sub(f.errCovrs, covrs)

	states = *f.states
	errCovrs = *f.errCovrs

	return
}
