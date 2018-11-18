package kalmf

import (
	"sync"

	"utils"
)

// KalmanFilterC  'C' for chained
type KalmanFilterC interface {
	Predict(*utils.Dense, *utils.Dense, *utils.Dense) (utils.Dense, utils.Dense, error)
	Update([]float64, *utils.Dense) (utils.Dense, utils.Dense, error)
}

type BaseKalmanFilterC struct {
	predictNoiseCovrs *utils.BandDense // Q, only diagonal elements, n * n
	measureNoiseCovrs *utils.BandDense // R, only diagonal elements, n * n

	states   *utils.Dense // may be predict or fused, n * 1
	errCovrs *utils.Dense // may be predict or fused, n * n

	lock sync.RWMutex
}

func NewBaseKalmanFilterC(initStates []float64, initErrCovrsMat *utils.Dense, predictNoiseVar, measureNoiseVar []float64) (KalmanFilterC, error) {
	f := &BaseKalmanFilterC{}
	nStates := len(initStates)
	f.states = utils.NewDense(nStates, 1, initStates)
	if r, c := initErrCovrsMat.Dims(); r != nStates || c != nStates {
		return nil, ErrMatrixDimNotMatch
	}
	f.errCovrs = initErrCovrsMat
	if len(predictNoiseVar) != nStates || len(measureNoiseVar) != nStates {
		return nil, ErrMatrixDimNotMatch
	}
	f.predictNoiseCovrs = utils.NewDiagonalRect(nStates, nStates, predictNoiseVar)
	f.measureNoiseCovrs = utils.NewDiagonalRect(nStates, nStates, measureNoiseVar)
	return f, nil
}

// Predict ctrlTransMat * ctrlMat must be n * 1
func (f *BaseKalmanFilterC) Predict(statesTrans *utils.Dense, ctrlTrans *utils.Dense, ctrl *utils.Dense) (states, errCovrs utils.Dense, err error) {
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

	f.lock.Lock()
	defer f.lock.Unlock()

	f.states = statesTrans.Mul(f.states).Add(ctrlTrans.Mul(ctrl))
	f.errCovrs = statesTrans.Mul(f.errCovrs).Mul(statesTrans.T()).Add(f.predictNoiseCovrs)

	states = *f.states
	errCovrs = *f.errCovrs

	return
}

// Update correct states using measurements
func (f *BaseKalmanFilterC) Update(measures []float64, measureTrans *utils.Dense) (states, errCovrs utils.Dense, err error) {
	nStates, _ := f.states.Dims()
	if len(measures) != nStates {
		err = ErrMatrixDimNotMatch
		return
	}
	if r, c := measureTrans.Dims(); r != nStates || c != nStates {
		err = ErrMatrixDimNotMatch
		return
	}

	kg := utils.NewDense(nStates, nStates, nil)
	measuresMat := utils.NewDense(nStates, 1, measures)

	f.lock.Lock()
	defer f.lock.Unlock()

	// kalman gain
	measureTransT := measureTrans.T()
	inversedMat, err := measureTrans.Mul(f.errCovrs).Mul(measureTransT).Add(f.measureNoiseCovrs).Inverse()
	if err != nil {
		return
	}
	kg = f.errCovrs.Mul(measureTransT).Mul(inversedMat)

	// new states
	f.states = f.states.Add(kg.Mul(measuresMat.Sub(measureTrans.Mul(f.states))))

	// new err covrs
	f.errCovrs = f.errCovrs.Sub(kg.Mul(measureTrans).Mul(f.errCovrs))

	states = *f.states
	errCovrs = *f.errCovrs

	return
}
