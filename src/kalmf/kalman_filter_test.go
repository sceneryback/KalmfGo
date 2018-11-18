package kalmf

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"

	"github.com/googollee/go-assert"
)

var Tolerance = math.Pow10(-3)

func EqualFloats(floats1, floats2 []float64) bool {
	len1 := len(floats1)
	len2 := len(floats2)
	if len1 != len2 {
		return false
	}

	for i := 0; i < len1; i++ {
		if math.Abs(floats1[i]-floats2[i]) > Tolerance {
			return false
		}
	}

	return true
}

func TestKalmanFilter(t *testing.T) {
	initStates := []float64{0, 0}
	initErrCovrs := mat.NewDense(2, 2, []float64{0, 0, 0, 0})
	predictNoiseVar := []float64{0.2, 0.2}
	updateNoiseVar := []float64{0.2, 0.2}
	kalmf, err := NewBaseKalmanFilter(initStates, initErrCovrs, predictNoiseVar, updateNoiseVar)
	assert.Equal(t, err, nil, "no error")

	// predict 1
	stateTrans1 := mat.NewDense(2, 2, []float64{1, 1, 0, 1})
	ctrlTrans1 := mat.NewDense(2, 1, []float64{0.5, 1})
	ctrl1 := mat.NewDense(1, 1, []float64{2})
	states1, covrs1, err := kalmf.Predict(stateTrans1, ctrlTrans1, ctrl1)
	assert.Equal(t, err, nil, "no error")
	expectedStates1 := mat.NewDense(2, 1, []float64{1, 2})
	assert.Equal(t, EqualFloats(states1.RawMatrix().Data, expectedStates1.RawMatrix().Data), true, "states after predict 1")
	expectedCovrs1 := mat.NewDense(2, 2, []float64{0.2, 0, 0, 0.2})
	assert.Equal(t, EqualFloats(covrs1.RawMatrix().Data, expectedCovrs1.RawMatrix().Data), true, "covrs after predict 1")

	// predict 2
	stateTrans2 := mat.NewDense(2, 2, []float64{1, 1, 0, 1})
	ctrlTrans2 := mat.NewDense(2, 1, []float64{0.5, 1})
	ctrl2 := mat.NewDense(1, 1, []float64{3})
	states2, covrs2, err := kalmf.Predict(stateTrans2, ctrlTrans2, ctrl2)
	assert.Equal(t, err, nil, "no error")
	expectedStates2 := mat.NewDense(2, 1, []float64{4.5, 5})
	assert.Equal(t, EqualFloats(states2.RawMatrix().Data, expectedStates2.RawMatrix().Data), true, "states after predict 2")
	expectedCovrs2 := mat.NewDense(2, 2, []float64{0.6, 0.2, 0.2, 0.4})
	assert.Equal(t, EqualFloats(covrs2.RawMatrix().Data, expectedCovrs2.RawMatrix().Data), true, "covrs after predict 2")

	// update
	measures := []float64{4, 5.5}
	measureTrans := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	states3, covrs3, err := kalmf.Update(measures, measureTrans)
	assert.Equal(t, err, nil, "no error")
	expectedStates3 := mat.NewDense(2, 1, []float64{4.182, 5.273})
	assert.Equal(t, EqualFloats(states3.RawMatrix().Data, expectedStates3.RawMatrix().Data), true, "states after update")
	expectedCovrs3 := mat.NewDense(2, 2, []float64{0.146, 0.019, 0.019, 0.128})
	assert.Equal(t, EqualFloats(covrs3.RawMatrix().Data, expectedCovrs3.RawMatrix().Data), true, "covrs after update")

}
