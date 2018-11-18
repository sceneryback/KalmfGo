package utils

import (
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

type Dense struct {
	m *mat.Dense
}

type BandDense struct {
	m *mat.BandDense
}

func NewDense(r, c int, data []float64) *Dense {
	return &Dense{mat.NewDense(r, c, data)}
}

func NewDiagonalRect(r, c int, data []float64) *BandDense {
	return &BandDense{mat.NewBandDense(r, c, 0, 0, data)}
}

func (d *Dense) Dims() (r, c int) {
	return d.m.Dims()
}

func (d *Dense) At(i, j int) float64 {
	return d.m.At(i, j)
}

func (d *Dense) T() mat.Matrix {
	return d.m.T()
}

func (d *BandDense) Dims() (r, c int) {
	return d.m.Dims()
}

func (d *BandDense) At(i, j int) float64 {
	return d.m.At(i, j)
}

func (d *BandDense) T() mat.Matrix {
	return d.m.T()
}

func (d *Dense) Add(a mat.Matrix) (ret *Dense) {
	r, c := a.Dims()
	ret = NewDense(r, c, nil)
	ret.m.Add(d.m, a)
	return
}

func (d *Dense) Sub(a mat.Matrix) (ret *Dense) {
	r, c := a.Dims()
	ret = NewDense(r, c, nil)
	ret.m.Sub(d.m, a)
	return
}

func (d *Dense) Mul(a mat.Matrix) (ret *Dense) {
	r, _ := d.m.Dims()
	_, c := a.Dims()
	ret = NewDense(r, c, nil)
	ret.m.Mul(d.m, a)
	return
}

func (d *Dense) Inverse() (ret *Dense, err error) {
	r, c := d.m.Dims()
	ret = NewDense(r, c, nil)
	err = ret.m.Inverse(d.m)
	return
}

func (d *Dense) RawMatrix() blas64.General {
	return d.m.RawMatrix()
}
