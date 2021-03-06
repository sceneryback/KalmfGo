# KalmfGo
This is Kalman filter in Golang implemention, basically the following 5 equations:

<p align="left">
    <img src="resources/filter.png"  width="450" height="300">
</p>

It uses gonum (https://github.com/gonum/gonum) for matrix manipulations and also provides a ***prettier*** version in chained manner, e.g.:
```
mRes.Mul(mA, mB)
mRes.Add(mRes, mC)
```
becomes:
```
mA.Mul(mB).Add(mC)
```
Hope it helps.

## References
For detailed derivations of kalman filter, please refer:   
https://medium.com/@bronzesword/kalman-filter-primer-derivations-dc92983911e5