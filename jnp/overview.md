# jax.numpy functions

Reference page: https://docs.jax.dev/en/latest/jax.numpy.html

## Pointwise functions

### Math

[`abs`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.abs.html#jax.numpy.abs "jax.numpy.abs")(x, /)
[`absolute`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.absolute.html#jax.numpy.absolute "jax.numpy.absolute")(x, /)
: absolute value.

[`fabs`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fabs.html#jax.numpy.fabs "jax.numpy.fabs")(x, /)
: absolute value but real.

[`sign`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sign.html#jax.numpy.sign "jax.numpy.sign")(x, /)

[`signbit`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.signbit.html#jax.numpy.signbit "jax.numpy.signbit")(x, /)

[`copysign`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.copysign.html#jax.numpy.copysign "jax.numpy.copysign")(x1, x2, /)
: copy the sign of `x2` to the element of `x1`.


#### Trigonometry, Complex, Hyperbolic

[`exp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.exp.html#jax.numpy.exp "jax.numpy.exp")(x, /)

[`exp2`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.exp2.html#jax.numpy.exp2 "jax.numpy.exp2")(x, /)
: base-2.

[`expm1`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.expm1.html#jax.numpy.expm1 "jax.numpy.expm1")(x, /)
: `exp(x) - 1`. Direct calculation might have some precision issue when `x` is too close to zero.

[`log`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.log.html#jax.numpy.log "jax.numpy.log")(x, /)

[`log2`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.log2.html#jax.numpy.log2 "jax.numpy.log2")(x, /)

[`log10`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.log10.html#jax.numpy.log10 "jax.numpy.log10")(x, /)

[`log1p`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.log1p.html#jax.numpy.log1p "jax.numpy.log1p")(x, /)
: `log(x + 1)`. Direct calculation can have some precision issues when `x` is too close to zero.

[`logaddexp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logaddexp.html#jax.numpy.logaddexp "jax.numpy.logaddexp")(x1, x2, ...)
[`logaddexp2`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logaddexp2.html#jax.numpy.logaddexp2 "jax.numpy.logaddexp2")(x1, x2, ...)
: `log(exp(x1) + exp(x2))` (base e or 2) but carefully implemented to avoid overflow.

[`cos`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cos.html#jax.numpy.cos "jax.numpy.cos")(x, /)
[`sin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sin.html#jax.numpy.sin "jax.numpy.sin")(x, /)
[`tan`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tan.html#jax.numpy.tan "jax.numpy.tan")(x, /)

[`arccos`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arccos.html#jax.numpy.arccos "jax.numpy.arccos")(x, /)
[`acos`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.acos.html#jax.numpy.acos "jax.numpy.acos")(x, /)

[`arcsin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arcsin.html#jax.numpy.arcsin "jax.numpy.arcsin")(x, /)
[`asin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.asin.html#jax.numpy.asin "jax.numpy.asin")(x, /)

[`arctan`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arctan.html#jax.numpy.arctan "jax.numpy.arctan")(x, /)
[`atan`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atan.html#jax.numpy.atan "jax.numpy.atan")(x, /)
[`arctan2`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arctan2.html#jax.numpy.arctan2 "jax.numpy.arctan2")(x1, x2, /)
[`atan2`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atan2.html#jax.numpy.atan2 "jax.numpy.atan2")(x1, x2, /)

[`cosh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cosh.html#jax.numpy.cosh "jax.numpy.cosh")(x, /)
[`sinh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sinh.html#jax.numpy.sinh "jax.numpy.sinh")(x, /)
[`tanh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tanh.html#jax.numpy.tanh "jax.numpy.tanh")(x, /)

[`arccosh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arccosh.html#jax.numpy.arccosh "jax.numpy.arccosh")(x, /)
[`acosh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.acosh.html#jax.numpy.acosh "jax.numpy.acosh")(x, /)

[`arcsinh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arcsinh.html#jax.numpy.arcsinh "jax.numpy.arcsinh")(x, /
[`asinh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.asinh.html#jax.numpy.asinh "jax.numpy.asinh")(x, /)

[`arctanh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arctanh.html#jax.numpy.arctanh "jax.numpy.arctanh")(x, /)
[`atanh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atanh.html#jax.numpy.atanh "jax.numpy.atanh")(x, /)

[`sinc`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sinc.html#jax.numpy.sinc "jax.numpy.sinc")(x, /)
: `sinc(x) = sin(pi x) / (pi x)`.

[`deg2rad`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.deg2rad.html#jax.numpy.deg2rad "jax.numpy.deg2rad")(x, /)
[`radians`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.radians.html#jax.numpy.radians "jax.numpy.radians")(x, /): Convert degree to radian.

[`degrees`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.degrees.html#jax.numpy.degrees "jax.numpy.degrees")(x, /)
[`rad2deg`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.rad2deg.html#jax.numpy.rad2deg "jax.numpy.rad2deg")(x, /): Convert radian to degree.

[`angle`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.angle.html#jax.numpy.angle "jax.numpy.angle")(z[, deg])
: angle of a complex (like `carg()`).

[`real`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.real.html#jax.numpy.real "jax.numpy.real")(val, /)
: real part.

[`imag`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.imag.html#jax.numpy.imag "jax.numpy.imag")(val, /)
: imaginary part.

[`conjugate`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.conjugate.html#jax.numpy.conjugate "jax.numpy.conjugate")(x, /)
[`conj`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.conj.html#jax.numpy.conj "jax.numpy.conj")(x, /)
: complex conjugate.

#### Arithmetic

[`add`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.add.html#jax.numpy.add "jax.numpy.add")
[`subtract`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.subtract.html#jax.numpy.subtract "jax.numpy.subtract")
[`multiply`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.multiply.html#jax.numpy.multiply "jax.numpy.multiply")

[`divide`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.divide.html#jax.numpy.divide "jax.numpy.divide")(x1, x2, /)
[`true_divide`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.true_divide.html#jax.numpy.true_divide "jax.numpy.true_divide")(x1, x2, /)
: divide of float/complex/etc.

[`floor_divide`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.floor_divide.html#jax.numpy.floor_divide "jax.numpy.floor_divide")(x1, x2, /)
: integer divide (but always round down; careful with negative quotients) or float divide with round down). Eg., `floor_divide([-10], [9]) == [-2])`.

[`remainder`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.remainder.html#jax.numpy.remainder "jax.numpy.remainder")(x1, x2, /)
[`mod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.mod.html#jax.numpy.mod "jax.numpy.mod")(x1, x2, /)
: tricky on negative number rounding.

[`fmod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fmod.html#jax.numpy.fmod "jax.numpy.fmod")(x1, x2, /)
: floating-point modulo.


[`modf`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.modf.html#jax.numpy.modf "jax.numpy.modf")(x, /[, out])
: returns two arrays: fractional results, then integral results.


[`divmod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.divmod.html#jax.numpy.divmod "jax.numpy.divmod")(x1, x2, /)
: warning: always round down.

[`reciprocal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.reciprocal.html#jax.numpy.reciprocal "jax.numpy.reciprocal")(x, /)
: `1 / x`.

[`float_power`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.float_power.html#jax.numpy.float_power "jax.numpy.float_power")(x, y, /)
: base x, power y.

[`power`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.power.html#jax.numpy.power "jax.numpy.power")(x1, x2, /)
[`pow`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.pow.html#jax.numpy.pow "jax.numpy.pow")(x1, x2, /)
: base x1, power x2. Can be integer typed.

[`square`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.square.html#jax.numpy.square "jax.numpy.square")(x, /)

[`sqrt`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sqrt.html#jax.numpy.sqrt "jax.numpy.sqrt")(x, /)


[`cbrt`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cbrt.html#jax.numpy.cbrt "jax.numpy.cbrt")(x, /)
: cubic root

[`gcd`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.gcd.html#jax.numpy.gcd "jax.numpy.gcd")(x1, x2)
: greatest common divisor.

[`lcm`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.lcm.html#jax.numpy.lcm "jax.numpy.lcm")(x1, x2)
: least common multiplier.

[`hypot`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hypot.html#jax.numpy.hypot "jax.numpy.hypot")(x1, x2, /): hypotenuse (assuming `x1` and `x2` are sides of the right angle).

[`ldexp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ldexp.html#jax.numpy.ldexp "jax.numpy.ldexp")(x1, x2, /): x1 * (2**x2).

#### Rounding

[`floor`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.floor.html#jax.numpy.floor "jax.numpy.floor")(x, /)
: round down.

[`ceil`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ceil.html#jax.numpy.ceil "jax.numpy.ceil")(x, /)
: round up.

[`rint`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.rint.html#jax.numpy.rint "jax.numpy.rint")(x, /)
: round to nearest.

[`round`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.round.html#jax.numpy.round "jax.numpy.round")(a[, decimals, out])
[`around`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.around.html#jax.numpy.around "jax.numpy.around")(a[, decimals, out])
: round **evenly**. See [notes](https://numpy.org/doc/stable/reference/generated/numpy.round.html#numpy.round) from numpy original doc.

[`trunc`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.trunc.html#jax.numpy.trunc "jax.numpy.trunc")(x)
: Round towards zero.

#### Logic, Bit, Compare-ish

[`equal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.equal.html#jax.numpy.equal "jax.numpy.equal")(x, y, /)
[`not_equal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.not_equal.html#jax.numpy.not_equal "jax.numpy.not_equal")(x, y, /)
[`less`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.less.html#jax.numpy.less "jax.numpy.less")(x, y, /)
[`less_equal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.less_equal.html#jax.numpy.less_equal "jax.numpy.less_equal")(x, y, /)
[`greater`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.greater.html#jax.numpy.greater "jax.numpy.greater")(x, y, /)
[`greater_equal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.greater_equal.html#jax.numpy.greater_equal "jax.numpy.greater_equal")(x, y, /)

[`maximum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.maximum.html#jax.numpy.maximum "jax.numpy.maximum")
[`fmax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fmax.html#jax.numpy.fmax "jax.numpy.fmax")(x1, x2)
[`minimum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.minimum.html#jax.numpy.minimum "jax.numpy.minimum")
[`fmin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fmin.html#jax.numpy.fmin "jax.numpy.fmin")(x1, x2)
: element-wise max/min. Different from `max/min`, which reduces along some axis.

[`clip`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.clip.html#jax.numpy.clip "jax.numpy.clip")([arr, min, max])
: bound each element to the given [min, max] range.

[`logical_not`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logical_not.html#jax.numpy.logical_not "jax.numpy.logical_not")(x, /)
[`logical_and`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logical_and.html#jax.numpy.logical_and "jax.numpy.logical_and")
[`logical_or`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logical_or.html#jax.numpy.logical_or "jax.numpy.logical_or")
[`logical_xor`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logical_xor.html#jax.numpy.logical_xor "jax.numpy.logical_xor")

[`bitwise_not`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bitwise_not.html#jax.numpy.bitwise_not "jax.numpy.bitwise_not")(x, /)
[`bitwise_invert`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bitwise_invert.html#jax.numpy.bitwise_invert "jax.numpy.bitwise_invert")(x, /)
[`invert`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.invert.html#jax.numpy.invert "jax.numpy.invert")(x, /)
: bitwise inversion.

[`bitwise_and`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bitwise_and.html#jax.numpy.bitwise_and "jax.numpy.bitwise_and")
[`bitwise_or`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bitwise_or.html#jax.numpy.bitwise_or "jax.numpy.bitwise_or")
[`bitwise_xor`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bitwise_xor.html#jax.numpy.bitwise_xor "jax.numpy.bitwise_xor")

[`bitwise_left_shift`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bitwise_left_shift.html#jax.numpy.bitwise_left_shift "jax.numpy.bitwise_left_shift")(x, y, /)
[`left_shift`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.left_shift.html#jax.numpy.left_shift "jax.numpy.left_shift")(x, y, /)
[`bitwise_right_shift`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bitwise_right_shift.html#jax.numpy.bitwise_right_shift "jax.numpy.bitwise_right_shift")(x1, x2, /)
[`right_shift`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.right_shift.html#jax.numpy.right_shift "jax.numpy.right_shift")(x1, x2, /)

[`bitwise_count`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bitwise_count.html#jax.numpy.bitwise_count "jax.numpy.bitwise_count")(x, /)
: count bit 1s. Eg., 255 has 8 bits of 1s.

[`isnan`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isnan.html#jax.numpy.isnan "jax.numpy.isnan")(x, /)
[`isfinite`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isfinite.html#jax.numpy.isfinite "jax.numpy.isfinite")(x, /)
[`isinf`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isinf.html#jax.numpy.isinf "jax.numpy.isinf")(x, /)
[`isneginf`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isneginf.html#jax.numpy.isneginf "jax.numpy.isneginf")(x, /[, out])
[`isposinf`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isposinf.html#jax.numpy.isposinf "jax.numpy.isposinf")(x, /[, out])

[`isreal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isreal.html#jax.numpy.isreal "jax.numpy.isreal")(x)

#### Others

[`heaviside`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.heaviside.html#jax.numpy.heaviside "jax.numpy.heaviside")(x1, x2, /)
: the "heaviside" function: 0 if x1 < 0; 1 if x1 > 0; x2 if x1 is just 0.

[`i0`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.i0.html#jax.numpy.i0 "jax.numpy.i0")(x)
: some version of modified Bessel function (related to multiplication of gaussians).

## Reducing functions

### Logic, Compare-ish

[`apply_along_axis`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.apply_along_axis.html#jax.numpy.apply_along_axis "jax.numpy.apply_along_axis")(func1d, axis, arr, *args, ...)
: general reduction on 1 axis.

[`apply_over_axes`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.apply_over_axes.html#jax.numpy.apply_over_axes "jax.numpy.apply_over_axes")(func, a, axes)
: generlizes `apply_along_axis` even more by allowing multiple axes given.

[`all`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.all.html#jax.numpy.all "jax.numpy.all")(a[, axis, out, keepdims, where])
[`any`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.any.html#jax.numpy.any "jax.numpy.any")(a[, axis, out, keepdims, where])

[`array_equal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_equal.html#jax.numpy.array_equal "jax.numpy.array_equal")(a1, a2[, equal_nan])
[`array_equiv`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_equiv.html#jax.numpy.array_equiv "jax.numpy.array_equiv")(a1, a2)
: all are respectively pairwise equal.

[`allclose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.allclose.html#jax.numpy.allclose "jax.numpy.allclose")(a, b[, rtol, atol, equal_nan])
: all are respectively pairwise close.

### Arithmetic

[`einsum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.einsum.html#jax.numpy.einsum "jax.numpy.einsum")(subscripts, /, *operands[, out, ...])
: the swiss army knife of matrix-multiplication-like operatoins.

[`einsum_path`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.einsum_path.html#jax.numpy.einsum_path "jax.numpy.einsum_path")(subscripts, /, *operands[, optimize])
: some extra utility to help `einsum`?

[`matmul`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul")(a, b, *[, precision, ...])
: matrix multiplication.

[`dot`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dot.html#jax.numpy.dot "jax.numpy.dot")(a, b, *[, precision, ...])
: dot product. When dealing with higher dimensional arrays, look suspiciously different from `matmul` in very subtle ways.

[`inner`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.inner.html#jax.numpy.inner "jax.numpy.inner")(a, b, *[, precision, ...])
: inner product.

[`outer`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.outer.html#jax.numpy.outer "jax.numpy.outer")(a, b[, out])
: outer product (a[i] * b[j])_ij.

[`matvec`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matvec.html#jax.numpy.matvec "jax.numpy.matvec")(x1, x2, /)
: matrix times vector.

[`cross`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cross.html#jax.numpy.cross "jax.numpy.cross")(a, b[, axisa, axisb, axisc, axis])
: the `axis*` params designate along which axis of `a`, `b` and the return value `c` to perform cross product. Inputs 3D along the axes result in output 3D; input 2D result in output 1D.

[`sum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum")(a[, axis, dtype, out, keepdims, ...])
: sum along the axis.

[`prod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.prod.html#jax.numpy.prod "jax.numpy.prod")(a[, axis, dtype, out, keepdims, ...])
: product along the axis.

## Array

[`array`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array.html#jax.numpy.array "jax.numpy.array")(object[, dtype, copy, order, ndmin, ...])
[`asarray`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.asarray.html#jax.numpy.asarray "jax.numpy.asarray")(a[, dtype, order, copy, device, ...])
: Array construction from `object`, where the object can be things like python native numbers, lists, etc.
Two versions have some subtle differences.

[`astype`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.astype.html#jax.numpy.astype "jax.numpy.astype")(x, dtype, /, *[, copy, device])
: array type conversion.

[`copy`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.copy.html#jax.numpy.copy "jax.numpy.copy")(a[, order])
: copy an array.

[`arange`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange")(start[, stop, step, dtype, device, ...])
: Like python `range()`.

[`atleast_1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_1d.html#jax.numpy.atleast_1d "jax.numpy.atleast_1d")(*arys)
[`atleast_2d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_2d.html#jax.numpy.atleast_2d "jax.numpy.atleast_2d")(*arys)
[`atleast_3d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_3d.html#jax.numpy.atleast_3d "jax.numpy.atleast_3d")(*arys)
: trying to increase the dimension of tensor/array without increase the number of elements, to 1/2/3.
Feels sloppy... Shouldn't the input dimension be known exactly and increase dimension with things like `broadcast_to`?

### Array manipulation

[`append`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.append.html#jax.numpy.append "jax.numpy.append")(arr, values[, axis])

[`array_split`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_split.html#jax.numpy.array_split "jax.numpy.array_split")(ary, indices_or_sections[, axis])

[`insert`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.insert.html#jax.numpy.insert "jax.numpy.insert")(arr, obj, values[, axis])
: insert entries into the array.

[`delete`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.delete.html#jax.numpy.delete "jax.numpy.delete")(arr, obj[, axis, assume_unique_indices])
: delete entries from the array.

[`compress`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.compress.html#jax.numpy.compress "jax.numpy.compress")(condition, a[, axis, size, ...])
: delete/filter array `a` along with axis, according to boolean array `condition`. True for keeping, False for removing.

[`flip`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.flip.html#jax.numpy.flip "jax.numpy.flip")(m[, axis])
[`fliplr`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fliplr.html#jax.numpy.fliplr "jax.numpy.fliplr")(m)
[`flipud`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.flipud.html#jax.numpy.flipud "jax.numpy.flipud")(m)
: flip along an axis. `fliplr` and `flipud` are fixed on axis=1/0 respectively.

[`concat`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concat.html#jax.numpy.concat "jax.numpy.concat")(arrays, /, *[, axis])
[`concatenate`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate")(arrays[, axis, dtype])
: concatenate the given sequence of arrays along the given axis.

[`c_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.c_.html#jax.numpy.c_ "jax.numpy.c_")
: concatenate a sequence of arrays along the last axis.
Seems the interface can supply an "instruction" as the first argument.
TODO: check again.

### Matrix-like

[`matrix_transpose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matrix_transpose.html#jax.numpy.matrix_transpose "jax.numpy.matrix_transpose")(x, /)
: when the array is higher dimension, transpose the last two dimensions.

[`zeros`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.zeros.html#jax.numpy.zeros "jax.numpy.zeros")(shape[, dtype, device, out_sharding])
[`empty`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.empty.html#jax.numpy.empty "jax.numpy.empty")(shape[, dtype, device, out_sharding])
[`zeros_like`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.zeros_like.html#jax.numpy.zeros_like "jax.numpy.zeros_like")(a[, dtype, shape, device, ...])
[`empty_like`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.empty_like.html#jax.numpy.empty_like "jax.numpy.empty_like")(prototype[, dtype, shape, device])
: create zeros.

[`ones`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ones.html#jax.numpy.ones "jax.numpy.ones")(shape[, dtype, device, out_sharding])
[`ones_like`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ones_like.html#jax.numpy.ones_like "jax.numpy.ones_like")(a[, dtype, shape, device, ...])
: create ones.

[`full`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.full.html#jax.numpy.full "jax.numpy.full")(shape, fill_value[, dtype, device])
[`full_like`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.full_like.html#jax.numpy.full_like "jax.numpy.full_like")(a, fill_value[, dtype, shape, device])
: array full of `fill_value`.

[`eye`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.eye.html#jax.numpy.eye "jax.numpy.eye")(N[, M, k, dtype, device])
[`identity`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.identity.html#jax.numpy.identity "jax.numpy.identity")(n[, dtype])
: identity matrix. `identity` is more rigorously/narrowly defined.

[`diag`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.diag.html#jax.numpy.diag "jax.numpy.diag")(v[, k])
: actually two functions under one definition. If the input `v` is 1D, create a matrix with diagnal elements from `v`; if the input `v` is a matrix, extract its diagnal and return a 1D array. `k` is the offset. (But why? Is function name a scarce resource?)

[`diagonal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.diagonal.html#jax.numpy.diagonal "jax.numpy.diagonal")(a[, offset, axis1, axis2])
: use `axis1` and `axis2` to designate the array's "matrix" dimensions, then extract diagnals.

[`diagflat`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.diagflat.html#jax.numpy.diagflat "jax.numpy.diagflat")(v[, k])
: similar to `diag`'s 1D->2D version, but `v` can be multi-dimensional and will be flattened into 1D. (Why?)

[`fill_diagonal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fill_diagonal.html#jax.numpy.fill_diagonal "jax.numpy.fill_diagonal")(a, val[, wrap, inplace])
: return a copy, with the diagnal elements overwritten (off-dignal elements are kept as is).

[`diag_indices`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.diag_indices.html#jax.numpy.diag_indices "jax.numpy.diag_indices")(n[, ndim])
[`diag_indices_from`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.diag_indices_from.html#jax.numpy.diag_indices_from "jax.numpy.diag_indices_from")(arr)
: indices of an array that point to diagnal elements.

[`block`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.block.html#jax.numpy.block "jax.numpy.block")(arrays)
: array from "sub"-arrays like gluing sub-matrices into a matrix.

[`kron`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.kron.html#jax.numpy.kron "jax.numpy.kron")(a, b)
: Kronecker product.

## Shape manipulation

[`reshape`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.reshape.html#jax.numpy.reshape "jax.numpy.reshape")(a, shape[, order, copy, out_sharding])
: well, very drastic and dramatic approach...

[`broadcast_to`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.broadcast_to.html#jax.numpy.broadcast_to "jax.numpy.broadcast_to")(array, shape, *[, out_sharding])

[`broadcast_shapes`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.broadcast_shapes.html#jax.numpy.broadcast_shapes "jax.numpy.broadcast_shapes")(*shapes)

[`broadcast_arrays`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.broadcast_arrays.html#jax.numpy.broadcast_arrays "jax.numpy.broadcast_arrays")(*args)

[`column_stack`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.column_stack.html#jax.numpy.column_stack "jax.numpy.column_stack")(tup)
: stack along axis 1, basically.

[`repeat`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.repeat.html#jax.numpy.repeat "jax.numpy.repeat")(a, repeats[, axis, ...])
: extend dimension by repeating.

[`resize`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.resize.html#jax.numpy.resize "jax.numpy.resize")(a, new_shape)
: resize an array. Larger size along an axis will be filled with repetition; smaller size causes truncation.

[`expand_dims`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.expand_dims.html#jax.numpy.expand_dims "jax.numpy.expand_dims")(a, axis)
: expand 1 dimension along axis (i.e., x -> [x]).

[`hsplit`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hsplit.html#jax.numpy.hsplit "jax.numpy.hsplit")(ary, indices_or_sections)
[`hstack`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hstack.html#jax.numpy.hstack "jax.numpy.hstack")(tup[, dtype])
: horizontally (axis 1) split and stack.

[`dsplit`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dsplit.html#jax.numpy.dsplit "jax.numpy.dsplit")(ary, indices_or_sections)
[`dstack`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dstack.html#jax.numpy.dstack "jax.numpy.dstack")(tup[, dtype])
: depth-wise (axis 2) split and stack.

### Debug-related

[`array_repr`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_repr.html#jax.numpy.array_repr "jax.numpy.array_repr")(arr[, max_line_width, precision, ...])

[`array_str`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_str.html#jax.numpy.array_str "jax.numpy.array_str")(a[, max_line_width, precision, ...])

## Selection, Sorting

[`argwhere`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argwhere.html#jax.numpy.argwhere "jax.numpy.argwhere")(a, *[, size, fill_value])
: returns an array (dynamic-sized if `size` is not given) of indices of non-zero elements in `a`, each index is n dimensions if `a` is an n-order tensor/array.

[`nonzero`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero")(a, *[, size, fill_value])
: also returns indices like `argwhere`, but indices are parallel n arrays, each array corresponds to one dimension of `a`'s shape.

[`sort`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sort.html#jax.numpy.sort "jax.numpy.sort")(a[, axis, kind, order, stable, descending])
: sort along axis.

[`argpartition`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argpartition.html#jax.numpy.argpartition "jax.numpy.argpartition")(a, kth[, axis])

[`argsort`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argsort.html#jax.numpy.argsort "jax.numpy.argsort")(a[, axis, kind, order, stable, ...])

## Types

[`can_cast`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.can_cast.html#jax.numpy.can_cast "jax.numpy.can_cast")(from_, to[, casting])
: NOT ASYNC. Help function to test whether one type can be casted to another. Just returns True/False, simple value, not in the array container.

[`dtype`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dtype.html#jax.numpy.dtype "jax.numpy.dtype")(dtype[, align, copy])
: NOT ASYNC. A type object. TODO: take another look.

[`generic`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.generic.html#jax.numpy.generic "jax.numpy.generic")()
: base class of "most" scalar types.

[`bool_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bool_.html#jax.numpy.bool_ "jax.numpy.bool_")
: alias of python `bool`.

[`uint8`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.uint8.html#jax.numpy.uint8 "jax.numpy.uint8")(x)
[`int8`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.int8.html#jax.numpy.int8 "jax.numpy.int8")(x)

[`uint16`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.uint16.html#jax.numpy.uint16 "jax.numpy.uint16")(x)
[`int16`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.int16.html#jax.numpy.int16 "jax.numpy.int16")(x)

[`uint32`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.uint32.html#jax.numpy.uint32 "jax.numpy.uint32")(x)
[`int32`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.int32.html#jax.numpy.int32 "jax.numpy.int32")(x)

[`uint64`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.uint64.html#jax.numpy.uint64 "jax.numpy.uint64")(x)
[`uint`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.uint.html#jax.numpy.uint "jax.numpy.uint")
[`int64`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.int64.html#jax.numpy.int64 "jax.numpy.int64")(x)
[`int_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.int_.html#jax.numpy.int_ "jax.numpy.int_")
: 64-bit integers.

[`float16`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.float16.html#jax.numpy.float16 "jax.numpy.float16")(x)
: 16-bit float.

[`float32`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.float32.html#jax.numpy.float32 "jax.numpy.float32")(x)
[`single`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.single.html#jax.numpy.single "jax.numpy.single")
: 32-bit float.

[`double`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.double.html#jax.numpy.double "jax.numpy.double")
[`float_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.float_.html#jax.numpy.float_ "jax.numpy.float_")
[`float64`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.float64.html#jax.numpy.float64 "jax.numpy.float64")(x)
: 64-bit float. (Why is "float_" 64 bit? Very confusing for people coming from C-like languages.)

[`complex64`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.complex64.html#jax.numpy.complex64 "jax.numpy.complex64")(x)
[`csingle`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.csingle.html#jax.numpy.csingle "jax.numpy.csingle")
: numpy's complex64, with 32-bit real and 32-bit imaginary.

[`complex128`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.complex128.html#jax.numpy.complex128 "jax.numpy.complex128")(x)
[`cdouble`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cdouble.html#jax.numpy.cdouble "jax.numpy.cdouble")
[`complex_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.complex_.html#jax.numpy.complex_ "jax.numpy.complex_")
: numpy's complex128, with 64-bit real and 64-bit imaginary.

### Abstract base classes

[`flexible`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.flexible.html#jax.numpy.flexible "jax.numpy.flexible")()
: ABC of all scalar types "without predefined length".

[`number`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.number.html#jax.numpy.number "jax.numpy.number")()
: ABC of all numeric scalars.

[`integer`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.integer.html#jax.numpy.integer "jax.numpy.integer")()
: ABC of integers.

[`unsignedinteger`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unsignedinteger.html#jax.numpy.unsignedinteger "jax.numpy.unsignedinteger")()
: ABC of unsigned integers.

[`signedinteger`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.signedinteger.html#jax.numpy.signedinteger "jax.numpy.signedinteger")()
: ABC of signed integers.

[`inexact`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.inexact.html#jax.numpy.inexact "jax.numpy.inexact")()
: ABC of all scalar types whose representations are not exact in its range (like floating-point numbers).

[`floating`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.floating.html#jax.numpy.floating "jax.numpy.floating")()
: ABC of all floating-point numbers.

[`complexfloating`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.complexfloating.html#jax.numpy.complexfloating "jax.numpy.complexfloating")()
: ABC of all float-point-based complex numbers.

[`character`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.character.html#jax.numpy.character "jax.numpy.character")()
: ABC of, "character string".

## Advanced calcuation

[`gradient`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.gradient.html#jax.numpy.gradient "jax.numpy.gradient")(f, *varargs[, axis, edge_order])
: gradient.

[`convolve`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.convolve.html#jax.numpy.convolve "jax.numpy.convolve")(a, v[, mode, precision, ...])
: convolution (but only for 1d arrays? no tensors?).

[`correlate`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.correlate.html#jax.numpy.correlate "jax.numpy.correlate")(a, v[, mode, precision, ...])
: a construction looking like a variant of `convolve`. TODO: take another look.

[`diff`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.diff.html#jax.numpy.diff "jax.numpy.diff")(a[, n, axis, prepend, append])
: n'th order difference, along axis.

[`ediff1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ediff1d.html#jax.numpy.ediff1d "jax.numpy.ediff1d")(ary[, to_end, to_begin])
: a different, weaker version of `diff` but with a subtly different interface (append/prepend happens after not before diffing).

### Statistics

[`cumulative_sum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cumulative_sum.html#jax.numpy.cumulative_sum "jax.numpy.cumulative_sum")(x, /, *[, axis, dtype, ...])
[`cumsum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cumsum.html#jax.numpy.cumsum "jax.numpy.cumsum")(a[, axis, dtype, out])
[`cumulative_prod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cumulative_prod.html#jax.numpy.cumulative_prod "jax.numpy.cumulative_prod")(x, /, *[, axis, dtype, ...])
[`cumprod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cumprod.html#jax.numpy.cumprod "jax.numpy.cumprod")(a[, axis, dtype, out])
: cummulative (prefix) sum/prod along the given axis.

[`average`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.average.html#jax.numpy.average "jax.numpy.average")(a[, axis, weights, returned, keepdims])
[`mean`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.mean.html#jax.numpy.mean "jax.numpy.mean")(a[, axis, dtype, out, keepdims, where])

[`median`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.median.html#jax.numpy.median "jax.numpy.median")(a[, axis, out, overwrite_input, keepdims])

[`size`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.size.html#jax.numpy.size "jax.numpy.size")(a[, axis])
: along the axis, simply count the elements.

[`count_nonzero`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.count_nonzero.html#jax.numpy.count_nonzero "jax.numpy.count_nonzero")(a[, axis, keepdims])
: along the axis, count how many nonzeros.

[`max`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.max.html#jax.numpy.max "jax.numpy.max")(a[, axis, out, keepdims, initial, where])
[`amax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.amax.html#jax.numpy.amax "jax.numpy.amax")(a[, axis, out, keepdims, initial, where])
[`min`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.min.html#jax.numpy.min "jax.numpy.min")(a[, axis, out, keepdims, initial, where])
[`amin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.amin.html#jax.numpy.amin "jax.numpy.amin")(a[, axis, out, keepdims, initial, where])
: taking max/min along axis.

[`argmax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmax.html#jax.numpy.argmax "jax.numpy.argmax")(a[, axis, out, keepdims])
[`argmin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmin.html#jax.numpy.argmin "jax.numpy.argmin")(a[, axis, out, keepdims])
: index version of max/min.

[`percentile`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.percentile.html#jax.numpy.percentile "jax.numpy.percentile")(a, q[, axis, out, ...])

[`quantile`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.quantile.html#jax.numpy.quantile "jax.numpy.quantile")(a, q[, axis, out, overwrite_input, ...])

### Versions ignoring NaN

[`nanprod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanprod.html#jax.numpy.nanprod "jax.numpy.nanprod")(a[, axis, dtype, out, keepdims, ...])

[`nansum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nansum.html#jax.numpy.nansum "jax.numpy.nansum")(a[, axis, dtype, out, keepdims, ...])

[`nanmean`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmean.html#jax.numpy.nanmean "jax.numpy.nanmean")(a[, axis, dtype, out, keepdims, where])

[`nanmedian`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmedian.html#jax.numpy.nanmedian "jax.numpy.nanmedian")(a[, axis, out, overwrite_input, ...])

[`nanmax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmax.html#jax.numpy.nanmax "jax.numpy.nanmax")(a[, axis, out, keepdims, initial, where])

[`nanmin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmin.html#jax.numpy.nanmin "jax.numpy.nanmin")(a[, axis, out, keepdims, initial, where])

[`nanargmax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanargmax.html#jax.numpy.nanargmax "jax.numpy.nanargmax")(a[, axis, out, keepdims])

[`nanargmin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanargmin.html#jax.numpy.nanargmin "jax.numpy.nanargmin")(a[, axis, out, keepdims])

[`nancumprod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nancumprod.html#jax.numpy.nancumprod "jax.numpy.nancumprod")(a[, axis, dtype, out])

[`nancumsum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nancumsum.html#jax.numpy.nancumsum "jax.numpy.nancumsum")(a[, axis, dtype, out])

[`nanpercentile`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanpercentile.html#jax.numpy.nanpercentile "jax.numpy.nanpercentile")(a, q[, axis, out, ...])

[`nanquantile`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanquantile.html#jax.numpy.nanquantile "jax.numpy.nanquantile")(a, q[, axis, out, ...])

[`nanstd`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanstd.html#jax.numpy.nanstd "jax.numpy.nanstd")(a[, axis, dtype, out, ddof, ...])

[`nanvar`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanvar.html#jax.numpy.nanvar "jax.numpy.nanvar")(a[, axis, dtype, out, ddof, ...])

### More

[`cov`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cov.html#jax.numpy.cov "jax.numpy.cov")(m[, y, rowvar, bias, ddof, fweights, ...])
: correlation matrix.

[`corrcoef`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.corrcoef.html#jax.numpy.corrcoef "jax.numpy.corrcoef")(x[, y, rowvar, dtype])
: Pearson correlation coefficients.

[`histogram`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogram.html#jax.numpy.histogram "jax.numpy.histogram")(a[, bins, range, weights, density])
[`histogram2d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogram2d.html#jax.numpy.histogram2d "jax.numpy.histogram2d")(x, y[, bins, range, weights, ...])
[`histogramdd`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogramdd.html#jax.numpy.histogramdd "jax.numpy.histogramdd")(sample[, bins, range, weights, ...])
: 1D, 2D and N-dim histogram.

[`histogram_bin_edges`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogram_bin_edges.html#jax.numpy.histogram_bin_edges "jax.numpy.histogram_bin_edges")(a[, bins, range, weights])
: edges of bins of histogram.

## Misc (TODO: revisit all and classify properly)

[`bincount`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bincount.html#jax.numpy.bincount "jax.numpy.bincount")(x[, weights, minlength, length])

[`digitize`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.digitize.html#jax.numpy.digitize "jax.numpy.digitize")(x, bins[, right, method])

### Window

[`bartlett`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bartlett.html#jax.numpy.bartlett "jax.numpy.bartlett")(M)

[`blackman`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.blackman.html#jax.numpy.blackman "jax.numpy.blackman")(M)

[`hamming`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hamming.html#jax.numpy.hamming "jax.numpy.hamming")(M)

[`hanning`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hanning.html#jax.numpy.hanning "jax.numpy.hanning")(M)

[`kaiser`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.kaiser.html#jax.numpy.kaiser "jax.numpy.kaiser")(M, beta)

