# Elementwise functions

### From an ordinary function to something parallel

There is NO performance concern on python language overhead here on the underlying (ordinary) function here: all these functions will have to go through `jax.jit` before going to GPU; and `jax.jit` aren't able to deal arbitrary python bytecode. The wrapped functions must be "green".

[`ufunc`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc")(func, /, nin, nout, *[, name, nargs, ...])
[`frompyfunc`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.frompyfunc.html#jax.numpy.frompyfunc "jax.numpy.frompyfunc")(func, /, nin, nout, *[, identity])
: one is class, one is factory method.
They will add a bunch extra functions like `reduce`, `accumulate` to the class.

[`vectorize`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vectorize.html#jax.numpy.vectorize "jax.numpy.vectorize")(pyfunc, *[, excluded, signature])
: a decorator that wraps a function over with [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html).

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

[`positive`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.positive.html#jax.numpy.positive "jax.numpy.positive")(x, /)
[`negative`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.negative.html#jax.numpy.negative "jax.numpy.negative")
: make `x` as `+x` or `-x`. For plus, it's basically a NOOP for most number types.

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

[`frexp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.frexp.html#jax.numpy.frexp "jax.numpy.frexp")(x, /)
[`ldexp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ldexp.html#jax.numpy.ldexp "jax.numpy.ldexp")(x1, x2, /): `frexp` splits a float into two parts: the part with absolute value between 0.5 (inclusive) and 1 (exclusive); the part of exponent of power 2. `ldexp` restores the two into one. Eg., `frexp(-0.3) == (-0.6, -1)` because `-0.3 == -0.6 * 2**-1`, and `ldexp(-0.6, -1) == -0.3`. (Original documentation, as of Apr 2026, said the first part will also be within -1 and 1, so underspecified.)

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

[`where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html#jax.numpy.where "jax.numpy.where")(condition[, x, y, size, fill_value])
: the asynchronized and multidimensional `if-then-else`.
_Note_: when `condition` is a single boolean (might be asynchronous), consider using `jax.lax.cond` instead, whose `x` and `y` branches are functions so can be run lazily.

[`place`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.place.html#jax.numpy.place "jax.numpy.place")(arr, mask, vals, *[, inplace])
: if `mask`, replace `arr` by `vals`. Simplified alternative to `where`.

[`equal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.equal.html#jax.numpy.equal "jax.numpy.equal")(x, y, /)
[`not_equal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.not_equal.html#jax.numpy.not_equal "jax.numpy.not_equal")(x, y, /)
[`less`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.less.html#jax.numpy.less "jax.numpy.less")(x, y, /)
[`less_equal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.less_equal.html#jax.numpy.less_equal "jax.numpy.less_equal")(x, y, /)
[`greater`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.greater.html#jax.numpy.greater "jax.numpy.greater")(x, y, /)
[`greater_equal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.greater_equal.html#jax.numpy.greater_equal "jax.numpy.greater_equal")(x, y, /)

[`isclose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isclose.html#jax.numpy.isclose "jax.numpy.isclose")(a, b[, rtol, atol, equal_nan])

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

[`isreal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isreal.html#jax.numpy.isreal "jax.numpy.isreal")(x)

#### Irregular Floating-Point Values

[`isnan`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isnan.html#jax.numpy.isnan "jax.numpy.isnan")(x, /)
[`isfinite`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isfinite.html#jax.numpy.isfinite "jax.numpy.isfinite")(x, /)
[`isinf`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isinf.html#jax.numpy.isinf "jax.numpy.isinf")(x, /)
[`isneginf`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isneginf.html#jax.numpy.isneginf "jax.numpy.isneginf")(x, /[, out])
[`isposinf`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isposinf.html#jax.numpy.isposinf "jax.numpy.isposinf")(x, /[, out])

[`nan_to_num`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nan_to_num.html#jax.numpy.nan_to_num "jax.numpy.nan_to_num")(x[, copy, nan, posinf, neginf])
: patch nan, posinf, neginf values in `x`.

#### Others

[`interp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.interp.html#jax.numpy.interp "jax.numpy.interp")(x, xp, fp[, left, right, period])
: linear interpolation. Each individual element of `x` is interpolated against the piece-wise linear function defined by 1D arrays `(xp -> fp)`.

[`heaviside`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.heaviside.html#jax.numpy.heaviside "jax.numpy.heaviside")(x1, x2, /)
: the "heaviside" function: 0 if x1 < 0; 1 if x1 > 0; x2 if x1 is just 0.

[`i0`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.i0.html#jax.numpy.i0 "jax.numpy.i0")(x)
: some version of modified Bessel function (related to multiplication of gaussians).

[`nextafter`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nextafter.html#jax.numpy.nextafter "jax.numpy.nextafter")(x, y, /)
: the next floating point value of `x`, towards `y`. Minimally incremented.

[`spacing`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.spacing.html#jax.numpy.spacing "jax.numpy.spacing")(x, /)
: the gap to the nearest but different value.
