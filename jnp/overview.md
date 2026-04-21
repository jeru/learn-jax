# jax.numpy functions

Reference page: https://docs.jax.dev/en/latest/jax.numpy.html

## Pointwise functions

### Math

[`abs`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.abs.html#jax.numpy.abs "jax.numpy.abs")(x, /)
[`absolute`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.absolute.html#jax.numpy.absolute "jax.numpy.absolute")(x, /)
: absolute value.

[`fabs`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fabs.html#jax.numpy.fabs "jax.numpy.fabs")(x, /)
: absolute value but real.

#### Trigonometry, Complex, Hyperbolic

[`exp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.exp.html#jax.numpy.exp "jax.numpy.exp")(x, /)

[`exp2`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.exp2.html#jax.numpy.exp2 "jax.numpy.exp2")(x, /)
: base-2.

[`expm1`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.expm1.html#jax.numpy.expm1 "jax.numpy.expm1")(x, /)
: `exp(x) - 1`. Direct calculation might have some precision issue when `x` is too negative.

[`arccos`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arccos.html#jax.numpy.arccos "jax.numpy.arccos")(x, /)
[`acos`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.acos.html#jax.numpy.acos "jax.numpy.acos")(x, /)

[`arcsin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arcsin.html#jax.numpy.arcsin "jax.numpy.arcsin")(x, /)
[`asin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.asin.html#jax.numpy.asin "jax.numpy.asin")(x, /)

[`arctan`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arctan.html#jax.numpy.arctan "jax.numpy.arctan")(x, /)
[`atan`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atan.html#jax.numpy.atan "jax.numpy.atan")(x, /)
[`arctan2`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arctan2.html#jax.numpy.arctan2 "jax.numpy.arctan2")(x1, x2, /)
[`atan2`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atan2.html#jax.numpy.atan2 "jax.numpy.atan2")(x1, x2, /)

[`arccosh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arccosh.html#jax.numpy.arccosh "jax.numpy.arccosh")(x, /)
[`acosh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.acosh.html#jax.numpy.acosh "jax.numpy.acosh")(x, /)

[`arcsinh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arcsinh.html#jax.numpy.arcsinh "jax.numpy.arcsinh")(x, /
[`asinh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.asinh.html#jax.numpy.asinh "jax.numpy.asinh")(x, /)

[`arctanh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arctanh.html#jax.numpy.arctanh "jax.numpy.arctanh")(x, /)
[`atanh`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atanh.html#jax.numpy.atanh "jax.numpy.atanh")(x, /)

[`deg2rad`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.deg2rad.html#jax.numpy.deg2rad "jax.numpy.deg2rad")(x, /)
[`radians`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.radians.html#jax.numpy.radians "jax.numpy.radians")(x, /): Convert degree to radian.

[`degrees`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.degrees.html#jax.numpy.degrees "jax.numpy.degrees")(x, /)
[`rad2deg`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.rad2deg.html#jax.numpy.rad2deg "jax.numpy.rad2deg")(x, /): Convert radian to degree.

[`angle`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.angle.html#jax.numpy.angle "jax.numpy.angle")(z[, deg])
: angle of a complex (like `carg()`).

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

[`modf`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.modf.html#jax.numpy.modf "jax.numpy.modf")(x, /[, out])
: Returns two arrays: fractional results, then integral results.


[`divmod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.divmod.html#jax.numpy.divmod "jax.numpy.divmod")(x1, x2, /)
: warning: always round down.

[`reciprocal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.reciprocal.html#jax.numpy.reciprocal "jax.numpy.reciprocal")(x, /)
: `1 / x`.

[`float_power`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.float_power.html#jax.numpy.float_power "jax.numpy.float_power")(x, y, /)
: base x, power y.



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
