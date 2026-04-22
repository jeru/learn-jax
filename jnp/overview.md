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
: `exp(x) - 1`. Direct calculation might have some precision issue when `x` is too close to zero.

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

[`cbrt`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cbrt.html#jax.numpy.cbrt "jax.numpy.cbrt")(x, /)
: cubic root

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

[`max`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.max.html#jax.numpy.max "jax.numpy.max")(a[, axis, out, keepdims, initial, where])
[`amax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.amax.html#jax.numpy.amax "jax.numpy.amax")(a[, axis, out, keepdims, initial, where])
[`min`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.min.html#jax.numpy.min "jax.numpy.min")(a[, axis, out, keepdims, initial, where])
[`amin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.amin.html#jax.numpy.amin "jax.numpy.amin")(a[, axis, out, keepdims, initial, where])
: taking max/min along axis.

[`argmax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmax.html#jax.numpy.argmax "jax.numpy.argmax")(a[, axis, out, keepdims])
[`argmin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmin.html#jax.numpy.argmin "jax.numpy.argmin")(a[, axis, out, keepdims])
: index version of max/min.

[`average`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.average.html#jax.numpy.average "jax.numpy.average")(a[, axis, weights, returned, keepdims])

## Array

[`array`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array.html#jax.numpy.array "jax.numpy.array")(object[, dtype, copy, order, ndmin, ...])
[`asarray`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.asarray.html#jax.numpy.asarray "jax.numpy.asarray")(a[, dtype, order, copy, device, ...])
: Array construction from `object`, where the object can be things like python native numbers, lists, etc.
Two versions have some subtle differences.

[`astype`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.astype.html#jax.numpy.astype "jax.numpy.astype")(x, dtype, /, *[, copy, device])
: array type conversion.

[`arange`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange")(start[, stop, step, dtype, device, ...])
: Like python `range()`.

[`atleast_1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_1d.html#jax.numpy.atleast_1d "jax.numpy.atleast_1d")(*arys)
[`atleast_2d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_2d.html#jax.numpy.atleast_2d "jax.numpy.atleast_2d")(*arys)
[`atleast_3d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_3d.html#jax.numpy.atleast_3d "jax.numpy.atleast_3d")(*arys)
: trying to increase the dimension of tensor/array without increase the number of elements, to 1/2/3.
Feels sloppy... Shouldn't the input dimension be known exactly and increase dimension with things like `broadcast_to`?

#### Array manipulation

[`append`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.append.html#jax.numpy.append "jax.numpy.append")(arr, values[, axis])

[`array_split`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_split.html#jax.numpy.array_split "jax.numpy.array_split")(ary, indices_or_sections[, axis])


### Debug-related

[`array_repr`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_repr.html#jax.numpy.array_repr "jax.numpy.array_repr")(arr[, max_line_width, precision, ...])

[`array_str`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_str.html#jax.numpy.array_str "jax.numpy.array_str")(a[, max_line_width, precision, ...])

## Selection, Sorting

[`argwhere`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argwhere.html#jax.numpy.argwhere "jax.numpy.argwhere")(a, *[, size, fill_value])
: returns an array (dynamic-sized if `size` is not given) of indices of non-zero elements in `a`, each index is n dimensions if `a` is an n-order tensor/array.

[`nonzero`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero")(a, *[, size, fill_value])
: also returns indices like `argwhere`, but indices are parallel n arrays, each array corresponds to one dimension of `a`'s shape.

[`argpartition`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argpartition.html#jax.numpy.argpartition "jax.numpy.argpartition")(a, kth[, axis])

[`argsort`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argsort.html#jax.numpy.argsort "jax.numpy.argsort")(a[, axis, kind, order, stable, ...])

