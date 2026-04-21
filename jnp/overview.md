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
