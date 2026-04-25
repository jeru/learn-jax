# Polynomials and Sets

A little bit exotic in the world of model training...

## Polynomial

Polynomial is represented as a 1D array (of its coefficients). Note that it is reverse-indexed, eg., [1, 2, 3] is `1 * x**2 + 2 * x + 3`.

[`polyval`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polyval.html#jax.numpy.polyval "jax.numpy.polyval")(p, x, *[, unroll])
: evaluate the polynomial at `x`.

[`poly`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.poly.html#jax.numpy.poly "jax.numpy.poly")(seq_of_zeros)
: given the roots, construct a polynomial.

[`roots`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.roots.html#jax.numpy.roots "jax.numpy.roots")(p, *[, strip_zeros])
: given a polynomial, compute roots.

[`polyadd`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polyadd.html#jax.numpy.polyadd "jax.numpy.polyadd")(a1, a2)
[`polysub`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polysub.html#jax.numpy.polysub "jax.numpy.polysub")(a1, a2)
: add/subtract two polynomials. Difference from elementwise add/subtract: the two polynomials can have different lengths.

[`polymul`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polymul.html#jax.numpy.polymul "jax.numpy.polymul")(a1, a2, *[, trim_leading_zeros])
: multiple of two polynomials.

[`polydiv`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polydiv.html#jax.numpy.polydiv "jax.numpy.polydiv")(u, v, *[, trim_leading_zeros])
: divide and returns the quotient and remainder.

[`polyder`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polyder.html#jax.numpy.polyder "jax.numpy.polyder")(p[, m])
[`polyint`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polyint.html#jax.numpy.polyint "jax.numpy.polyint")(p[, m, k])
: derivative/integral of polynomial.

[`polyfit`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polyfit.html#jax.numpy.polyfit "jax.numpy.polyfit")(x, y, deg[, rcond, full, w, cov])
: create a polynomial to fit data `(x, y)`, minimized by least square loss.

## Set (the set-theory set)

[`intersect1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.intersect1d.html#jax.numpy.intersect1d "jax.numpy.intersect1d")(ar1, ar2[, assume_unique, ...])

[`union1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.union1d.html#jax.numpy.union1d "jax.numpy.union1d")(ar1, ar2, *[, size, fill_value])

[`setdiff1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.setdiff1d.html#jax.numpy.setdiff1d "jax.numpy.setdiff1d")(ar1, ar2[, assume_unique, size, ...])

[`setxor1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.setxor1d.html#jax.numpy.setxor1d "jax.numpy.setxor1d")(ar1, ar2[, assume_unique, size, ...])

