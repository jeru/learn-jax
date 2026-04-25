# Matrix multiplication plus

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

[`sum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum")(a[, axis, dtype, out, keepdims, ...])
: sum along the axis.

[`tensordot`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tensordot.html#jax.numpy.tensordot "jax.numpy.tensordot")(a, b[, axes, precision, ...])
: generalized matrix multiplication with tensor tastes.
Can all be implemented with `einsum` anyway, so this one is probably to please the physics folks (especially, general relativity).

## Not covered by einsum

[`cross`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cross.html#jax.numpy.cross "jax.numpy.cross")(a, b[, axisa, axisb, axisc, axis])
: the `axis*` params designate along which axis of `a`, `b` and the return value `c` to perform cross product. Inputs 3D along the axes result in output 3D; input 2D result in output 1D.

[`vdot`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vdot.html#jax.numpy.vdot "jax.numpy.vdot")(a, b, *[, precision, ...])
[`vecdot`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vecdot.html#jax.numpy.vecdot "jax.numpy.vecdot")(x1, x2, /, *[, axis, precision, ...])
[`vecmat`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vecmat.html#jax.numpy.vecmat "jax.numpy.vecmat")(x1, x2, /)
: basically `dot(conj(a), b)`, either 1D x 1D or higher dimensionally batched (allow broadcasting). Quantum mechanics folks love these.
