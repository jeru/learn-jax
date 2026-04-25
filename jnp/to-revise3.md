## Set (the set-theory set)

[`intersect1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.intersect1d.html#jax.numpy.intersect1d "jax.numpy.intersect1d")(ar1, ar2[, assume_unique, ...])

[`union1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.union1d.html#jax.numpy.union1d "jax.numpy.union1d")(ar1, ar2, *[, size, fill_value])

[`setdiff1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.setdiff1d.html#jax.numpy.setdiff1d "jax.numpy.setdiff1d")(ar1, ar2[, assume_unique, size, ...])

[`setxor1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.setxor1d.html#jax.numpy.setxor1d "jax.numpy.setxor1d")(ar1, ar2[, assume_unique, size, ...])

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

[`sum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum")(a[, axis, dtype, out, keepdims, ...])
: sum along the axis.

[`prod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.prod.html#jax.numpy.prod "jax.numpy.prod")(a[, axis, dtype, out, keepdims, ...])
: product along the axis.

[`trapezoid`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.trapezoid.html#jax.numpy.trapezoid "jax.numpy.trapezoid")(y[, x, dx, axis])
: integrate by treating `x -> y` as piecewise linear function.

## Array

[`shape`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.shape.html#jax.numpy.shape "jax.numpy.shape")(a)
[`ndim`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndim.html#jax.numpy.ndim "jax.numpy.ndim")(a)
: shape (or just dimension) of an array.

[`array`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array.html#jax.numpy.array "jax.numpy.array")(object[, dtype, copy, order, ndmin, ...])
[`ndarray`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.html#jax.numpy.ndarray "jax.numpy.ndarray")
[`asarray`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.asarray.html#jax.numpy.asarray "jax.numpy.asarray")(a[, dtype, order, copy, device, ...])
: Array construction from `object`, where the object can be things like python native numbers, lists, etc.
Two versions have some subtle differences.

[`astype`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.astype.html#jax.numpy.astype "jax.numpy.astype")(x, dtype, /, *[, copy, device])
: array type conversion.

[`copy`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.copy.html#jax.numpy.copy "jax.numpy.copy")(a[, order])
: copy an array.

[`fromfunction`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fromfunction.html#jax.numpy.fromfunction "jax.numpy.fromfunction")(function, shape, *[, dtype])
: array by calling `function` on array indices.

[`arange`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange")(start[, stop, step, dtype, device, ...])
: Like python `range()`.

[`linspace`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linspace.html#jax.numpy.linspace "jax.numpy.linspace")(start, stop[, num, endpoint, ...])
: evenly spaced.

[`logspace`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logspace.html#jax.numpy.logspace "jax.numpy.logspace")(start, stop[, num, endpoint, base, ...])
: logarithmically spaced.

[`geomspace`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.geomspace.html#jax.numpy.geomspace "jax.numpy.geomspace")(start, stop[, num, endpoint, ...])
: geometrically spaced.

[`atleast_1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_1d.html#jax.numpy.atleast_1d "jax.numpy.atleast_1d")(*arys)
[`atleast_2d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_2d.html#jax.numpy.atleast_2d "jax.numpy.atleast_2d")(*arys)
[`atleast_3d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_3d.html#jax.numpy.atleast_3d "jax.numpy.atleast_3d")(*arys)
: trying to increase the dimension of tensor/array without increase the number of elements, to 1/2/3.
Feels sloppy... Shouldn't the input dimension be known exactly and increase dimension with things like `broadcast_to`?

### Indices and mesh

[`indices`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.indices.html#jax.numpy.indices "jax.numpy.indices")(dimensions[, dtype, sparse])
: get indices of an array of the given shape `dimensions`.
Returned indices are in "column"-style: `ret[i]` is an array of shape `dimensions` representing the i-th index. That is, `(ret[0][pos], ret[1][pos], ...)` as a whole specifies the index of cell `pos` in the array.
When `sparse=True`, the replications are eliminated (recursively) to save memory; the original indices can be reconstructed via array broadcasting.

[`mask_indices`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.mask_indices.html#jax.numpy.mask_indices "jax.numpy.mask_indices")(n, mask_func[, k, size])
: indices of a mask. The mask is computed by `mask_func` on an `(n,n)`-array. (Why 2D?)

[`ix_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ix_.html#jax.numpy.ix_ "jax.numpy.ix_")(*args)
: returns an "open mesh", not necessarily but often indices, of multiple 1D sequences.
Eg., `ix_([a0, a1, a2, a3], [b0, b1, b2])` will try to generate `(x, y)` so when varying `i`, tuple `(x[i], y[i])` will loop through all combinations of `(aj, bk)` for all possible `j, k`.

[`meshgrid`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.meshgrid.html#jax.numpy.meshgrid "jax.numpy.meshgrid")(*xi[, copy, sparse, indexing])
[`mgrid`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.mgrid.html#jax.numpy.mgrid "jax.numpy.mgrid")
[`ogrid`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ogrid.html#jax.numpy.ogrid "jax.numpy.ogrid")
: create meshes.

[`ravel_multi_index`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ravel_multi_index.html#jax.numpy.ravel_multi_index "jax.numpy.ravel_multi_index")(multi_index, dims[, mode, ...])
[`unravel_index`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unravel_index.html#jax.numpy.unravel_index "jax.numpy.unravel_index")(indices, shape)
: conversion between multi-dimensional indices and flattened 1D indices. (Why are the names inconsistent?)

### Array manipulation

[`ndarray.at`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at")
: returns a "slice" of the array that can be manipulated (and the manipulation returns a copy of the whole array). A lot of these "manipulations" are implicitly used when `x[idx]` syntax is used.

[`append`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.append.html#jax.numpy.append "jax.numpy.append")(arr, values[, axis])

[`array_split`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_split.html#jax.numpy.array_split "jax.numpy.array_split")(ary, indices_or_sections[, axis])

[`put`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.put.html#jax.numpy.put "jax.numpy.put")(a, ind, v[, mode, inplace])
: perform `a[index] = value` by `(ind, v)`. Returns a copy of course given jnp objects are immutable (rely on XLA to optimize away unnecessary copies).
*Subtlety*: array `a` is treated as if it is flattened, `ind` is 1D numbers. It is suggested more to use [`ndarray.at`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at") instead.

[`put_along_axis`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.put_along_axis.html#jax.numpy.put_along_axis "jax.numpy.put_along_axis")(arr, indices, values, axis[, ...])
: `indices` must define, for each dimension outside `axis`, a value to tell along `axis`, which cell to put a value.

[`insert`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.insert.html#jax.numpy.insert "jax.numpy.insert")(arr, obj, values[, axis])
: insert entries into the array.

[`delete`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.delete.html#jax.numpy.delete "jax.numpy.delete")(arr, obj[, axis, assume_unique_indices])
: delete entries from the array.

[`flip`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.flip.html#jax.numpy.flip "jax.numpy.flip")(m[, axis])
[`fliplr`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fliplr.html#jax.numpy.fliplr "jax.numpy.fliplr")(m)
[`flipud`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.flipud.html#jax.numpy.flipud "jax.numpy.flipud")(m)
: flip along an axis. `fliplr` and `flipud` are fixed on axis=1/0 respectively.

[`rot90`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.rot90.html#jax.numpy.rot90 "jax.numpy.rot90")(m[, k, axes])
: `axes` is a 2-tuple defining the rotation plane; the whole array's each plane will be rotated 90 degrees CCW.

[`concat`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concat.html#jax.numpy.concat "jax.numpy.concat")(arrays, /, *[, axis])
[`concatenate`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate")(arrays[, axis, dtype])
: concatenate the given sequence of arrays along the given axis.

[`trim_zeros`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.trim_zeros.html#jax.numpy.trim_zeros "jax.numpy.trim_zeros")(filt[, trim, axis])
: remove leading and/or trailing zeros from array `filt`.
When in higher dimensions, eg., in 2D, a leading/trailing row/column will be removed if it is all-zero.

[`c_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.c_.html#jax.numpy.c_ "jax.numpy.c_")
[`r_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.r_.html#jax.numpy.r_ "jax.numpy.r_")
[`s_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.s_.html#jax.numpy.s_ "jax.numpy.s_")
[`index_exp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.index_exp.html#jax.numpy.index_exp "jax.numpy.index_exp")
:
TODO: check again.

#### Filter-style.

[`compress`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.compress.html#jax.numpy.compress "jax.numpy.compress")(condition, a[, axis, size, ...])
: delete/filter array `a` along with axis, according to boolean array `condition`. True for keeping, False for removing.
`condition` is 1D (along axis). Returned `a` has the same dimension but smaller along axis.

[`extract`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.extract.html#jax.numpy.extract "jax.numpy.extract")(condition, arr, *[, size, fill_value])
: delete/filter array by a matching or broadcastable `condition`. Returns a 1D array of kept element.

[`take`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html#jax.numpy.take "jax.numpy.take")(a, indices[, axis, out, mode, ...])
: only keep `a`'s elements designated by `indices`, which is an array of integers pointing to `a` as if it is flattened 1D.

[`take_along_axis`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take_along_axis.html#jax.numpy.take_along_axis "jax.numpy.take_along_axis")(arr, indices[, axis, mode, ...])
: the `axis` specifies which dimension the choice happens and the `indices` are within that dimension.

### Matrix-like

[`matrix_transpose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matrix_transpose.html#jax.numpy.matrix_transpose "jax.numpy.matrix_transpose")(x, /)
: when the array is higher dimension, transpose the last two dimensions.

[`swapaxes`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.swapaxes.html#jax.numpy.swapaxes "jax.numpy.swapaxes")(a, axis1, axis2)
: in between `matrix_transpose` and `transpose` in terms of generality.

[`transpose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose")(a[, axes])
: generalized transpose from `matrix_transpose` by allowing all axes to be shuffled, instead of just the last two.

[`rollaxis`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.rollaxis.html#jax.numpy.rollaxis "jax.numpy.rollaxis")(a, axis[, start])
: more exotic than `transpose`.

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

[`tri`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tri.html#jax.numpy.tri "jax.numpy.tri")(N[, M, k, dtype])
: a matrix, elements on and below the diagonal are ones; above, zeros.

[`trace`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.trace.html#jax.numpy.trace "jax.numpy.trace")(a[, offset, axis1, axis2, dtype, out])
: sum along the diagnal, plane defined by `axis1` and `axis2`.

[`block`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.block.html#jax.numpy.block "jax.numpy.block")(arrays)
: array from "sub"-arrays like gluing sub-matrices into a matrix.

[`kron`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.kron.html#jax.numpy.kron "jax.numpy.kron")(a, b)
: Kronecker product.

[`vander`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vander.html#jax.numpy.vander "jax.numpy.vander")(x[, N, increasing])
: Vandermonde matrix (a[i,j] = x[i]**j).

[`tril`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tril.html#jax.numpy.tril "jax.numpy.tril")(m[, k])
[`triu`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.triu.html#jax.numpy.triu "jax.numpy.triu")(m[, k])
: returns the lower/upper triangle of matrix `m` (i.e., elements above/below the diagonal are replaced by zeros).

[`tril_indices`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tril_indices.html#jax.numpy.tril_indices "jax.numpy.tril_indices")(n[, k, m])
[`tril_indices_from`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tril_indices_from.html#jax.numpy.tril_indices_from "jax.numpy.tril_indices_from")(arr[, k])
[`triu_indices`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.triu_indices.html#jax.numpy.triu_indices "jax.numpy.triu_indices")(n[, k, m])
[`triu_indices_from`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.triu_indices_from.html#jax.numpy.triu_indices_from "jax.numpy.triu_indices_from")(arr[, k])
: indices of a lower triangle matrix.

### Polynomial

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

## Shape manipulation

[`reshape`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.reshape.html#jax.numpy.reshape "jax.numpy.reshape")(a, shape[, order, copy, out_sharding])
: well, very drastic and dramatic approach...

[`ravel`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ravel.html#jax.numpy.ravel "jax.numpy.ravel")(a[, order, out_sharding])
: flatten the array into 1D.

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

[`unstack`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unstack.html#jax.numpy.unstack "jax.numpy.unstack")(x, /, *[, axis])
[`split`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.split.html#jax.numpy.split "jax.numpy.split")(ary, indices_or_sections[, axis])
[`stack`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.stack.html#jax.numpy.stack "jax.numpy.stack")(arrays[, axis, out, dtype])
: split/stack/unstack along an axis.

[`vsplit`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vsplit.html#jax.numpy.vsplit "jax.numpy.vsplit")(ary, indices_or_sections)
[`vstack`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vstack.html#jax.numpy.vstack "jax.numpy.vstack")(tup[, dtype])
: vertically (axis 0) split and stack.

[`hsplit`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hsplit.html#jax.numpy.hsplit "jax.numpy.hsplit")(ary, indices_or_sections)
[`hstack`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hstack.html#jax.numpy.hstack "jax.numpy.hstack")(tup[, dtype])
: horizontally (axis 1) split and stack.

[`dsplit`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dsplit.html#jax.numpy.dsplit "jax.numpy.dsplit")(ary, indices_or_sections)
[`dstack`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dstack.html#jax.numpy.dstack "jax.numpy.dstack")(tup[, dtype])
: depth-wise (axis 2) split and stack.

[`moveaxis`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.moveaxis.html#jax.numpy.moveaxis "jax.numpy.moveaxis")(a, source, destination)
[`permute_dims`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.permute_dims.html#jax.numpy.permute_dims "jax.numpy.permute_dims")(a, /, axes)
: TODO another look.

[`pad`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.pad.html#jax.numpy.pad "jax.numpy.pad")(array, pad_width[, mode])
: pad the array.

[`tile`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tile.html#jax.numpy.tile "jax.numpy.tile")(A, reps)
: expand the array by repeating. Each dimension's repeating count can be specified individually in `reps`; or a single repeating count so it applies to all dimensions.

[`roll`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.roll.html#jax.numpy.roll "jax.numpy.roll")(a, shift[, axis])
: roll `a` by `shift` amount rightward, i.e., value with index `i` at that `axis` goes to `(i + shift) mod N`.

[`squeeze`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.squeeze.html#jax.numpy.squeeze "jax.numpy.squeeze")(a[, axis])
: remove dimensions of size 1 (according to `axis`, or if not given, all such dimensions). Eg., an array of shape [3, 1, 2, 1] will become of shape [3, 2] after `squeeze()` without axis. Feels very sloppy and error-prone, because the 3 in the [3, 1, 2, 1] might be a variable and can occasionally become 1 due to different inputs, but we definitely don't want to eliminate that dimension.

### Debug-related

[`array_repr`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_repr.html#jax.numpy.array_repr "jax.numpy.array_repr")(arr[, max_line_width, precision, ...])

[`array_str`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_str.html#jax.numpy.array_str "jax.numpy.array_str")(a[, max_line_width, precision, ...])

[`fromstring`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fromstring.html#jax.numpy.fromstring "jax.numpy.fromstring")(string[, dtype, count])

## Selection, Sorting

[`argwhere`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argwhere.html#jax.numpy.argwhere "jax.numpy.argwhere")(a, *[, size, fill_value])
: returns an array (dynamic-sized if `size` is not given) of indices of non-zero elements in `a`, each index is n dimensions if `a` is an n-order tensor/array.

[`nonzero`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero")(a, *[, size, fill_value])
: also returns indices like `argwhere`, but indices are parallel n arrays, each array corresponds to one dimension of `a`'s shape.

[`flatnonzero`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.flatnonzero.html#jax.numpy.flatnonzero "jax.numpy.flatnonzero")(a, *[, size, fill_value])
: returns indices of `a`, N-dim array treated as a flattened 1D array, that are non-zeros.

[`sort`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sort.html#jax.numpy.sort "jax.numpy.sort")(a[, axis, kind, order, stable, descending])
[`argsort`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argsort.html#jax.numpy.argsort "jax.numpy.argsort")(a[, axis, kind, order, stable, ...])
: sort along axis.

[`partition`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.partition.html#jax.numpy.partition "jax.numpy.partition")(a, kth[, axis])
[`argpartition`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argpartition.html#jax.numpy.argpartition "jax.numpy.argpartition")(a, kth[, axis])
: partition for sort along an axis (the quicksort style partition, but `kth` is rigorous).

[`searchsorted`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.searchsorted.html#jax.numpy.searchsorted "jax.numpy.searchsorted")(a, v[, side, sorter, method])
: binary search, returns the index. `a` is 1D.

[`lexsort`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.lexsort.html#jax.numpy.lexsort "jax.numpy.lexsort")(keys[, axis])
: sort the array lexicographically.

[`sort_complex`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sort_complex.html#jax.numpy.sort_complex "jax.numpy.sort_complex")(a)
: sort complex numbers, lexicographically.

[`unique`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unique.html#jax.numpy.unique "jax.numpy.unique")(ar[, return_index, return_inverse, ...])
[`unique_all`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unique_all.html#jax.numpy.unique_all "jax.numpy.unique_all")(x, /, *[, size, fill_value])
[`unique_counts`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unique_counts.html#jax.numpy.unique_counts "jax.numpy.unique_counts")(x, /, *[, size, fill_value])
[`unique_inverse`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unique_inverse.html#jax.numpy.unique_inverse "jax.numpy.unique_inverse")(x, /, *[, size, fill_value])
[`unique_values`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unique_values.html#jax.numpy.unique_values "jax.numpy.unique_values")(x, /, *[, size, fill_value])
: remove duplicated values in an array. Feels like a lot of versions of functions squeezed into one function signature. Unless `size` is specified, very unfriendly to `jit`.

## Types

[`generic`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.generic.html#jax.numpy.generic "jax.numpy.generic")()
: base class of "most" scalar types.

[`object_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.object_.html#jax.numpy.object_ "jax.numpy.object_")([value])
: alias of python `object`.

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

### Type operations

[`can_cast`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.can_cast.html#jax.numpy.can_cast "jax.numpy.can_cast")(from_, to[, casting])
: NOT ASYNC. Help function to test whether one type can be casted to another. Just returns True/False, simple value, not in the array container.

[`dtype`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dtype.html#jax.numpy.dtype "jax.numpy.dtype")(dtype[, align, copy])
: NOT ASYNC. A type object. TODO: take another look.

[`iinfo`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.iinfo.html#jax.numpy.iinfo "jax.numpy.iinfo")(int_type)
[`finfo`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.finfo.html#jax.numpy.finfo "jax.numpy.finfo")(dtype)
: NOT ASYNC. Some interesting metadata about integer or floating point types.

[`isdtype`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isdtype.html#jax.numpy.isdtype "jax.numpy.isdtype")(dtype, kind)
: NOT ASYNC.

[`issubdtype`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.issubdtype.html#jax.numpy.issubdtype "jax.numpy.issubdtype")(arg1, arg2)
: NOT ASYNC.

[`isscalar`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isscalar.html#jax.numpy.isscalar "jax.numpy.isscalar")(element)
: NOT ASYNC.

[`iterable`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.iterable.html#jax.numpy.iterable "jax.numpy.iterable")(y)
: NOT ASYNC. Whether the object is iterable.

[`promote_types`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.promote_types.html#jax.numpy.promote_types "jax.numpy.promote_types")(a, b)
[`result_type`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.result_type.html#jax.numpy.result_type "jax.numpy.result_type")(*args)
: NOT ASYNC. A common type to hold a binary calculation between `a` and `b`. Eg., `promote_types('float32', 'int32') == dtype('float32')`.

[`isrealobj`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isrealobj.html#jax.numpy.isrealobj "jax.numpy.isrealobj")(x)
: whether it is a non-complex, or an array of non-complex numbers.

[`iscomplexobj`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.iscomplexobj.html#jax.numpy.iscomplexobj "jax.numpy.iscomplexobj")(x)
: whether it is a complex, or an array of complex.

[`iscomplex`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.iscomplex.html#jax.numpy.iscomplex "jax.numpy.iscomplex")(x)
: whether it is a complex.

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

[`std`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.std.html#jax.numpy.std "jax.numpy.std")(a[, axis, dtype, out, ddof, keepdims, ...])

[`var`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.var.html#jax.numpy.var "jax.numpy.var")(a[, axis, dtype, out, ddof, keepdims, ...])

[`ptp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ptp.html#jax.numpy.ptp "jax.numpy.ptp")(a[, axis, out, keepdims])
: "peak-to-peak" difference along a given `axis` (default: no axis, whole n-dim array viewed flattened).

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

[`select`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.select.html#jax.numpy.select "jax.numpy.select")(condlist, choicelist[, default])
: `condlist` and `choicelist` are both sequences of broadcast-compatible arrays. For each position `pos` in the final array, the returned value `ret[pos]` will be `choicelist[i][pos]` where `i` is the first `condlist[i][pos]` with value True.

[`piecewise`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.piecewise.html#jax.numpy.piecewise "jax.numpy.piecewise")(x, condlist, funclist, *args, **kw)
: the `function`-instead-of-value-list version of `select`.
Each element in `x` corresponds to an element in `condlist`, which determines which function in `funclist` to apply.

[`choose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.choose.html#jax.numpy.choose "jax.numpy.choose")(a, choices[, out, mode])
: the `axiom of choice`-style representation? Feels like a failed attempt to get something engineering-wise useful in a math library: too confusing, the library documentation only explained the 1D case and pointed to `jax.lax.switch` instead (select functoin first, then apply to array).

[`bincount`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bincount.html#jax.numpy.bincount "jax.numpy.bincount")(x[, weights, minlength, length])

[`digitize`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.digitize.html#jax.numpy.digitize "jax.numpy.digitize")(x, bins[, right, method])

[`isin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isin.html#jax.numpy.isin "jax.numpy.isin")(element, test_elements[, ...])
: each of `element` is mapped to a boolean: whether this element is in `test_elements` (another array as collection).

[`packbits`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.packbits.html#jax.numpy.packbits "jax.numpy.packbits")(a[, axis, bitorder])
[`unpackbits`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unpackbits.html#jax.numpy.unpackbits "jax.numpy.unpackbits")(a[, axis, count, bitorder])
: pack 8 bits in an array into a uint8.
If the array is longer than 8, it will be truncated into window-8 and each one becomes an element.
Incomplete window-8 will also be treated as a result number.
`axis` controls which dimension to pack values (the other dimensions are simply treated as a batch); if `axis` is not given, the whole array will be flattened into 1D.
Also the inverse version.

[`unwrap`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unwrap.html#jax.numpy.unwrap "jax.numpy.unwrap")(p[, discont, axis, period])
: unwrap a periodic signal.
The example basically said if you only have a sequence `(a[i] mod period)` (so lost some informatoin from `(a[i])`) and you know it is quite continuous, this function will try to recover the original `a[i]` by providing `discount` and `period`.

### Window

[`bartlett`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bartlett.html#jax.numpy.bartlett "jax.numpy.bartlett")(M)

[`blackman`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.blackman.html#jax.numpy.blackman "jax.numpy.blackman")(M)

[`hamming`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hamming.html#jax.numpy.hamming "jax.numpy.hamming")(M)

[`hanning`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hanning.html#jax.numpy.hanning "jax.numpy.hanning")(M)

[`kaiser`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.kaiser.html#jax.numpy.kaiser "jax.numpy.kaiser")(M, beta)

### Probably don't care much

Either too trivial or not quite relevant to model training/inference.

[`ComplexWarning`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ComplexWarning.html#jax.numpy.ComplexWarning "jax.numpy.ComplexWarning")

[`frombuffer`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.frombuffer.html#jax.numpy.frombuffer "jax.numpy.frombuffer")(buffer[, dtype, count, offset])

[`printoptions`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.printoptions.html#jax.numpy.printoptions "jax.numpy.printoptions")(*args, **kwargs)
[`get_printoptions`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.get_printoptions.html#jax.numpy.get_printoptions "jax.numpy.get_printoptions")()
[`set_printoptions`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.set_printoptions.html#jax.numpy.set_printoptions "jax.numpy.set_printoptions")(*args, **kwargs)
: simply forwarded to numpy ([this]((https://numpy.org/doc/stable/reference/generated/numpy.printoptions.html#numpy.printoptions), [this](https://numpy.org/doc/stable/reference/generated/numpy.get_printoptions.html#numpy.get_printoptions) and [this](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html#numpy.set_printoptions)).


[`from_dlpack`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.from_dlpack.html#jax.numpy.from_dlpack "jax.numpy.from_dlpack")(x, /, *[, device, copy])

[`load`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.load.html#jax.numpy.load "jax.numpy.load")(file, *args, **kwargs)
[`save`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.save.html#jax.numpy.save "jax.numpy.save")(file, arr[, allow_pickle])
: from/to .npy file.

[`savez`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.savez.html#jax.numpy.savez "jax.numpy.savez")(file, *args[, allow_pickle])
: save to .npz file. (No load? too asymmetric...)

#### Not even impelemented

Even the jax developers don't care about these :-)

[`fromfile`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fromfile.html#jax.numpy.fromfile "jax.numpy.fromfile")(*args, **kwargs)

[`fromiter`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fromiter.html#jax.numpy.fromiter "jax.numpy.fromiter")(*args, **kwargs)

