# Array Operations.

## Basic constructors

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

## A few 1D "constants"

[`arange`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange")(start[, stop, step, dtype, device, ...])
: Like python `range()`.

[`linspace`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linspace.html#jax.numpy.linspace "jax.numpy.linspace")(start, stop[, num, endpoint, ...])
: evenly spaced.

[`logspace`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logspace.html#jax.numpy.logspace "jax.numpy.logspace")(start, stop[, num, endpoint, base, ...])
: logarithmically spaced.

[`geomspace`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.geomspace.html#jax.numpy.geomspace "jax.numpy.geomspace")(start, stop[, num, endpoint, ...])
: geometrically spaced.

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

[`s_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.s_.html#jax.numpy.s_ "jax.numpy.s_")
[`index_exp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.index_exp.html#jax.numpy.index_exp "jax.numpy.index_exp")
:
TODO: check again.

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

[`trim_zeros`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.trim_zeros.html#jax.numpy.trim_zeros "jax.numpy.trim_zeros")(filt[, trim, axis])
: remove leading and/or trailing zeros from array `filt`.
When in higher dimensions, eg., in 2D, a leading/trailing row/column will be removed if it is all-zero.

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

[`roll`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.roll.html#jax.numpy.roll "jax.numpy.roll")(a, shift[, axis])
: roll `a` by `shift` amount rightward, i.e., value with index `i` at that `axis` goes to `(i + shift) mod N`.

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
