# Shape manipulation

This section lists tools to put essentially the same information in different shapes.

[`shape`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.shape.html#jax.numpy.shape "jax.numpy.shape")(a)
[`ndim`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndim.html#jax.numpy.ndim "jax.numpy.ndim")(a)
: shape (or just dimension) of an array.

## Only shape change

[`reshape`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.reshape.html#jax.numpy.reshape "jax.numpy.reshape")(a, shape[, order, copy, out_sharding])
: nuke option (powerful, not enough safe guards; use with caution). Ignore the array's original shape and fit in a new shape.

[`ravel`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ravel.html#jax.numpy.ravel "jax.numpy.ravel")(a[, order, out_sharding])
: another nuke option. flatten the array into 1D.

[`expand_dims`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.expand_dims.html#jax.numpy.expand_dims "jax.numpy.expand_dims")(a, axis)
: expand 1 dimension along axis (eg., x -> [x]).

[`squeeze`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.squeeze.html#jax.numpy.squeeze "jax.numpy.squeeze")(a[, axis])
: remove dimensions of size 1 (according to `axis`, or if not given, all such dimensions). Eg., an array of shape [3, 1, 2, 1] will become of shape [3, 2] after `squeeze()` without axis. Feels very sloppy and error-prone, because the 3 in the [3, 1, 2, 1] might be a variable and can occasionally become 1 due to different inputs, but we definitely don't want to eliminate that dimension.

[`atleast_1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_1d.html#jax.numpy.atleast_1d "jax.numpy.atleast_1d")(*arys)
[`atleast_2d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_2d.html#jax.numpy.atleast_2d "jax.numpy.atleast_2d")(*arys)
[`atleast_3d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_3d.html#jax.numpy.atleast_3d "jax.numpy.atleast_3d")(*arys)
: trying to increase the dimension of tensor/array without increase the number of elements, to 1/2/3.
Feels sloppy... Shouldn't the input dimension be known exactly and increase dimension with things like `broadcast_to`?

## Matrix-transpose-like

Basically swap two or more axes without content change.

[`matrix_transpose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matrix_transpose.html#jax.numpy.matrix_transpose "jax.numpy.matrix_transpose")(x, /)
: when the array is higher dimension, transpose the last two dimensions.

[`swapaxes`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.swapaxes.html#jax.numpy.swapaxes "jax.numpy.swapaxes")(a, axis1, axis2)
: in between `matrix_transpose` and `transpose` in terms of generality.

[`transpose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose")(a[, axes])
[`permute_dims`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.permute_dims.html#jax.numpy.permute_dims "jax.numpy.permute_dims")(a, /, axes)
: generalized transpose from `matrix_transpose` by allowing all axes to be shuffled, instead of just the last two.

[`moveaxis`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.moveaxis.html#jax.numpy.moveaxis "jax.numpy.moveaxis")(a, source, destination)
: different taste of `swapaxes`, by viewing the shape as a "mutable list".

## Repetition

This part actually increases the number of cells by repeating the original content in some ways. Two options: broadcasting or tiling.

[`broadcast_to`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.broadcast_to.html#jax.numpy.broadcast_to "jax.numpy.broadcast_to")(array, shape, *[, out_sharding])
: expand the `array` to `shape` by broadcasting. Each dimension of `array` should either match `shape` or is one; the latter case, `array` will be logically repeated to fill `shape`.
Note that if `array` has lower dimension, its shape will be padded up from left with 1s.

[`broadcast_shapes`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.broadcast_shapes.html#jax.numpy.broadcast_shapes "jax.numpy.broadcast_shapes")(*shapes)
: not async. Given a bunch of shapes, calculate the common shape.

[`broadcast_arrays`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.broadcast_arrays.html#jax.numpy.broadcast_arrays "jax.numpy.broadcast_arrays")(*args)
: given arrays directly, get a common shape if they are broadcast-compatible.

[`tile`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tile.html#jax.numpy.tile "jax.numpy.tile")(A, reps)
: expand the array by repeating. Each dimension's repeating count can be specified individually in `reps`; or a single repeating count so it applies to all dimensions.
Note: this looks quite simiar to `broadcast_to`, but each dimension is not given by its final size but given by times of repetition.

[`repeat`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.repeat.html#jax.numpy.repeat "jax.numpy.repeat")(a, repeats[, axis, ...])
: expand the dimension along `axis` by repeating values. `repeats` can designate each value along `axis` how many times it needs to repeat.

[`resize`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.resize.html#jax.numpy.resize "jax.numpy.resize")(a, new_shape)
: nuke option (use with caution). Resize an array to the given shape. Larger size along an axis will be filled with repetition; smaller size causes truncation. No shape sanity checks.

[`pad`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.pad.html#jax.numpy.pad "jax.numpy.pad")(array, pad_width[, mode])
: pad the array, each dimension can add from both sides some value. This value, according to the `mode` argument, can be some "fill value" or some statistic value computed from the original array.

## Multiple operands

[`concat`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concat.html#jax.numpy.concat "jax.numpy.concat")(arrays, /, *[, axis])
[`concatenate`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate")(arrays[, axis, dtype])
: concatenate the given sequence of arrays along the given axis.

[`column_stack`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.column_stack.html#jax.numpy.column_stack "jax.numpy.column_stack")(tup)
: stack multiple arrays along axis 1.

[`c_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.c_.html#jax.numpy.c_ "jax.numpy.c_")
[`r_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.r_.html#jax.numpy.r_ "jax.numpy.r_")
:
numpy-taste objects with bracket syntax instead of functions. Column-stacking and row-stacking.
Note: `r_` can take a secret first arg to do some very tricky maneuvers.

[`split`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.split.html#jax.numpy.split "jax.numpy.split")(ary, indices_or_sections[, axis])
[`array_split`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_split.html#jax.numpy.array_split "jax.numpy.array_split")(ary, indices_or_sections[, axis])
: `split` allows splitting an array either evenly by number or by explicit boundaries. `array_split` allow uneven split by number.

[`unstack`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unstack.html#jax.numpy.unstack "jax.numpy.unstack")(x, /, *[, axis])
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

