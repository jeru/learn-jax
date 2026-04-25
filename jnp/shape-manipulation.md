# Shape manipulation

[`shape`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.shape.html#jax.numpy.shape "jax.numpy.shape")(a)
[`ndim`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndim.html#jax.numpy.ndim "jax.numpy.ndim")(a)
: shape (or just dimension) of an array.

[`atleast_1d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_1d.html#jax.numpy.atleast_1d "jax.numpy.atleast_1d")(*arys)
[`atleast_2d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_2d.html#jax.numpy.atleast_2d "jax.numpy.atleast_2d")(*arys)
[`atleast_3d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.atleast_3d.html#jax.numpy.atleast_3d "jax.numpy.atleast_3d")(*arys)
: trying to increase the dimension of tensor/array without increase the number of elements, to 1/2/3.
Feels sloppy... Shouldn't the input dimension be known exactly and increase dimension with things like `broadcast_to`?

[`c_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.c_.html#jax.numpy.c_ "jax.numpy.c_")
[`r_`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.r_.html#jax.numpy.r_ "jax.numpy.r_")
:
TODO: check again.

## Matrix-transpose-like

[`matrix_transpose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matrix_transpose.html#jax.numpy.matrix_transpose "jax.numpy.matrix_transpose")(x, /)
: when the array is higher dimension, transpose the last two dimensions.

[`swapaxes`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.swapaxes.html#jax.numpy.swapaxes "jax.numpy.swapaxes")(a, axis1, axis2)
: in between `matrix_transpose` and `transpose` in terms of generality.

[`transpose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose")(a[, axes])
: generalized transpose from `matrix_transpose` by allowing all axes to be shuffled, instead of just the last two.

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

[`squeeze`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.squeeze.html#jax.numpy.squeeze "jax.numpy.squeeze")(a[, axis])
: remove dimensions of size 1 (according to `axis`, or if not given, all such dimensions). Eg., an array of shape [3, 1, 2, 1] will become of shape [3, 2] after `squeeze()` without axis. Feels very sloppy and error-prone, because the 3 in the [3, 1, 2, 1] might be a variable and can occasionally become 1 due to different inputs, but we definitely don't want to eliminate that dimension.
