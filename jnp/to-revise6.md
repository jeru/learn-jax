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

## Misc (TODO: revisit all and classify properly)

[`select`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.select.html#jax.numpy.select "jax.numpy.select")(condlist, choicelist[, default])
: `condlist` and `choicelist` are both sequences of broadcast-compatible arrays. For each position `pos` in the final array, the returned value `ret[pos]` will be `choicelist[i][pos]` where `i` is the first `condlist[i][pos]` with value True.

[`piecewise`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.piecewise.html#jax.numpy.piecewise "jax.numpy.piecewise")(x, condlist, funclist, *args, **kw)
: the `function`-instead-of-value-list version of `select`.
Each element in `x` corresponds to an element in `condlist`, which determines which function in `funclist` to apply.

[`choose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.choose.html#jax.numpy.choose "jax.numpy.choose")(a, choices[, out, mode])
: the `axiom of choice`-style representation? Feels like a failed attempt to get something engineering-wise useful in a math library: too confusing, the library documentation only explained the 1D case and pointed to `jax.lax.switch` instead (select functoin first, then apply to array).

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

