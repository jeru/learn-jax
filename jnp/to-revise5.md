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

## Misc (TODO: revisit all and classify properly)

[`select`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.select.html#jax.numpy.select "jax.numpy.select")(condlist, choicelist[, default])
: `condlist` and `choicelist` are both sequences of broadcast-compatible arrays. For each position `pos` in the final array, the returned value `ret[pos]` will be `choicelist[i][pos]` where `i` is the first `condlist[i][pos]` with value True.

[`piecewise`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.piecewise.html#jax.numpy.piecewise "jax.numpy.piecewise")(x, condlist, funclist, *args, **kw)
: the `function`-instead-of-value-list version of `select`.
Each element in `x` corresponds to an element in `condlist`, which determines which function in `funclist` to apply.

[`choose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.choose.html#jax.numpy.choose "jax.numpy.choose")(a, choices[, out, mode])
: the `axiom of choice`-style representation? Feels like a failed attempt to get something engineering-wise useful in a math library: too confusing, the library documentation only explained the 1D case and pointed to `jax.lax.switch` instead (select functoin first, then apply to array).

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

