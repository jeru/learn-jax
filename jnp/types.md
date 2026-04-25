# Types

## Available types
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

## Abstract base classes

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

## Operations on types themselves

[`dtype`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dtype.html#jax.numpy.dtype "jax.numpy.dtype")(dtype[, align, copy])
: a type object.

[`can_cast`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.can_cast.html#jax.numpy.can_cast "jax.numpy.can_cast")(from_, to[, casting])
: help function to test whether one type can be casted to another. Just returns True/False, simple value, not in the array container.

[`iinfo`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.iinfo.html#jax.numpy.iinfo "jax.numpy.iinfo")(int_type)
[`finfo`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.finfo.html#jax.numpy.finfo "jax.numpy.finfo")(dtype)
: some interesting metadata about integer or floating point types.

[`isdtype`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isdtype.html#jax.numpy.isdtype "jax.numpy.isdtype")(dtype, kind)
: help function to test types. `kind` can narrow down test ranges.

[`issubdtype`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.issubdtype.html#jax.numpy.issubdtype "jax.numpy.issubdtype")(arg1, arg2)
: help function to test subtyping.

[`isscalar`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isscalar.html#jax.numpy.isscalar "jax.numpy.isscalar")(element)

[`iterable`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.iterable.html#jax.numpy.iterable "jax.numpy.iterable")(y)
: whether the object is iterable.

[`promote_types`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.promote_types.html#jax.numpy.promote_types "jax.numpy.promote_types")(a, b)
[`result_type`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.result_type.html#jax.numpy.result_type "jax.numpy.result_type")(*args)
: a common type to hold a binary/other calculation between `a` and `b`. Eg., `promote_types('float32', 'int32') == dtype('float32')`.

## Operations on values

[`isrealobj`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isrealobj.html#jax.numpy.isrealobj "jax.numpy.isrealobj")(x)
: whether it is a non-complex, or an array of non-complex numbers.

[`iscomplexobj`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.iscomplexobj.html#jax.numpy.iscomplexobj "jax.numpy.iscomplexobj")(x)
: whether it is a complex, or an array of complex.

[`iscomplex`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.iscomplex.html#jax.numpy.iscomplex "jax.numpy.iscomplex")(x)
: whether it is a complex.

