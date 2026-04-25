# jax.numpy functions

Reference page: https://docs.jax.dev/en/latest/jax.numpy.html

## Concepts

An `array` in the world of jax (and numpy) is generally multi-dimensional.

### Universal function (ufunc)

Comes [from numpy](https://numpy.org/doc/stable/reference/ufuncs.html).
In short, a `ufunc` is a function that, originally works on a single element, now applies to an array on each of its element one by one.
So, just adding parallelism.
This also generalizes to operations with more than one operands.
Eg., `[1, 2, 3] + [1, 2, 3] == [2, 4, 6]`, i.e., given the 

#### Broadcasting

For conveience, "broadcasting" is added so eg., when a user wants to multiple a vector by 2, they don't need to expand 2 to be an array before doing the multiplication.
They can simply go with `2 * jnp.array([1, 2, 3])`.

This concept is actually broader than just scalar operation.
When two operands are to be matched, element by element, if at any level/dimension one side is size 1 and the other side is size N, this size-1 side will be automatically extended to N.
Eg.,

```python
>>> jnp.array([[1, 2, 3]]) + jnp.array([[1], [2], [3]])
Array([[2, 3, 4],
       [3, 4, 5],
       [4, 5, 6]], dtype=int32)
```

There are also a few functions that allow a user do explicit broadcasting, like 2->6. Such kind of non-one broadcasting is not allowed in the implicit setup.

## Elementwise functions

Listed [here](elementwise.md).

Key takes:
* Every basic math function widely used and operates on numbers (integer, real, complex) is already here.
* Be careful about integer modulo when it comes to negative operands. Always tricky in every single language, and cannot be made consistent across all. Noticeably jnp (and numpy) isn't consistent with C/C++.
* The existence of functions like `expm1`, `log1p` and `logaddexp` exposes how tricky sometimes numerical stability can be. This is very relevant in model training as well.

## Matrix-multiplication+
