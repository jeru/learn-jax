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

## Elementwise functions

Listed [here](elementwise.md).

Key takes:
* Every basic math function widely used and operates on numbers (integer, real, complex) is already here.
* Be careful about integer modulo when it comes to negative operands. Always tricky in every single language, and cannot be made consistent across all. Noticeably jnp (and numpy) isn't consistent with C/C++.
* The existence of functions like `expm1`, `log1p` and `logaddexp` exposes how tricky sometimes numerical stability can be. This is very relevant in model training as well.

## Shape manipulation

Listed [here](shape-manipulation.md).

Matrix transpose is also listed inside this.

Key takes:
* Useful stuff with a strong feeling of data plumbing. Like gluing two inputs together to feed into a new module, etc.

## Array operations

Listed [here](array-ops.md). Matrix-multiplication-like stuff are listed separately (not here).

Key takes:
* Very necessary stuff. Some parts feel very mathematical.
* All kinds of construction of basic values, and operations.

## Matrix-multiplication+

Listed [here](matrix-multiplication-plus.md).

Key takes:
* If you decide to only learn one thing here, learn `einsum`. Most other stuff here is just a special case of `einsum`.

## Reduction functions and statistics

Listed [here](reduction-statistics.md).

## Polynomial and set

Listed [here](polynomial-set.md).

A little bit exotic in model training.

## Types

Listed [here](types.md).

Good to know.

## Miscellanous stuff

Listed [here](misc.md).

Noticeably:
* Convolution.
* High-order diffs of arrays.
