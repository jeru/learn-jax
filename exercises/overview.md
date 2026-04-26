# Overview of exercises

Following [this list](https://cses.fi/problemset/) to solve some simple problems and get a taste of `jax.numpy`.

## [CSES-1068](https://cses.fi/problemset/task/1068) (simple loop)

[Solution](cses-1068/solve.py).

Key takes:
* lax loops are quite static, independent from what's supposed to be training data.
  Can be varied by parameters (like "I want the graph to be N layers" then change N)
  but cannot do dynamic looping without python dynamic orchestration (like "iterate inside the GPU until the array is fully zero without bothering the CPU").

## [CSES-1083](https://cses.fi/problemset/task/1083) (summation)

[Solution](cses-1083/solve.py).

Key takes:
* A function decorated with `jax.jit` can be compiled multiple times, each time the underlying computation flow must deal with a static shape of arrays under the hood.
  Any change to any array shape:
  - Input side change: `jax.jit` will compile a new version of XLA binary.
  - Internal change: the compiler will reject dynamic changes.
* When some array shapes depend on the function's input argument, that argument's value should be declared static with
  ```python
  @partial(jax.jit, static_argnames=["arg1", ...])
  ```
  This way, the value of this argument (not only its type, which might contain shape information) is treated as static.
  So a call to the function with a different value will also cause a compilation of a new XLA binary.

## [CSES-1069](https://cses.fi/problemset/task/1069) (sequence processing)

[Solution](cses-1069/solve.py).

Key takes:
* The jax.numpy API is quite suitable to process sequences, but only if a parallel algorithm can be found.
* A user doesn't need to distinguish an input raw int type and an asynchronized int (eg., `arr.shape[0]`) when the API reference says the parameter is an int. The jax function can actually handle both.
* When using a "filter"-style function (like here [`jnp.compress`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.compress.html#jax.numpy.compress)) that shrinks an array with `jax.jit`, an explicit `size` should be given to make the output static-sized, with tails filled with some kind of invalid values.
* No type promotion is needed if all the operands have the same type. Jax also complains that there's no type promotion defined for 1-bit, 2-bit and 4-bit integers, which feels like a good thing (most langaues, including python and C/C++, are in general too sloppy on type promotion; should learn from Rust).
