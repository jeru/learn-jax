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

## [CSES-1094](https://cses.fi/problemset/task/1094/) (prefix max)

[Solution](cses-1094/solve.py). Surprisingly simple for jax.numpy.

## [CSES-1070](https://cses.fi/problemset/task/1094/) (interleave)

[Solution](cses-1070/solve.py) and custom [answer checker](cses-1070/checker.py).

Key takes:
* Benchmarking: calling `block_until_ready()` will ensure the asynchronous computation is done.
* Benchmarking: compilation still takes time. It is recommended to dry run the jax function once before the full test, to exclude the compilation time. (So called "warm up".)
* Not relevant to jax: converting jax results to python via `[int(x) for x in ans]` is terribly slow. Use `map(int, numpy.asarray(ans))` with the raw `numpy` (NOT jax.numpy) makes things significantly faster (9 seconds vs 0.25 seconds for the biggest test of this problem).
