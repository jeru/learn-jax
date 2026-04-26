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
