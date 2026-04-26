# Overview of exercises

## Some "simple" loops

A jax-based [solution](cses-1068/solve.py) to [CSES-1068](https://cses.fi/problemset/task/1068).

Key take:
* lax loops are quite static, independent from what's supposed to be training data.
  Can be varied by parameters (like "I want the graph to be N layers" then change N)
  but cannot do dynamic looping without python dynamic orchestration (like "iterate inside the GPU until the array is fully zero without bothering the CPU").
