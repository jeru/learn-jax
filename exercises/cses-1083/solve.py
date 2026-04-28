# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1068/
#
# Given an array from 1 to N, one is missing. Which one?
#
# Can simply compute the sum/xor of all numbers from input and all numbers
# from 1-N manually and check the difference. So, totally converted into
# aggregation calcuation.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial

import jax
from jax import numpy as jnp
from jaxtyping import Array, Int64

# Learn note: treat `n` as a static value (jax.jit compile-time known).
# The consequenec is that for each different `n`, jax will create one
# compilation.
#
# jax.jit requires static array shape for each compilation, so a dynamic
# `n` won't compile.
@partial(jax.jit, static_argnames=["n"])
def run_in_jax(n: int, a: Int64[Array, " NMinusOne"]) -> Int64[Array, "1"]:
    b = jnp.arange(1, n + 1, dtype=jnp.int64)
    return jnp.bitwise_xor.reduce(jnp.concatenate([a, b], axis=0))


def main():
    import sys
    n = int(sys.stdin.readline())
    arr = [int(x) for x in sys.stdin.readline().split()]
    assert len(arr) == n - 1
    ans = run_in_jax(n, jnp.array(arr, dtype=jnp.int64))
    print(int(ans))


if __name__ == "__main__":
    main()
