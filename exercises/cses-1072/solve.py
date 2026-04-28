# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1072
#
# Compute a sequence of numbers, each one is single f(N) for some simply
# math function on integer N.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial

import jax
from jax import numpy as jnp
from jaxtyping import Array, Int64


@partial(jax.jit, static_argnames=['n'])
def run_in_jax(n: int) -> Int64[Array, ' N']:
    num = jnp.arange(1, n + 1, dtype=jnp.int64)
    sqr = num * num
    pairs = sqr * (sqr - 1) // 2
    two_by_threes = (num - 1) * (num - 2)
    return pairs - two_by_threes * 4


def main():
    import numpy
    import sys
    n = int(sys.stdin.readline())
    ans = run_in_jax(n)
    ans_str = '\n'.join(map(str, numpy.asarray(ans)))
    print(ans_str)



if __name__ == "__main__":
    main()
