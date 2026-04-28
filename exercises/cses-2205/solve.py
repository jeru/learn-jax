# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/2205
#
# Generate gray code (i.e., binary code with adjacent codes differ by only 1
# bit).

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial

import jax
from jax import numpy as jnp
from jaxtyping import Array, Int64

@partial(jax.jit, static_argnames=["n"])
def run_in_jax(n: int) -> Int64[Array, " M"]:
    # With this recursion, the function will be compiled `n` times.
    if n == 0:
        return jnp.zeros((1,), dtype=jnp.int64)
    codes = run_in_jax(n - 1)
    mask = codes[-1] ^ (1 << (n - 1))
    return jnp.concat([codes, codes ^ mask])


def main():
    import numpy
    import sys
    n = int(sys.stdin.readline())
    ans = run_in_jax(n)
    ans = '\n'.join(map(lambda x: f'{x:0{n}b}', numpy.asarray(ans)))
    print(ans)


if __name__ == "__main__":
    main()
