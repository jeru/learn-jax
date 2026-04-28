# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1094
#
# Increase array elements by 1 until it is monotonely non-decreasing.
#
# Basically prefix-max(a) - a.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
from jax import numpy as jnp
from jaxtyping import Array, Int64


@jax.jit
def run_in_jax(a: Int64[Array, ' N']) -> Int64[Array, '1']:
    pmax = jnp.maximum.accumulate(a)
    return jnp.sum(pmax - a)


def main():
    import sys
    sys.stdin.readline()
    seq = [int(x) for x in sys.stdin.readline().split()]
    ans = run_in_jax(jnp.array(seq, dtype=jnp.uint64))
    print(int(ans))


if __name__ == "__main__":
    main()
