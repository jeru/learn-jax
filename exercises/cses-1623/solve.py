# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1623
#
# Small N, N numbers. Find the subset whose sum is as close to half sum as
# possible. NP-hard. N small enough to run O(2^N).
#
# And just output the difference of the in-sum and out-sum, no concrete plan
# needed.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"


import jax
from jax import numpy as jnp
from jaxtyping import Array, Int64

@jax.jit
def run_in_jax(a: Int64[Array, " N"]) -> Int64[Array, "1"]:
    n = a.shape[0]
    m = 1 << n
    # Operand 1: (m, 1). Operand 2: (1, n). End result: (m, n).
    masks = (jnp.expand_dims(jnp.arange(m), axis=1)
             >> jnp.expand_dims(jnp.arange(n), axis=0) & 1)
    # Each set (mask)'s sum. `a` is broadcast.
    sum_sets = jnp.sum(masks * a, axis=1)
    sum_a = jnp.sum(a)
    diff_sets = jnp.abs(sum_a - sum_sets * 2)
    return jnp.min(diff_sets)


def main():
    import sys
    sys.stdin.readline()
    a = jnp.array([int(x) for x in sys.stdin.readline().split()],
                  dtype=jnp.int64)
    ans = int(run_in_jax(a))
    print(ans)


if __name__ == "__main__":
    main()
