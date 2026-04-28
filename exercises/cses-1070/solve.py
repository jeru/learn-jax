# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1070
#
# Given N, construact a permutation of 1-N such that no to adjacent numbers
# have adjacent values.
#
# Just construct it and give a mathematical prove.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
from jax import numpy as jnp
from jaxtyping import Array, Int64


@jax.jit
def run_in_jax(alike: Int64[Array, ' N']) -> Int64[Array, ' N']:
    n = alike.shape[0]
    left = jnp.arange(0, n // 2)
    right = jnp.arange(n // 2, n)
    # Interleave `right` and `left`, basically put them into matrix of (2, n/2)
    # and traverse the cells in column-major.

    # `left` might be short by 1.
    padded_left = jnp.resize(left, right.shape)
    flipped_matrix = jnp.matrix_transpose(jnp.stack([right, padded_left]))
    # Flatten matrix, and drop the potential padding.
    ans = jnp.resize(flipped_matrix, (n,))

    return ans + 1  # Format adaptation.


def main():
    import numpy
    import sys
    import time
    n = int(sys.stdin.readline().strip())
    if n == 1:
        ans = '1'
    elif n == 2 or n == 3:
        ans = None
    else:
        start_jax_time = time.perf_counter()
        ans = run_in_jax(jnp.zeros((n,), dtype=jnp.int64)).block_until_ready()
        jax_time = time.perf_counter() - start_jax_time
        print(f'Jax run time including compilation: {jax_time:.4f}s',
              file=sys.stderr)
        ans = ' '.join(map(str, numpy.asarray(ans)))
        ans_time = time.perf_counter() - start_jax_time
        print(f'Jax plus output materialization: {ans_time:.4f}s',
              file=sys.stderr)
    if ans:
        print(ans)
    else:
        print('NO SOLUTION')


if __name__ == "__main__":
    main()
