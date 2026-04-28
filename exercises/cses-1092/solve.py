# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1092
#
# Whether 1..N can be partitioned into two sets of equal sum.
#
# If the sum itself is an odd number, then the answer is no.
# Otherwise, try the following strategy:
# * Start with the half-sum, subtract starting from N until the
#   remaining sum is too small for the next number.
# * Then if the remaining sum is not zero, pick this number as well.
# The long search process can be replaced by a prefix sum plus a binary search.
#
# The jax part will be made to return a mask array of length N.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial

import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool


@partial(jax.jit, static_argnames=['n'])
def run_in_jax(n: int) -> Bool[Array, ' N']:
    total = n * (n + 1) // 2
    if total % 2 == 1:
        return jnp.zeros((n,), dtype=jnp.bool_)
    half_sum = total // 2

    revs = jnp.flip(jnp.arange(1, n + 1, dtype=jnp.int64))
    rev_sums = jnp.add.accumulate(revs)
    idx = jnp.searchsorted(rev_sums, half_sum)

    remaining = half_sum - jnp.where(idx == 0, 0, rev_sums[idx - 1])
    mask_plus_one = (jnp.arange(0, n + 1) > n - idx).at[remaining].set(True)
    return jnp.delete(mask_plus_one, 0)


def main():
    import numpy
    import sys
    n = int(sys.stdin.readline())

    mask = run_in_jax(n)
    indices = numpy.indices((n,))
    left = numpy.compress(mask, indices)
    right = numpy.compress(numpy.logical_not(mask), indices)

    if left.size == 0 or right.size == 0:
        print('NO')
    else:
        print('YES')
        print(left.size)
        print(' '.join(map(lambda x: str(x + 1), left)))
        print(right.size)
        print(' '.join(map(lambda x: str(x + 1), right)))


if __name__ == "__main__":
    main()
