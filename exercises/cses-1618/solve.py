# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1618
#
# How many trailing zeros are in N! (factorial).
#
# Basically N // 5 + N // 5^2 + ...
#
# In CPU implementation, complexity O(log_5 N) and space O(1).
#
# In GPU setup, in theory things can be run faster by using space O(log_5 N):
# it would only take O(log_2 log_5 N) rounds to compute all the powers of 5.
# In practice, however, likely the N won't be big enough to make a dent here.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial

import jax
from jax import numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Int64


MAX_POWER = 13
assert 5**MAX_POWER >= 1e9


@jax.jit
def run_in_jax(n: int) -> Int64[Array, '1']:
    # `n` is NOT static here!

    # After getting the one-row Vandermond matrix, drop the 1.0 at the end.
    pows = jnp.delete(jnp.vander(jnp.array([5], dtype=jnp.int64), N=MAX_POWER + 1), -1)
    return jnp.sum(n // pows)


def main():
    import numpy
    import sys
    n = int(sys.stdin.readline())

    ans = run_in_jax(n)
    print(int(ans))


if __name__ == "__main__":
    main()
