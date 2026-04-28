# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/3399
#
# Parallel task handling with different task sizes. Each task is a small
# combinatorial problem.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"


import jax
from jax import numpy as jnp
from jaxtyping import Array, Int64


MAX_N = 100


@jax.jit
def run_in_jax(nab: Int64[Array, "C 3"]) -> Int64[Array, "C M"]:
    # Shape (C, 1).
    n, a, b = jnp.split(nab, 3, axis=1)
    # Canonicalize all no-solution cases to (a, b) = (0, 1).
    nosol = jnp.logical_or(a + b > n, (a != 0) != (b != 0))
    a = jnp.where(nosol, jnp.zeros((), dtype=jnp.int64), a)
    b = jnp.where(nosol, jnp.ones((), dtype=jnp.int64), b)
    # Largest n - a - b cards are to be played tied.
    # Top a of player 1's cards win against bottom a of player 2's cards.
    # Top b of player 2's cards win against bottom b of player 1's cards.
    # So:
    # [0, b) maps to [a, a+b).
    # [b, a+b) maps to [0, a).
    # [a+b, n) maps to [a+b, n).
    # [n, MAX_N) maps to -1.
    loop = jnp.arange(MAX_N, dtype=jnp.int64)
    soln = jnp.where(loop < a + b,
                     jnp.where(loop < b, loop + a, loop - b),
                     jnp.where(loop < n, loop, jnp.full((), -1)))
    return jnp.where(nosol, jnp.full((), -1), soln)


def main():
    import numpy
    import sys
    n = int(sys.stdin.readline())
    a = numpy.array([sys.stdin.readline().split() for _ in range(n)],
                    dtype=jnp.int64)
    ans = run_in_jax(a)
    for soln in ans:
        soln = list(soln)
        if soln[0] == -1:
            print('NO')
        else:
            print('YES')
            while soln and soln[-1] == -1: soln.pop()
            print(' '.join(str(x + 1) for x in range(len(soln))))
            print(' '.join(str(x + 1) for x in soln))


if __name__ == "__main__":
    main()
