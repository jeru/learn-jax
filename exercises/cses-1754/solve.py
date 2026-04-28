# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1754
#
# A sequence of tests in tuple `(a, b)`. Tell for each whether it is possible by
# clearing both numbers to zero via two operations: (-2, -1) and (-1, -2).
#
# By assuming (-2, -1) is used `x` times, to clear `a`, an extra `a - 2x` times
# of (-1, -2) must be applied. So the `b` side will have `b = x + 2 * (a - 2x)`.
# This ends with `x = (2a - b) // 3` but `x` also must be valid, i.e., between
# zero and a/2.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"


import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int64


@jax.jit
def run_in_jax(a: Int64[Array, ' N'],
               b: Int64[Array, ' N']) -> Bool[Array, ' N']:
    x, r = jnp.divmod(a * 2 - b, 3)
    return jnp.all(jnp.stack([x >= 0, x * 2 <= a, r == 0]), axis=0)


def main():
    import numpy
    import sys
    n = int(sys.stdin.readline())
    rows = list(map(lambda _: tuple(map(int, sys.stdin.readline().split())),
                    range(n)))

    a = jnp.array([x[0] for x in rows], dtype=jnp.int64)
    b = jnp.array([x[1] for x in rows], dtype=jnp.int64)
    ans = run_in_jax(a, b)
    print('\n'.join(map(lambda x: 'YES' if x else 'NO', numpy.asarray(ans))))


if __name__ == "__main__":
    main()
