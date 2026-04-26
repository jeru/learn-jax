# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# Custom answer checker for https://cses.fi/problemset/task/2205
#
# Check the following:
# 1. Has 2**n codes.
# 2. Each code has length n.
# 3. Each code is adjacent to its neighbors.
# 4. Each code appears only once.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial

import jax
from jax import numpy as jnp
from jax import lax
from jaxtyping import Array, Int64

@partial(jax.jit, static_argnames=["n"])
def run_in_jax(n: int, a: Int64[Array, "TwoToN"]) -> Int64[Array, "1"]:
    adj_dists = jnp.bitwise_count(jnp.delete(a, 0) - jnp.delete(a, -1))
    dist1 = jnp.all(adj_dists == 1)
    unique = jnp.max(jnp.bincount(a, length=2**n)) == 1
    return jnp.logical_and(dist1, unique)


def main():
    import numpy
    import sys
    infile, outfile, ansfile = sys.argv[1:]
    with open(infile) as f:
        n = int(f.readline())
    with open(outfile) as f:
        seq = f.readlines()
        seq = [x.strip() for x in seq]
        seq = [x for x in seq if x]
    assert len(seq) == 2**n
    assert all(len(x) == n for x in seq)
    assert all(all(b in '01' for b in x) for x in seq)
    verify = bool(run_in_jax(n, jnp.array([int(x, 2) for x in seq],
                                          dtype=jnp.int64)))
    if verify:
        print('OK')
    else:
        print('FAILED!!!')
        sys.exit(1)


if __name__ == "__main__":
    main()
