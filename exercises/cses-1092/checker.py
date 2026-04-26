# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# Output verifier for https://cses.fi/problemset/task/1070 given the output is
# not unique and no official verifier is provided.
#
# Need to check the ground-truth file whether it is a "NO". If not,
# go to check the test output by:
# 1. Two arrays' total size match N.
# 2. Each of 1 to N exactly once.
# 3. Two arrays' sums match.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial

import jax
from jax import numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Int64


NO_SOL_TEXT = 'NO'


@partial(jax.jit, static_argnames=['n'])
def run_in_jax(n: int, a1: Int64[Array, 'M'], a2: Int64[Array, 'N']) -> Bool[Array, '1']:
    elements_matched = jnp.bincount(jnp.concat([a1, a2]), length=n + 1) == (
            jnp.concat([jnp.zeros((1,)), jnp.ones((n,))]))
    same_sum = (jnp.sum(a1) - jnp.sum(a2) == 0)
    return jnp.logical_and(jnp.all(elements_matched), same_sum)


def main():
    import sys
    infile, outfile, ansfile = sys.argv[1:]
    with open(infile) as f:
        n = int(f.readline())
    with open(ansfile) as f:
        no_solution = f.readline().strip() == NO_SOL_TEXT
    with open(outfile) as f:
        if no_solution:
            matched = f.readline().strip() == NO_SOL_TEXT
        else:
            line1 = f.readline().strip()
            assert line1 == 'YES'
            n1 = int(f.readline())
            arr1 = jnp.array(list(map(int, f.readline().split())), dtype=jnp.int64)
            assert n1 == arr1.size
            n2 = int(f.readline())
            arr2 = jnp.array(list(map(int, f.readline().split())), dtype=jnp.int64)
            assert n2 == arr2.size
            matched = bool(run_in_jax(n, arr1, arr2))
    if matched:
        print('OK')
    else:
        print('MISMATCHED!!!')
        sys.exit(1)


if __name__ == "__main__":
    main()
