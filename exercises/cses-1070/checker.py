# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# Output verifier for https://cses.fi/problemset/task/1070 given the output is
# not unique and no official verifier is provided.
#
# Need to check the ground-truth file whether it is a "NO SOLUTION". If not,
# go to check the test output by:
# 1. Array size matches N.
# 2. Each of 1 to N exactly once.
# 3. No two adjcent numbers have adjacent values.
#
# Actually the checker sounds more involving than the solution with jax!

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial

import jax
from jax import numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Int64


NO_SOL_TEXT = 'NO SOLUTION'


@partial(jax.jit, static_argnames=['n'])
def run_in_jax(n: int, a: Int64[Array, 'N']) -> Bool[Array, '1']:
    size_matched = a.shape == (n,)
    elements_matched = jnp.bincount(a, length=n + 1) == (
            jnp.concat([jnp.zeros((1,)), jnp.ones((n,))]))
    distances_far = jnp.abs(jnp.delete(a, 0) - jnp.delete(a, -1)) > 1
    return jnp.logical_and(
            size_matched, jnp.logical_and(
                jnp.all(elements_matched),
                jnp.all(distances_far)))


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
            arr = [int(x) for x in f.readline().split()]
            matched = bool(run_in_jax(n, jnp.array(arr, dtype=jnp.int64)))
    if matched:
        print('OK')
    else:
        print('MISMATCHED!!!')
        sys.exit(1)


if __name__ == "__main__":
    main()
