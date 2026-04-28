# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# Output verifier for https://cses.fi/problemset/task/1755 given the output is
# not unique and no official verifier is provided.
#
# Need to check the ground-truth file whether it is a "NO SOLUTION". If not,
# go to check the test output by:
# 1. Same composition of letters as input (i.e. frequency counts match).
# 2. Is a palindrome.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"


import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int8


NO_SOL_TEXT = 'NO SOLUTION'
MAX_CHAR = ord('Z') + 1


@jax.jit
def run_in_jax(in_str: Int8[Array, ' N'],
               out_str: Int8[Array, ' N']) -> Bool[Array, '1']:
    in_freqs = jnp.bincount(in_str, length=MAX_CHAR)
    out_freqs = jnp.bincount(out_str, length=MAX_CHAR)
    chars_matched = jnp.all(in_freqs == out_freqs)
    is_palindrome = jnp.all(out_str == jnp.flip(out_str))
    return jnp.logical_and(chars_matched, is_palindrome)


def str_to_jarr(s: str):
    return jnp.array([ord(x) for x in s], dtype=jnp.int8)


def main():
    import sys
    infile, outfile, ansfile = sys.argv[1:]
    with open(infile) as f:
        in_str = f.readline().strip()
    with open(ansfile) as f:
        no_solution = f.readline().strip() == NO_SOL_TEXT
    with open(outfile) as f:
        out = f.readline().strip()
        if no_solution:
            matched = out == NO_SOL_TEXT
        else:
            matched = bool(run_in_jax(str_to_jarr(in_str), str_to_jarr(out)))
    if matched:
        print('OK')
    else:
        print('MISMATCHED!!!')
        sys.exit(1)


if __name__ == "__main__":
    main()
