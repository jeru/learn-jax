# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1069
#
# Count the longest consecutive substring with only one character.
#
# This one actually has a quite efficient parallel solution:
# 1. For each position i such that S[i - 1] != S[i] (including i the beginning),
#    mark i as a starting position.
# 2. For each position such that S[i + 1] != S[i] (including i the ending), mark
#    i as an ending position.
# 3. Can be proven all starting positions and all ending positions form two
#    parallel arrays, i.e., starting[i] corresponds to ending[i].

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
from jax import numpy as jnp
from jax import lax
from jaxtyping import Array, UInt4, Int64


CHAR_A: UInt4 = 0
CHAR_C: UInt4 = 1
CHAR_G: UInt4 = 2
CHAR_T: UInt4 = 3
INVALID: UInt4 = 4
CHAR_MAP = {'A': CHAR_A, 'C': CHAR_C, 'G': CHAR_G, 'T': CHAR_T}


@jax.jit
def run_in_jax(a: UInt4[Array, 'N']) -> Int64[Array, '1']:
    pad = jnp.array([INVALID], dtype=jnp.uint4)

    left_shifted = jnp.concatenate([pad, jnp.delete(a, -1)])
    starting_mask = (a != left_shifted)

    right_shifted = jnp.concatenate([jnp.delete(a, 0), pad])
    ending_mask = (a != right_shifted)

    indices = jnp.indices(a.shape, dtype=jnp.int64)
    # Collect all the indices of starting positions and ending positions to
    # the beginning of each array; remaining arrays are filled with the same
    # value so will only contribute zeros after subtraction to the max.
    starting_positions = jnp.compress(starting_mask, indices, size=a.shape[0],
                                      fill_value=-1)
    ending_positions = jnp.compress(ending_mask, indices, size=a.shape[0],
                                    fill_value=-1)
    return jnp.max(ending_positions - starting_positions) + 1


def main():
    import sys
    seq = sys.stdin.readline().strip()
    seq_arr = [CHAR_MAP[x] for x in seq]
    ans = run_in_jax(jnp.array(seq_arr, dtype=jnp.uint4))
    print(int(ans))


if __name__ == "__main__":
    main()
