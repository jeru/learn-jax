# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1755
#
# Shuffle a string to make it a palindrome.
#
# Basically bin-counting, then check whether there is at most one bin with an
# odd count.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"


import jax
from jax import numpy as jnp
from jaxtyping import Array, Int8


@jax.jit
def run_in_jax(a: Int8[Array, ' N']) -> Int8[Array, ' N']:
    no_solution = jnp.full_like(a, -1)

    freqs = jnp.bincount(a, length=ord('Z') + 1)
    # Best effort to construct half of the palindrome assuming valid.
    # Behavior if not enough repetition is given (i.e., not valid) is to repeat
    # the final value, so no harm.
    half_seq = jnp.repeat(jnp.arange(freqs.shape[0]), repeats=freqs // 2,
                          total_repeat_length=a.shape[0] // 2)
    # Shape is static, so fine to do if-else.
    if a.shape[0] % 2 == 0:
        yes_solution = jnp.concat([half_seq, jnp.flip(half_seq)])
        # Even length. Requires all counts even.
        return jnp.where(jnp.all(freqs % 2 == 0), yes_solution, no_solution)
    else:
        middle, = jnp.nonzero(freqs % 2 == 1, size=1)
        yes_solution = jnp.concat([half_seq, middle, jnp.flip(half_seq)])
        # Odd length. Requires exactly one count odd.
        return jnp.where(jnp.count_nonzero(freqs % 2 == 1) == 1, yes_solution,
                         no_solution)


def main():
    import numpy
    import sys
    s = sys.stdin.readline().strip()
    a = jnp.array(list(ord(x) for x in s), dtype=jnp.int8)
    ans = run_in_jax(a)
    ans = numpy.asarray(ans)
    if numpy.any(ans < 0):
        print('NO SOLUTION')
    else:
        print(''.join(map(chr, ans)))


if __name__ == "__main__":
    main()
