# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0

# https://cses.fi/problemset/task/1068/
#
# Basically Collatz conjecture simulation for up to n=1e6.
#
# A straightforward while-loop use-case if in plain CPU languages. Complicated
# by:
# - jnp.while_loop requires its body_fn return the same (static) type of its
#   input, i.e., no dynamic array appending values.
# - jnp.whlie_loop requires its cond_fn return a Bool instead of a jax.Array,
#   i.e., the condition, thus the whole loop, is actually statically shaped.
#
# So the strategy: each call to the jax block computes a fixed number of steps;
# let the external python caller take care of the dynamic part.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
from jax import numpy as jnp
from jax import lax
from jaxtyping import Array, Int64


# Set to a small value for debugging.
BLOCK_SIZE = 5


NEXT_FN = lambda n: jnp.where(n < 2,
                              0,
                              jnp.where(n % 2 == 1, n * 3 + 1, n // 2))

######################################################################
# Scenario 1: `n` is provided asynchronously (the "data" flavor).
######################################################################


@jax.jit
def run_in_jax1(n: Int64[Array, "1"]) -> Int64[Array, "SeqLen"]:
    # Fixed size sequence to be filled.
    seq = jnp.concatenate([jnp.expand_dims(n, axis=0),
                           jnp.full((BLOCK_SIZE,), -1)])
    def body_fn(idx, seq):
        return seq.at[idx + 1].set(NEXT_FN(seq[idx]))
    computed_seq = lax.fori_loop(0, BLOCK_SIZE, body_fn, seq)
    return jnp.delete(computed_seq, 0)


def orchestrate1(n):
    seq = [n]
    while seq[-1] >= 2:
        next_block = run_in_jax1(jnp.array(seq[-1], dtype=jnp.int64))
        seq.extend(int(x) for x in next_block)
    while seq[-1] < 1: seq.pop()
    return seq


######################################################################
# Scenario 2: `n` is provided synchronously (the "parameter" flavor).
######################################################################


@jax.jit
def run_in_jax2(n: int) -> Int64[Array, "SeqLen"]:
    seq = jnp.concatenate([jnp.expand_dims(n, axis=0),
                           jnp.full((BLOCK_SIZE,), -1)])
    def cond_fn(val):
        idx, _ = val
        return idx < BLOCK_SIZE
    def body_fn(val):
        idx, seq = val
        return (idx + 1, seq.at[idx + 1].set(NEXT_FN(seq[idx])))
    _, computed_seq = lax.while_loop(cond_fn, body_fn, (0, seq))
    return jnp.delete(computed_seq, 0)


def orchestrate2(n):
    seq = [n]
    while seq[-1] >= 2:
        next_block = run_in_jax2(seq[-1])
        seq.extend(int(x) for x in next_block)
    while seq[-1] < 1: seq.pop()
    return seq


def main():
    import sys
    n = int(sys.stdin.readline())
    if sys.argv[1] == 'opt1':
        seq = orchestrate1(n)
    elif sys.argv[1] == 'opt2':
        seq = orchestrate2(n)
    else:
        raise ValueError(f'Unrecognized: {sys.argv[1]}')
    print(" ".join(str(x) for x in seq))


if __name__ == "__main__":
    main()
