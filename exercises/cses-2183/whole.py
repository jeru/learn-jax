# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/2183
#
# Batching function in library.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import os.path
from typing import Sequence

from absl import app
import grain
import grain.transforms
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int64
import numpy

import my_lib
from my_lib.tests_data_source import (
    TestsDataSource,
    INPUT_FIELDNAME,
    ANSWER_FIELDNAME,
)


@jax.jit
def solve_in_jax(a: Int64[Array, 'B N'],
                 a_mask: Bool[Array, 'B N']) -> Int64[Array, ' B']:
    # Sort, but keep pad at the end.
    a = jnp.sort(jnp.where(a_mask, a, jnp.iinfo(jnp.int64).max), axis=1)
    # Same size: each row remove tail and prepend 0.
    a0 = jnp.insert(jnp.delete(a, -1, axis=1), 0, 0, axis=1)
    sum1 = jnp.add.accumulate(a0, axis=1) + 1
    # missed[b][i]: a[b][i] is too large to be the next of sum(a[b][:i]).
    missed = (sum1 < a) & a_mask
    # Tie breaking of `jnp.argmax`: smallest index.
    idx_if_any = jnp.argmax(missed, axis=1)
    soln_if_any = sum1[jnp.arange(a.shape[0]), idx_if_any]
    # If no a[b][i] missed, then the next missed number is sum(a[b]) + 1.
    soln_otherwise = jnp.sum(a * a_mask, axis=1) + 1
    return jnp.where(jnp.any(missed, axis=1), soln_if_any, soln_otherwise)


class ParseInput(grain.transforms.Map):
    def map(self, record: dict) -> dict:
        content = record[INPUT_FIELDNAME]
        nums = content.split()
        record['arr'] = numpy.fromiter(map(int, nums[1:]), dtype=jnp.int64)
        return record


def solve_and_check(batch: dict):
    out = solve_in_jax(batch['arr'], batch['arr/mask'])
    out = list(int(x) for x in out)
    for i, ans in enumerate(batch[ANSWER_FIELDNAME]):
        ans = ans.strip()
        assert str(out[i]) == ans


def main(argv):
    ds = TestsDataSource(argv[1])
    dl = grain.load(ds, num_epochs=2, transformations=[
        ParseInput(),
        my_lib.batch_with_masked_power2_padding(2, jax_field_names=['arr']),
    ])
    num = 0
    for batch in dl:
        print(f'Testing {num}...')
        solve_and_check(batch)
        num += 1
    print(f'All {num} tests passed.')


if __name__ == "__main__":
    app.run(main)
