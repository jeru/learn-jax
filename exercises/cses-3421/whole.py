# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/3421
#
# Break and resume via grain+orbax.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial
import os.path
import pickle
from typing import Sequence

from absl import app
import grain
import grain.python
import grain.transforms
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int64
import numpy
import orbax.checkpoint as ocp

import my_lib
from my_lib.tests_data_source import (
    TestsDataSource,
    INPUT_FIELDNAME,
    ANSWER_FIELDNAME,
)


@jax.jit
def solve_in_jax(a: Int64[Array, 'B N'],
                 a_mask: Bool[Array, 'B N']) -> Int64[Array, ' B']:
    n = a.shape[1]
    # Repad.
    INVALID: jnp.int64 = 10**10
    a = jnp.where(a_mask, a, jnp.full_like(a, INVALID))

    # Row by row. `jnp.unique`'s axis handling doesn't match what we need.
    def row_unique_values_and_counts(row):
        return jnp.unique(row, return_counts=True, size=n, fill_value=0)
    values, counts = jax.vmap(row_unique_values_and_counts, in_axes=0)(a)

    counts = jnp.where(values == INVALID, jnp.zeros_like(a), counts)

    MOD: jnp.int64 = 10**9 + 7
    @partial(jnp.ufunc, nin=2, nout=1)
    def mul(a, b): return a * b % MOD
    # Exclude the empty solution.
    return (mul.reduce(counts + 1, axis=1) - 1) % MOD


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
        assert str(out[i]) == ans, f'{out[i]} vs {ans} from {batch["arr"][i]} {batch["arr/mask"][i]} {sum(batch["arr/mask"][i])}'


def main(argv):
    ds = TestsDataSource(argv[1])
    dl = grain.load(ds, num_epochs=2, transformations=[
        ParseInput(),
        my_lib.batch_with_masked_power2_padding(2, jax_field_names=['arr']),
    ])
    mngr = ocp.CheckpointManager(os.path.join(argv[1], 'ckpt'))

    dl_iter = iter(dl)
    num = 0
    if mngr.latest_step():
        num = mngr.latest_step()
        mngr.restore(num, args=grain.python.PyGrainCheckpointRestore(dl_iter))
    for batch in dl_iter:
        print(f'Testing {num}...')
        solve_and_check(batch)
        num += 1
        if num % 3 == 0:
            mngr.save(num, args=grain.python.PyGrainCheckpointSave(dl_iter))
            mngr.wait_until_finished()
            print('Take a break. Please rerun to resume.')
            return
    print(f'All {num} tests passed.')


if __name__ == "__main__":
    app.run(main)
