# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1074
#
# Add batching.

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

from my_lib.tests_data_source import (
    TestsDataSource,
    INPUT_FIELDNAME,
    ANSWER_FIELDNAME,
)


@jax.jit
def solve_in_jax(a: Int64[Array, 'B N'],
                 a_mask: Bool[Array, 'B N']) -> Int64[Array, ' B']:
    # It's actually tricker to take median from arrays of different lengths,
    # now padded up.
    ns = jnp.sum(a_mask.astype(jnp.int64), axis=1)  # (B,)
    middles = (a.shape[1] + ns) // 2
    # Replace the pads with half of -inf and half of +inf.
    rng = jnp.arange(a.shape[1])[None, :]  # (1, N)
    i64info = jnp.iinfo(jnp.int64)
    newly_padded = jnp.where(
            rng < ns[:, None], a,
            jnp.where(rng < middles[:, None], i64info.min, i64info.max))
    median = jnp.percentile(newly_padded, 50, method='lower', axis=1)
    # Cost is distance to median; ignore padded.
    return jnp.sum(jnp.abs(a - median[:, None]) * a_mask,
                   axis=1)


class ParseInput(grain.transforms.Map):
    def map(self, record: dict) -> dict:
        content = record[INPUT_FIELDNAME]
        nums = content.split()
        record['arr'] = numpy.fromiter(map(int, nums[1:]), dtype=jnp.int64)
        return record


class BatchPaddingPower2WithMask:
    """Unify the batched array sizes a little.

    Two duties:
    1. Align shapes inside the batch.
    2. Make the total number of possible shapes small so `jax.jit` doesn't
       compile too many versions.
    """

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def __call__(self, ds: Sequence[dict]) -> dict:
        new_dict = {}
        # Fields to be kept as is. Not sent to the jitted function anyway.
        for name in [INPUT_FIELDNAME, ANSWER_FIELDNAME]:
            new_dict[name] = [d[name] for d in ds]

        # 'arr' should be padded up.
        n0 = max(d['arr'].shape[0] for d in ds)
        n = 1
        while n < n0: n *= 2
        new_arr = [numpy.pad(d['arr'], (0, n - d['arr'].shape[0])) for d in ds]
        new_arr = numpy.stack(new_arr)
        new_arr = numpy.pad(new_arr, [(0, 0), (0, self._batch_size - len(ds))])
        new_dict['arr'] = new_arr

        mask = numpy.stack([numpy.arange(n) < d['arr'].shape[0] for d in ds])
        mask = numpy.pad(mask, [(0, 0), (0, self._batch_size - len(ds))],
                         constant_values=False)
        new_dict['arr_mask'] = mask
        return new_dict


def solve_and_check(batch: dict):
    out = solve_in_jax(batch['arr'], batch['arr_mask'])
    out = list(int(x) for x in out)
    for i, ans in enumerate(batch[ANSWER_FIELDNAME]):
        ans = ans.strip()
        assert str(out[i]) == ans


def main(argv):
    ds = TestsDataSource(argv[1])
    batch_size = 2
    dl = grain.load(ds, num_epochs=2, transformations=[
        ParseInput(),
        grain.transforms.Batch(batch_size,
                               batch_fn=BatchPaddingPower2WithMask(batch_size)),
    ])
    num = 0
    for batch in dl:
        print(f'Testing {num}...')
        solve_and_check(batch)
        num += 1
    print(f'All {num} tests passed.')


if __name__ == "__main__":
    app.run(main)
