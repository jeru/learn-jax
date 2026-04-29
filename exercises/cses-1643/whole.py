# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1643
#
# Add input masks.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import os.path

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
def solve_in_jax(a: Int64[Array, ' N'],
                 a_mask: Bool[Array, ' N']) -> Int64[Array, '1']:
    LO = jnp.iinfo(jnp.int64).min
    prev_sum = jnp.concat([jnp.zeros((1,)), jnp.add.accumulate(a)])
    prev_sum_min = jnp.minimum.accumulate(prev_sum)
    # gap[i] = (a[0] + ... + a[i]) - min_{j<i}(a[0] + ... + a[j]}
    gap = jnp.delete(prev_sum, 0) - jnp.delete(prev_sum_min, -1)
    masked_gap = jnp.where(a_mask, gap, LO)
    return jnp.max(masked_gap)


class ParseInput(grain.transforms.Map):
    def map(self, record: dict) -> dict:
        content = record[INPUT_FIELDNAME]
        nums = content.split()
        record['arr'] = numpy.fromiter(map(int, nums[1:]), dtype=jnp.int64)
        return record


class PadArrayPower2WithMask(grain.transforms.Map):
    """Unify the array size a little.

    `jax.jit` compiles one version for each different shape. Make it easier."""

    def map(self, record: dict) -> dict:
        arr = record['arr']
        n0 = arr.shape[0]
        n = 1
        while n < n0: n *= 2
        arr = numpy.pad(arr, (0, n - n0))
        mask = numpy.arange(n) < n0
        record['arr'] = arr
        record['arr_mask'] = mask
        return record


def solve(record: dict) -> str:
    arr = record['arr']
    arr_mask = record['arr_mask']
    ans = solve_in_jax(arr, arr_mask)
    return str(int(ans))


def solve_and_check(record: dict):
    out = solve(record)
    ans = record[ANSWER_FIELDNAME].strip()
    if out == ans: return
    assert out == ans


def main():
    import sys
    ds = TestsDataSource(sys.argv[1])
    dl = grain.load(ds, num_epochs=2, transformations=[
        ParseInput(),
        PadArrayPower2WithMask(),
    ])
    num = 0
    for record in dl:
        print(f'Testing {num}...')
        solve_and_check(record)
        num += 1
    print(f'All {num} tests passed.')


if __name__ == "__main__":
    main()
