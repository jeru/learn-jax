# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1640
#
# First try on Transformation.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import os.path

import grain
import grain.transforms
import jax
from jax import numpy as jnp
from jaxtyping import Array, Int64
import numpy

from my_lib.tests_data_source import (
    TestsDataSource,
    INPUT_FIELDNAME,
    ANSWER_FIELDNAME,
)


INVALID: jnp.int64 = -10**10


@jax.jit
def solve_in_jax(s: jnp.int64, a: Int64[Array, ' N']) -> Int64[Array, '2']:
    has_rep_soln = (s % 2 == 0) & (jnp.count_nonzero(a == s // 2) >= 2)
    rep_soln = jnp.nonzero(a == s // 2, size=2)[0]

    # x+y = s, x != y. Reverse large half of the (unique) values and see
    # whether any overlap with the smaller half.
    values = jnp.unique(a, size=a.shape[0], fill_value=INVALID)
    half_reversed = jnp.where(values <= s // 2, values, s - values)
    values2, counts2 = jnp.unique(half_reversed, return_counts=True,
                                  size=a.shape[0], fill_value=INVALID)
    dist_mask = (values2 != INVALID) & (counts2 >= 2)
    has_dist_soln = jnp.any(dist_mask)
    dist_num = values2[jnp.nonzero(dist_mask, size=1)]
    dist_soln = jnp.concat([jnp.nonzero(a == dist_num, size=1)[0],
                            jnp.nonzero(a == s - dist_num, size=1)[0]])

    return jnp.where(has_rep_soln, rep_soln,
                     jnp.where(has_dist_soln, dist_soln, INVALID))


class ParseInput(grain.transforms.Map):
    def map(self, record: dict) -> dict:
        content = record[INPUT_FIELDNAME]
        nums = content.split()
        record['s'] = int(nums[1])
        record['arr'] = numpy.fromiter(map(int, nums[2:]), dtype=jnp.int64)
        return record


class PadArrayPower2(grain.transforms.Map):
    """Unify the array size a little.

    `jax.jit` compiles one version for each different shape. Make it easier."""

    def __init__(self, fill_value):
        self.fill_value = fill_value

    def map(self, record: dict) -> dict:
        arr = record['arr']
        n = 1
        while n < arr.shape[0]: n *= 2
        arr = numpy.pad(arr, (0, n - arr.shape[0]),
                        constant_values=self.fill_value)
        record['arr'] = arr
        return record


def solve(record: dict) -> str:
    s = record['s']
    arr = record['arr']
    ans = solve_in_jax(s, arr)
    ans = list(map(int, ans))
    if ans[0] == INVALID:
        return 'IMPOSSIBLE'
    else:
        return f'{ans[0] + 1} {ans[1] + 1}'


def solve_and_check(record: dict):
    out = solve(record)
    ans = record[ANSWER_FIELDNAME].strip()
    if out == ans: return

    arr = record['arr']
    s = record['s']

    assert (out == 'IMPOSSIBLE') == (ans == 'IMPOSSIBLE')
    out1, out2 = map(int, out.split())
    assert out1 != out2

    sum_out = arr[out1 - 1] + arr[out2 - 1]
    assert sum_out == s


def main():
    import sys
    ds = TestsDataSource(sys.argv[1])
    dl = grain.load(ds, num_epochs=2, transformations=[
        ParseInput(),
        PadArrayPower2(INVALID),
    ])
    num = 0
    for record in dl:
        print(f'Testing {num}...')
        solve_and_check(record)
        num += 1
    print(f'All {num} tests passed.')


if __name__ == "__main__":
    main()
