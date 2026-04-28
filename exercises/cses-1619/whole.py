# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1619
#
# Move the DataSource to a separate library.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import os.path

import grain
import jax
from jax import numpy as jnp
from jaxtyping import Array, Int64
import numpy

from my_lib.tests_data_source import (
    TestsDataSource,
    INPUT_FIELDNAME,
    ANSWER_FIELDNAME,
)


@jax.jit
def solve_in_jax(a: Int64[Array, ' N']) -> Int64[Array, '1']:
    n = a.shape[0]
    signed_times = jnp.sort(a * 2 + jnp.arange(n) % 2)
    signs = jnp.where(signed_times < 2, 0,
                      jnp.where(signed_times % 2 == 0, +1, -1)
                      ).astype(jnp.int64)
    return jnp.max(jnp.add.accumulate(signs))


def solve(content: str) -> str:
    lines = content.strip().split()
    arr = jnp.array(numpy.fromiter(map(int, lines[1:]), dtype=jnp.int64))
    assert arr.shape[0] % 2 == 0

    n = 1
    while n < arr.shape[0]: n *= 2
    # Limit the variety of array lengths to not explode the number of copies
    # that jax.jit produces.
    arr = jnp.pad(arr, (0, n - arr.shape[0]), constant_values=0)

    s = solve_in_jax(arr)
    return str(int(s))


def solve_and_check(record: dict):
    out = solve(record[INPUT_FIELDNAME])
    ans = record[ANSWER_FIELDNAME]
    assert out == ans.strip(), f'{out} vs {ans}'


def main():
    import sys
    ds = TestsDataSource(sys.argv[1])
    dl = grain.load(ds, num_epochs=2)
    num = 0
    for record in dl:
        print(f'Testing {num}...')
        solve_and_check(record)
        num += 1
    print(f'All {num} tests passed.')


if __name__ == "__main__":
    main()
