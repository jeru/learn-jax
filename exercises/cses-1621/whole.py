# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/1621
#
# Test the usage of higher-level APIs of grain.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import collections
from functools import partial
import glob
import itertools
import os.path
from typing import Any

import grain
import grain.sources
import jax
from jax import numpy as jnp
from jax import lax
from jaxtyping import Array, Int64
import numpy


def _get_tests_basenames(directory, ext):
    files = glob.glob(os.path.join(directory, '*' + ext))
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in files)


def _get_tests(directory):
    in_files = _get_tests_basenames(directory, '.in')
    ans_files = _get_tests_basenames(directory, '.out')
    assert in_files == ans_files
    return [os.path.join(directory, f) for f in in_files]


class MultiTextTestDataSource(
        grain.sources.RandomAccessDataSource[tuple[str, str]]):
    def __init__(self, directory):
        self._file_bases = _get_tests(directory)

    def __len__(self) -> int:
        return len(self._file_bases)

    def __getitem__(self, index: int) -> tuple[str, str]:
        fn = self._file_bases[index]
        with open(fn + '.in') as f:
            content1 = f.read()
        with open(fn + '.out') as f:
            content2 = f.read()
        return (content1, content2)


@jax.jit
def solve_in_jax(a: Int64[Array, 'N']) -> Int64[Array, '1']:
    a = jnp.unique(a, size=a.shape[0], fill_value=-1)
    return jnp.count_nonzero(a != -1)


def solve(content: str) -> str:
    lines = content.strip().split('\n')
    arr = jnp.array(numpy.fromiter(map(int, lines[1].split()),
                                   dtype=jnp.int64))
    n = 1
    while n < arr.shape[0]: n *= 2
    # Limit the variety of array lengths to not explode the number of copies
    # that jax.jit produces.
    arr = jnp.resize(arr, (n,))
    s = solve_in_jax(arr)
    return str(int(s))


def solve_and_check(content: str, ans: str):
    out = solve(content)
    assert out == ans.strip(), f'{out} vs {ans}'


def sanitize(s: str) -> str:
    rows = s.split('\n')
    rows = [r.strip() for r in rows]
    rows = [r for r in rows if r]
    return '\n'.join(rows)


def main():
    import sys
    ds = MultiTextTestDataSource(sys.argv[1])
    dl = grain.load(ds, num_epochs=2)
    num = 0
    for c_in, c_ans in dl:
        print(f'Testing {num}...')
        solve_and_check(c_in, c_ans)
        num += 1
    print(f'All {num} tests passed.')


if __name__ == "__main__":
    main()
