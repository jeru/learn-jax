# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# https://cses.fi/problemset/task/3419

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import collections
from functools import partial
import glob
import os.path
from typing import Any

import grain
import jax
from jax import numpy as jnp
from jax import lax
from jaxtyping import Array, Int64
import numpy


def _get_tests_basenames(directory, ext):
    files = glob.glob(os.path.join(directory, '*' + ext))
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in files)


class MultiTextTestDataset(grain.IterDataset[tuple[str, str]]):
    def __init__(self, directory):
        super().__init__()
        in_files = _get_tests_basenames(directory, '.in')
        ans_files = _get_tests_basenames(directory, '.out')
        assert in_files == ans_files
        self._file_bases = [os.path.join(directory, f) for f in in_files]

    def __iter__(self) -> grain.DatasetIterator[tuple[str, str]]:
        return _MultiTextTestIter(self._file_bases)


class _MultiTextTestIter(grain.DatasetIterator[tuple[str, str]]):
    def __init__(self, file_bases):
        super().__init__()
        self._file_bases = file_bases
        self._idx = 0

    def get_state(self) -> dict[str, Any]:
        return {'fb': self._file_bases, 'idx': self._idx}

    def set_state(self, state: dict[str, Any]):
        self._file_bases = state['fb']
        self._idx = state['idx']

    def __next__(self) -> tuple[str, str]:
        if self._idx >= len(self._file_bases): raise StopIteration
        fn = self._file_bases[self._idx]
        self._idx += 1
        with open(fn + '.in') as f:
            content1 = f.read()
        with open(fn + '.out') as f:
            content2 = f.read()
        return (content1, content2)


MAX_N = 100


@jax.jit
def solve_in_jax(n: int) -> Int64[Array, 'N']:
    a = jnp.arange(MAX_N)
    unmasked = jnp.expand_dims(a, 0) ^ jnp.expand_dims(a, 1)
    m1 = a < n
    mask = jnp.logical_and(jnp.expand_dims(m1, 0), jnp.expand_dims(m1, 1))
    return jnp.where(mask, unmasked, -1)


def solve(n: int) -> str:
    def row_to_str(row):
        if row[0] == -1: return ''
        return ' '.join(map(str, filter(lambda c: c != -1, row)))
    row_strs = map(row_to_str, numpy.asarray(solve_in_jax(n)))
    return '\n'.join(filter(lambda r: r, row_strs))


def sanitize(s: str) -> str:
    rows = s.split('\n')
    rows = [r.strip() for r in rows]
    rows = [r for r in rows if r]
    return '\n'.join(rows)


def main():
    import sys
    ds = MultiTextTestDataset(sys.argv[1])
    num = 0
    for c_in, c_ans in ds:
        n = int(c_in.strip())
        c_out = solve(n)
        assert sanitize(c_out) == sanitize(c_ans), f'\n{c_out}\n\n{c_ans}'
        num += 1
    print(f'All {num} tests passed.')


if __name__ == "__main__":
    main()
