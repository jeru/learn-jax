# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# Custom answer checker for 
#
# Check the following:

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial

import jax
from jax import numpy as jnp
from jax import lax
from jaxtyping import Array, Int64, Bool


MAX_N = 100


@jax.jit
def run_in_jax(plans: Int64[Array, "C 3"],
               ans: Bool[Array, "C"],
               outs: Int64[Array, "C 2 M"],
) -> Bool[Array, "1"]:
    # (C, M)
    p1, p2 = jnp.unstack(outs, axis=1)
    # (C,)
    no_score = p1[:, 0] == -1
    score_p1 = jnp.count_nonzero(p1 > p2, axis=1)
    score_p2 = jnp.count_nonzero(p2 > p1, axis=1)
    n, a, b = jnp.unstack(plans, axis=1)
    score_matched = jnp.logical_and(score_p1 == a, score_p2 == b)

    return jnp.all(jnp.where(ans, score_matched, no_score))


def parse_and_check_array(n: int, line: str) -> list[int]:
    a = [int(x) - 1 for x in line.split()]
    for x in a:
        assert x >= 0
        assert x < n
    assert len(a) == n
    assert len(set(a)) == n
    return a


def main():
    import numpy
    import sys
    infile, outfile, ansfile = sys.argv[1:]
    with open(infile) as f:
        n = int(f.readline())
        plans = numpy.array([f.readline().split() for _ in range(n)],
                            dtype=jnp.int64)
        for p in plans:
           assert p[0] <= MAX_N
    with open(ansfile) as f:
        ans = []
        for _ in range(n):
            if f.readline().strip() == 'YES':
                ans.append(True)
                f.readline()
                f.readline()
            else:
                ans.append(False)
        ans = jnp.array(ans, dtype=jnp.bool_)
    with open(outfile) as f:
        outs = []
        for i in range(n):
            key = f.readline().strip()
            assert key in ['YES', 'NO']
            if key == 'YES':
                a = parse_and_check_array(plans[i][0], f.readline())
                b = parse_and_check_array(plans[i][0], f.readline())
            else:
                a = [-1] * plans[i][0]
                b = a
            a = a + [-1] * (MAX_N - len(a))
            b = b + [-1] * (MAX_N - len(b))
            outs.append(jnp.array([a, b], dtype=jnp.int64))
        outs = jnp.stack(outs)
    verify = bool(run_in_jax(plans, ans, outs))
    if verify:
        print('OK')
    else:
        print('FAILED!!!')
        sys.exit(1)


if __name__ == "__main__":
    main()
