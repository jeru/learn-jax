# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
#

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial

import jax
from jax import numpy as jnp
from jax import lax
from jaxtyping import Array, Int64

@partial(jax.jit, static_argnames=["n"])
def run_in_jax(n: int, a: Int64[Array, "NMinusOne"]) -> Int64[Array, "1"]:
    pass


def main():
    import numpy
    import sys
    n = int(sys.stdin.readline())
    ans = run_in_jax(n, ...)
    ans = ' '.join(map(str, numpy.asarray(ans)))
    print(ans)


if __name__ == "__main__":
    main()
