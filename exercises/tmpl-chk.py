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
from jaxtyping import Array, Int64, Bool

@partial(jax.jit, static_argnames=["n"])
def run_in_jax(n: int, a: Int64[Array, " N"]) -> Bool[Array, "1"]:
    pass


def main():
    import sys
    infile, outfile, ansfile = sys.argv[1:]
    with open(infile) as f:
        del f
    with open(outfile) as f:
        del f
    with open(ansfile) as f:
        del f
    verify = bool(run_in_jax(...))
    if verify:
        print('OK')
    else:
        print('FAILED!!!')
        sys.exit(1)


if __name__ == "__main__":
    main()
