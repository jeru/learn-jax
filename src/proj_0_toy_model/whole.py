# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0
#
# First version: everything in raw jax.

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax.nn.initializers
from jax import numpy as jnp
import jax.random
from jaxtyping import Array, Float


def linear(x: Float[Array, " D"], *, w: Float[Array, "D R"],
           b: Float[Array, " R"]) -> Float[Array, " R"]:
    return x * w + b

def model(x: Float[Array, " D"], params: dict) -> Float[Array, " R"]:
    w: Float[Array, "D R"] = params["w"]
    b: Float[Array, " R"] = params["b"]
    return linear(x, w=w, b=b)


def ground_truth(x: Float[Array, " D"]) -> Float[Array, " R"]:
    return x * 3 + 5


@jax.jit
def loss(x: Float[Array, " D"], params: dict) -> float:
    value = model(x, params)
    expected = ground_truth(x)
    diff = value - expected
    # Square L2 norm as loss function.
    return jnp.sum(diff * diff)


def main():
    rng_key = jax.random.key(42)
    initer = jax.nn.initializers.uniform(2)
    params = {
        "w": initer(rng_key, (2, 2), jnp.float32),
        "b": initer(rng_key, (2,), jnp.float32),
    }
    # Compute the gradient against the params, not `x`.
    value_and_grad_fn = jax.value_and_grad(loss, argnums=1)
    lr = 1e-2
    x_sampler = jax.nn.initializers.uniform(10)
    # Feed in random "training" data.
    for step in range(1000):
        rng_key, rng_subkey = jax.random.split(rng_key)
        x = x_sampler(rng_subkey, (2,), jnp.float32)
        vg = value_and_grad_fn(x, params)
        if step % 20 == 0:
            print(f'Step {step}: x={x}, loss={vg[0]}, grad={vg[1]}')
        # Naive gradient descent.
        for k, v in vg[1].items():
            params[k] -= v * lr
    print(f'Final: w={params["w"]}, b={params["b"]}')


if __name__ == "__main__":
    main()
