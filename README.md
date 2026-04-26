# learn-jax
Learn about jax and related.

## Python environment
Managed by [`uv`](https://github.com/astral-sh/uv).

To start an interactive:

```bash
uv run python
```

## jax.numpy: (nearly) everything algorithmic

This is the main math / data-processing (i.e., algorithmic) interface.
[Went through](jnp/overview.md) all of the functions there.

## jax.lax (and jax.vmap)

A more primitive layer below `jax.numpy` (and `jax.scipi`).

It has something that should be, by application, in `jax.numpy` but cannot, presumably due to `numpy` interface consistency.
[Examples](jax/jnp-plus.md).

Of course, `jax.numpy` also has things that feel lower-level than it should, noticeably `ufunc` and `vectorize`. These feel more natural to stay with `jax.vmap` and all the `jax.lax`'s [parallel operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators).

On the other hand, these two functions don't feel to belong to `jax.lax`:
* [`stop_gradient`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html#jax.lax.stop_gradient "jax.lax.stop_gradient")(x)
  : this gradient is model training-specific, so not quite mathematically or logically focused. Feels better to be grouped with `jax.grad`.
* [`with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html#jax.lax.with_sharding_constraint "jax.lax.with_sharding_constraint")(x, shardings)
  : feels even farther. Maybe belong to an external library like [grain](https://google-grain.readthedocs.io/en/latest/)?

## Some algorithmic exercises

Pick some simple algorithm tasks (competition programming / interview questions).
They are designed for CPU solutions so mostly won't fit performance-wise (seriously sequential and dynamic).
But can provide enough variety to cover some basic uses.

All [here](exercises/overview.md).
