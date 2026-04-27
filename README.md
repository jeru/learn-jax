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

Listed [here](exercises/jax-only.md).

## grain

Now add [`grain`](https://google-grain.readthedocs.io/en/latest/index.html) to the recipe.

This is the input data processing part, in charge of:
* reading data from files,
* shuffling them (gradient descend optimizers generally assume training data in random order for their math to work),
* transforming data format (pre-GPU exclusively),
* sharding,
* batching.
These are where the complexity is.
But the output side: a `DatasetIterator` that spit out batches of data and fed into the `@jax.jit`-wrapped function.
It is just a python iterator but resumable: its internal state can be dumped to disk and loaded back.

Start with consumer side only and do some extra exercises by solving algorithmic problems, _input handling via a custom `DatasetIterator` through a bunch of text files_. Listed [here](exercises/jax-grain1.md).
