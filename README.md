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

Start with consumer side only and do some extra exercises by solving algorithmic problems, input handling via a custom `DatasetIterator` through a bunch of text files. Then move the usage to closer and closer to what real training data processing looks like (randomized, sharded, pause and resume, etc.). Listed [here](exercises/jax-grain1.md).

