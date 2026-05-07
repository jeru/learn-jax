# A toy model

## Version 1: raw jax-based.

[Code](v1_all_in_one.py). Noticeably,
```python
@jax.jit
def loss(x, params): ...

value_and_grad_fn = jax.value_and_grad(loss, argnums=1)
```
to setup to differentiate only the params.
Then the weights are updated manually with
```python
vg = value_and_grad_fn(x, params)
for k, v in vg[1].items():
    params[k] -= v * lr
```

## Version 2: generate "training" data with grain.

[Code](v2_with_grain.py). Used the grain `Dataset` interface to ease the generation.
The `DataLoader` needs a user supplying a random generator for each input data, so quite cumbersome to use.

Added functionality:
```python
class VecGen(grain.transforms.RandomMap):
    def random_map(self, element, rng: np.random.Generator):
        return jnp.array(rng.random((2,)) * 10, dtype=jnp.float32)
ds = grain.MapDataset.range(1000).random_map(VecGen(), seed=42)

for step, x in enumerate(ds):
    ...
```
