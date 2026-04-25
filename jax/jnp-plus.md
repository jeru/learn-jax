# Functions in `jax.lax` that feel like `jax.numpy` but not there

As a more primitive layer that `jax.numpy` is built upon, `jax.lax` has a lot of basic functions already in `jax.numpy`, like `cos`, `exp`, etc.

However, there are some function in `jax.lax` that feels like on the same primitivity-level as `jax.numpy` but is not there.

## Math functions

Incomplete list, extracted from the [reference page](https://docs.jax.dev/en/latest/jax.lax.html#operators).

[`clz`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.clz.html#jax.lax.clz "jax.lax.clz")(x)
: count leading zeros as bits.

[`cumlogsumexp`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cumlogsumexp.html#jax.lax.cumlogsumexp "jax.lax.cumlogsumexp")(operand[, axis, reverse])
: cummulative version of log-sum-exp.

[`cummax`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cummax.html#jax.lax.cummax "jax.lax.cummax")(operand[, axis, reverse])
: cummulative versoin of max.

[`fft`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.fft.html#jax.lax.fft "jax.lax.fft")(x, fft_type, fft_lengths)

[`top_k`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html#jax.lax.top_k "jax.lax.top_k")(operand, k, *[, axis])
: top `k` elements from `operand`.

## Control flow

From [this sections](https://docs.jax.dev/en/latest/jax.lax.html#control-flow-operators).

### Real if-then-else

[`cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html#jax.lax.cond "jax.lax.cond")(pred, true_fun, false_fun, *operands[, ...])
[`switch`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html#jax.lax.switch "jax.lax.switch")(index, branches, *operands[, operand])
: select one _function_ to run. This cannot be achieved by `jax.numpy.where` because the latter computes both branches despite the condition. So the real if-then-else needs `cond` instead (only one branch runs).

### Sequential/semi-sequential aggregation

[`scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan "jax.lax.scan")(f, init[, xs, length, reverse, unroll, ...])
: `f` takes `(carry, x)` and returns `(new_carry, y)`; and `scan` did this over the whole `xs` and returns `(final_carry, ys)`. Totally sequential.

[`associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html#jax.lax.associative_scan "jax.lax.associative_scan")(fn, elems[, reverse, axis])
: 
performs a scan with an associative binary operation, in parallel.
Returns the _prefix_ sum/max/whatever-fn-is.
But still needs O(log N) rounds. (Side note: `jax.numpy.sum` should also be O(log N) rounds instead of O(1) rounds.)

### Sequential looping

The most general version of looping, but totally sequential.
Use with caution obviously (no parallelism across the loop).
Might be good to save memory, or if the operation itself is sequential anyway.

[`map`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html#jax.lax.map "jax.lax.map")(f, xs, *[, batch_size])
: solely parallism management (there's even a `batch_size` to control it). Consider `jax.vmap` if the sequentialty isn't deliberate.

[`fori_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html#jax.lax.fori_loop "jax.lax.fori_loop")(lower, upper, body_fun, init_val, *)
[`while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html#jax.lax.while_loop "jax.lax.while_loop")(cond_fun, body_fun, init_val)
: sequential for-loop and while-loop.

### Can ignore

[`select`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.select.html#jax.lax.select "jax.lax.select")(pred, on_true, on_false)
[`select_n`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.select_n.html#jax.lax.select_n "jax.lax.select_n")(which, *cases)
: these don't matter. Use `jax.numpy.where` or so. The latter are built upon these. No uncovered capability.

## Linear algebra stuff

[These](https://docs.jax.dev/en/latest/jax.lax.html#module-jax.lax.linalg).

But, should probably check `jax.scipy` first.
