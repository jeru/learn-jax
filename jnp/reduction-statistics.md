# Reduction functions and statsitics

Note that the numpy [`ufunc`](https://numpy.org/doc/stable/reference/ufuncs.html) itself also provides some member methods in the reduction manner: `reduce`, `accumulate`, `outer`.

## General reduction

[`apply_along_axis`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.apply_along_axis.html#jax.numpy.apply_along_axis "jax.numpy.apply_along_axis")(func1d, axis, arr, *args, ...)
: general reduction on 1 axis.

[`apply_over_axes`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.apply_over_axes.html#jax.numpy.apply_over_axes "jax.numpy.apply_over_axes")(func, a, axes)
: generlizes `apply_along_axis` even more by allowing multiple axes given.

## Logic and comparison

[`all`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.all.html#jax.numpy.all "jax.numpy.all")(a[, axis, out, keepdims, where])
[`any`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.any.html#jax.numpy.any "jax.numpy.any")(a[, axis, out, keepdims, where])

[`array_equal`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_equal.html#jax.numpy.array_equal "jax.numpy.array_equal")(a1, a2[, equal_nan])
[`array_equiv`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array_equiv.html#jax.numpy.array_equiv "jax.numpy.array_equiv")(a1, a2)
: all are respectively pairwise equal.

[`allclose`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.allclose.html#jax.numpy.allclose "jax.numpy.allclose")(a, b[, rtol, atol, equal_nan])
: all are respectively pairwise close.

## Basic arithmetic

[`size`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.size.html#jax.numpy.size "jax.numpy.size")(a[, axis])
: along the axis, simply count the elements.

[`count_nonzero`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.count_nonzero.html#jax.numpy.count_nonzero "jax.numpy.count_nonzero")(a[, axis, keepdims])
: along the axis, count how many nonzeros.

[`sum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum")(a[, axis, dtype, out, keepdims, ...])
: sum along the axis.

[`prod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.prod.html#jax.numpy.prod "jax.numpy.prod")(a[, axis, dtype, out, keepdims, ...])
: product along the axis.

[`trapezoid`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.trapezoid.html#jax.numpy.trapezoid "jax.numpy.trapezoid")(y[, x, dx, axis])
: integrate by treating `x -> y` as piecewise linear function.

## Statistics

[`cumulative_sum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cumulative_sum.html#jax.numpy.cumulative_sum "jax.numpy.cumulative_sum")(x, /, *[, axis, dtype, ...])
[`cumsum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cumsum.html#jax.numpy.cumsum "jax.numpy.cumsum")(a[, axis, dtype, out])
[`cumulative_prod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cumulative_prod.html#jax.numpy.cumulative_prod "jax.numpy.cumulative_prod")(x, /, *[, axis, dtype, ...])
[`cumprod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cumprod.html#jax.numpy.cumprod "jax.numpy.cumprod")(a[, axis, dtype, out])
: cummulative (prefix) sum/prod along the given axis.

[`average`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.average.html#jax.numpy.average "jax.numpy.average")(a[, axis, weights, returned, keepdims])
[`mean`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.mean.html#jax.numpy.mean "jax.numpy.mean")(a[, axis, dtype, out, keepdims, where])

[`median`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.median.html#jax.numpy.median "jax.numpy.median")(a[, axis, out, overwrite_input, keepdims])

[`max`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.max.html#jax.numpy.max "jax.numpy.max")(a[, axis, out, keepdims, initial, where])
[`amax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.amax.html#jax.numpy.amax "jax.numpy.amax")(a[, axis, out, keepdims, initial, where])
[`min`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.min.html#jax.numpy.min "jax.numpy.min")(a[, axis, out, keepdims, initial, where])
[`amin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.amin.html#jax.numpy.amin "jax.numpy.amin")(a[, axis, out, keepdims, initial, where])
: taking max/min along axis.

[`argmax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmax.html#jax.numpy.argmax "jax.numpy.argmax")(a[, axis, out, keepdims])
[`argmin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmin.html#jax.numpy.argmin "jax.numpy.argmin")(a[, axis, out, keepdims])
: index version of max/min.

[`percentile`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.percentile.html#jax.numpy.percentile "jax.numpy.percentile")(a, q[, axis, out, ...])

[`quantile`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.quantile.html#jax.numpy.quantile "jax.numpy.quantile")(a, q[, axis, out, overwrite_input, ...])

[`std`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.std.html#jax.numpy.std "jax.numpy.std")(a[, axis, dtype, out, ddof, keepdims, ...])

[`var`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.var.html#jax.numpy.var "jax.numpy.var")(a[, axis, dtype, out, ddof, keepdims, ...])

[`ptp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ptp.html#jax.numpy.ptp "jax.numpy.ptp")(a[, axis, out, keepdims])
: "peak-to-peak" difference along a given `axis` (default: no axis, whole n-dim array viewed flattened).

### Versions ignoring NaN

[`nanprod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanprod.html#jax.numpy.nanprod "jax.numpy.nanprod")(a[, axis, dtype, out, keepdims, ...])

[`nansum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nansum.html#jax.numpy.nansum "jax.numpy.nansum")(a[, axis, dtype, out, keepdims, ...])

[`nanmean`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmean.html#jax.numpy.nanmean "jax.numpy.nanmean")(a[, axis, dtype, out, keepdims, where])

[`nanmedian`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmedian.html#jax.numpy.nanmedian "jax.numpy.nanmedian")(a[, axis, out, overwrite_input, ...])

[`nanmax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmax.html#jax.numpy.nanmax "jax.numpy.nanmax")(a[, axis, out, keepdims, initial, where])

[`nanmin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmin.html#jax.numpy.nanmin "jax.numpy.nanmin")(a[, axis, out, keepdims, initial, where])

[`nanargmax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanargmax.html#jax.numpy.nanargmax "jax.numpy.nanargmax")(a[, axis, out, keepdims])

[`nanargmin`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanargmin.html#jax.numpy.nanargmin "jax.numpy.nanargmin")(a[, axis, out, keepdims])

[`nancumprod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nancumprod.html#jax.numpy.nancumprod "jax.numpy.nancumprod")(a[, axis, dtype, out])

[`nancumsum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nancumsum.html#jax.numpy.nancumsum "jax.numpy.nancumsum")(a[, axis, dtype, out])

[`nanpercentile`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanpercentile.html#jax.numpy.nanpercentile "jax.numpy.nanpercentile")(a, q[, axis, out, ...])

[`nanquantile`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanquantile.html#jax.numpy.nanquantile "jax.numpy.nanquantile")(a, q[, axis, out, ...])

[`nanstd`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanstd.html#jax.numpy.nanstd "jax.numpy.nanstd")(a[, axis, dtype, out, ddof, ...])

[`nanvar`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanvar.html#jax.numpy.nanvar "jax.numpy.nanvar")(a[, axis, dtype, out, ddof, ...])

### More

[`cov`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cov.html#jax.numpy.cov "jax.numpy.cov")(m[, y, rowvar, bias, ddof, fweights, ...])
: correlation matrix.

[`corrcoef`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.corrcoef.html#jax.numpy.corrcoef "jax.numpy.corrcoef")(x[, y, rowvar, dtype])
: Pearson correlation coefficients.

[`bincount`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bincount.html#jax.numpy.bincount "jax.numpy.bincount")(x[, weights, minlength, length])

[`digitize`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.digitize.html#jax.numpy.digitize "jax.numpy.digitize")(x, bins[, right, method])

[`histogram`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogram.html#jax.numpy.histogram "jax.numpy.histogram")(a[, bins, range, weights, density])
[`histogram2d`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogram2d.html#jax.numpy.histogram2d "jax.numpy.histogram2d")(x, y[, bins, range, weights, ...])
[`histogramdd`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogramdd.html#jax.numpy.histogramdd "jax.numpy.histogramdd")(sample[, bins, range, weights, ...])
: 1D, 2D and N-dim histogram.

[`histogram_bin_edges`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogram_bin_edges.html#jax.numpy.histogram_bin_edges "jax.numpy.histogram_bin_edges")(a[, bins, range, weights])
: edges of bins of histogram.

## Window

[`bartlett`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.bartlett.html#jax.numpy.bartlett "jax.numpy.bartlett")(M)

[`blackman`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.blackman.html#jax.numpy.blackman "jax.numpy.blackman")(M)

[`hamming`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hamming.html#jax.numpy.hamming "jax.numpy.hamming")(M)

[`hanning`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hanning.html#jax.numpy.hanning "jax.numpy.hanning")(M)

[`kaiser`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.kaiser.html#jax.numpy.kaiser "jax.numpy.kaiser")(M, beta)
