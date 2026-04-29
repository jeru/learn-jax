# Copyright 2026 Cheng Sheng
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Sequence

import numpy
import grain.transforms


def batch_with_masked_power2_padding(
        batch_size: int | None, *, jax_field_names: Sequence[str],
        unify_dimension_sizes: bool = False,
) -> grain.transforms.Batch:
    """Batch and pad data.

    Fields in `jax_field_names` will be aligned to the same shape (each
    dimension has a power-2 size) and the final array will have the batching
    dimension at axis=0. Batching dimeninsion will also be padded.

    Fields not in `jax_field_names` will be simply grouped into sequences
    without any padding.

    For any padded fields, a corresponding field with suffix "/mask" will also
    be added to tell which elements are original.
    """
    batch_fn = BatchWithMaskedPower2PaddingFn(
            batch_size=batch_size,
            jax_field_names=jax_field_names,
            unify_dimension_sizes=unify_dimension_sizes)
    return grain.transforms.Batch(batch_size, batch_fn=batch_fn)



class BatchWithMaskedPower2PaddingFn:
    def __init__(self, *, batch_size: int | None,
                 jax_field_names: Sequence[str],
                 unify_dimension_sizes: bool):
        self.batch_size = batch_size
        self.jax_field_names = frozenset(jax_field_names)
        self.unify_dimension_sizes = unify_dimension_sizes

    def __call__(self, dicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
        field_list = {}
        for adict in dicts:
            for k, v in adict.items():
                field_list.setdefault(k, []).append(v)
        results = {}
        for k, v in field_list.items():
            if len(v) != len(dicts):
                raise ValueError(f"Field missing from some inputs: {k}")
            if k in self.jax_field_names:
                results[k], results[k + '/mask'] = self._pad_jax_field(k, v)
            else:
                results[k] = tuple(v)
        return results

    def _pad_jax_field(self, key: str, arrs: Sequence[numpy.array]
                       ) -> tuple[numpy.array, numpy.array]:
        dim0 = len(arrs[0].shape)
        for arr in arrs:
            if len(arr.shape) != dim0:
                raise ValueError(f"Inconsistent dimensions for field: {key}")

        shape = [max(arr.shape[i] for arr in arrs) for i in range(dim0)]
        if self.unify_dimension_sizes:
            shape = [_upgrade_to_power2(max(shape))] * dim0
        else:
            shape = [_upgrade_to_power2(x) for x in shape]

        results = []
        masks = []
        for arr in arrs:
            pad_width = [(0, x - y) for x, y in zip(shape, arr.shape)]
            results.append(numpy.pad(arr, pad_width, mode="edge"))
            mask = numpy.zeros(shape, dtype=bool)
            mask[tuple(map(slice, arr.shape))] = True
            masks.append(mask)

        results = numpy.stack(results)
        masks = numpy.stack(masks)
        if len(results) < self.batch_size:
            pad_width = [(0, self.batch_size - len(results))] + [(0, 0)] * dim0
            results = numpy.pad(results, pad_width, mode="edge")
            masks = numpy.pad(masks, pad_width, mode="constant",
                              constant_values=False)
        return results, masks


def _upgrade_to_power2(n: int) -> int:
    p2 = 1
    while p2 < n: p2 *= 2
    return p2
