"""GPU FP-tree arrays and weighted conditional pattern bases.

Local CUDA/CuPy implementation of the FP-tree/conditional-base operations in
Han, Pei and Yin, SIGMOD 2000. No source code was copied from that paper's
authors. GPU level grouping replaces sequential pointer insertion.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


_PROJECTION_SOURCE = r'''
extern "C" __global__ void ancestor_supports(
    const long long* nodes, unsigned long long n,
    const long long* parents, const long long* items,
    const unsigned long long* counts, unsigned long long* supports) {
    unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    long long node = nodes[row];
    unsigned long long weight = counts[node];
    for (long long parent = parents[node]; parent > 0; parent = parents[parent])
        atomicAdd(&supports[items[parent]], weight);
}

extern "C" __global__ void projected_lengths(
    const long long* nodes, unsigned long long n,
    const long long* parents, const long long* items,
    const bool* frequent, unsigned long long* lengths) {
    unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    unsigned long long length = 0;
    for (long long parent = parents[nodes[row]]; parent > 0; parent = parents[parent])
        if (frequent[items[parent]]) ++length;
    lengths[row] = length;
}

extern "C" __global__ void emit_projected_paths(
    const long long* nodes, unsigned long long n,
    const long long* parents, const long long* items,
    const bool* frequent, const unsigned long long* offsets,
    unsigned long long* rows, unsigned int* columns) {
    unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    unsigned long long offset = offsets[row];
    for (long long parent = parents[nodes[row]]; parent > 0; parent = parents[parent]) {
        long long item = items[parent];
        if (frequent[item]) {
            rows[offset] = row;
            columns[offset++] = (unsigned int)item;
        }
    }
}
'''


@lru_cache(maxsize=None)
def _projection_kernels(device):
    import cupy as cp

    module = cp.RawModule(code=_PROJECTION_SOURCE, options=("--std=c++11",))
    return tuple(module.get_function(name) for name in (
        "ancestor_supports", "projected_lengths", "emit_projected_paths"
    ))


@dataclass
class FPTree:
    parents: object
    items: object
    counts: object
    header: list
    single_path: bool
    depth: int


def unique_pairs(cp, rows, columns):
    """Deduplicate transaction/item pairs before counting their support."""
    if not rows.size:
        return rows, columns
    order = cp.lexsort(cp.stack((columns.astype(cp.uint64), rows.astype(cp.uint64))))
    rows, columns = rows[order], columns[order]
    keep = cp.concatenate((cp.ones(1, dtype=cp.bool_),
                           (rows[1:] != rows[:-1]) | (columns[1:] != columns[:-1])))
    return rows[keep], columns[keep]


def build_tree(cp, rows, columns, weights, item_count, min_support):
    """Build a weighted FP-tree on the GPU, merging equal prefixes by level.

    rows/columns encode weighted baskets. All operations on the baskets,
    weighted support counts, parent IDs, and nodes stay in device memory.
    The small header table is copied to the CPU for search scheduling.
    """
    if not rows.size:
        return None
    rows = rows.astype(cp.int64, copy=False)
    supports = cp.zeros(item_count, dtype=cp.uint64)
    cp.add.at(supports, columns, weights[rows])
    frequent = supports >= min_support
    keep = frequent[columns]
    rows, columns = rows[keep], columns[keep]
    if not rows.size:
        return None

    # Descending weighted support, with deterministic item-ID tie breaking.
    item_order = cp.lexsort(cp.stack((cp.arange(item_count, dtype=cp.uint64),
                                     cp.max(supports) - supports)))
    rank = cp.empty(item_count, dtype=cp.int64)
    rank[item_order] = cp.arange(item_count)
    order = cp.lexsort(cp.stack((rank[columns], rows)))
    rows, columns = rows[order], columns[order]
    lengths = cp.bincount(rows.astype(cp.int64), minlength=len(weights))
    offsets = cp.concatenate((cp.zeros(1, dtype=cp.int64), cp.cumsum(lengths)))
    parent_for_row = cp.zeros(len(weights), dtype=cp.int64)
    depth = int(lengths.max())

    parent_parts = [cp.full(1, -1, dtype=cp.int64)]
    item_parts = [cp.full(1, -1, dtype=cp.int64)]  # root; item ID 0 is valid
    count_parts = [cp.zeros(1, dtype=cp.uint64)]
    node_count = 1
    single_path = True
    for level in range(depth):
        active = cp.flatnonzero(lengths > level)
        level_items = columns[offsets[active] + level].astype(cp.int64)
        level_parents = parent_for_row[active]
        level_order = cp.lexsort(cp.stack((level_items, level_parents)))
        parents = level_parents[level_order]
        items = level_items[level_order]
        first = cp.concatenate((cp.ones(1, dtype=cp.bool_),
                                (parents[1:] != parents[:-1]) | (items[1:] != items[:-1])))
        groups = cp.cumsum(first) - 1
        group_count = int(groups[-1]) + 1
        counts = cp.zeros(group_count, dtype=cp.uint64)
        cp.add.at(counts, groups, weights[active[level_order]])
        parent_parts.append(parents[first])
        item_parts.append(items[first])
        count_parts.append(counts)
        row_nodes = cp.empty(len(active), dtype=cp.int64)
        row_nodes[level_order] = groups + node_count
        parent_for_row[active] = row_nodes
        node_count += group_count
        single_path = single_path and group_count == 1

    header_items = item_order[frequent[item_order]][::-1]
    header = list(zip(cp.asnumpy(header_items).tolist(),
                      cp.asnumpy(supports[header_items]).tolist()))
    return FPTree(cp.concatenate(parent_parts), cp.concatenate(item_parts),
                  cp.concatenate(count_parts), header, single_path, depth)


def conditional_tree(cp, tree, item, item_count, min_support):
    """Project the weighted ancestor paths of each node for the selected item."""
    nodes = cp.flatnonzero(tree.items == item)
    if not nodes.size:
        return None
    support_kernel, length_kernel, emit_kernel = _projection_kernels(cp.cuda.runtime.getDevice())
    grid, block = ((len(nodes) + 255) // 256,), (256,)
    supports = cp.zeros(item_count, dtype=cp.uint64)
    common = (nodes, np.uint64(len(nodes)), tree.parents, tree.items)
    support_kernel(grid, block, common + (tree.counts, supports))
    frequent = supports >= min_support
    header = cp.flatnonzero(frequent)
    if not header.size:
        return None
    if header.size == 1:
        selected = int(header[0])
        support = int(supports[selected])
        return FPTree(cp.asarray([-1, 0], dtype=cp.int64),
                      cp.asarray([-1, selected], dtype=cp.int64),
                      cp.asarray([0, support], dtype=cp.uint64),
                      [(selected, support)], True, 1)

    lengths = cp.empty(len(nodes), dtype=cp.uint64)
    length_kernel(grid, block, common + (frequent, lengths))
    offsets = cp.concatenate((cp.zeros(1, dtype=cp.uint64), cp.cumsum(lengths)))
    total = int(offsets[-1])
    rows = cp.empty(total, dtype=cp.uint64)
    columns = cp.empty(total, dtype=cp.uint32)
    emit_kernel(grid, block, common + (frequent, offsets, rows, columns))
    weights = tree.counts[nodes]
    return build_tree(cp, rows, columns, weights, item_count, min_support)
