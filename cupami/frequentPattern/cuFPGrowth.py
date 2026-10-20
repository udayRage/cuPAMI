"""CUDA FP-growth using GPU FP-trees and weighted conditional pattern bases.

Algorithm: Jiawei Han, Jian Pei and Yiwen Yin, "Mining Frequent Patterns
without Candidate Generation," SIGMOD 2000, pp. 1-12.
https://doi.org/10.1145/342009.335372
Algorithms 1 and 2 define tree construction and FP-growth; Lemma 3.2 gives
the single-path optimization. This is a local GPU adaptation, not the
authors' source code. See docs/references.md.
"""

from itertools import combinations
from pathlib import Path
import sys
import time

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "cupami.frequentPattern"

from .cuApriori import cuApriori
from . import _fptree


class cuFPGrowth(cuApriori):
    """Mine frequent itemsets through FP-trees, without Apriori candidates.

    Shares readers and output APIs with cuApriori. Weighted supports, sorted
    baskets, FP-tree arrays, and conditional projections are computed using
    CuPy on CUDA. CPU code schedules the search and formats output patterns.
    A positive minimum support count is required for nonempty input.
    """

    def mine(self):
        cp = self._getCudaModule()
        # cuDF can change CuPy's process-wide allocator during import. Keep
        # tree allocation consistent across readers without changing that
        # global choice for the caller or for cuDF itself.
        with cp.cuda.using_allocator(cp.get_default_memory_pool().malloc):
            self._mineWithPool(cp)

    def _mineWithPool(self, cp):
        self._startTime = time.perf_counter()
        self._finalPatterns = {}
        labels, rows, columns = self._getDevicePairs(cp)
        if labels and self._minSup < 1:
            raise ValueError("FP-growth requires a positive minimum support count")
        rows, columns = _fptree.unique_pairs(cp, rows, columns)
        weights = cp.ones(self._transactionCount, dtype=cp.uint64)
        root = _fptree.build_tree(cp, rows, columns, weights, len(labels), self._minSup)
        if root is not None:
            self._search(cp, root, labels)
        cp.cuda.get_current_stream().synchronize()
        self._endTime = time.perf_counter()

    def _storePattern(self, item_ids, support, labels):
        pattern = tuple(sorted(labels[index] for index in item_ids))
        self._finalPatterns[pattern] = int(support)

    def _search(self, cp, root, labels):
        # Explicit frames preserve depth-first scheduling without relying on
        # Python's recursion limit. Each frame owns its conditional FP-tree.
        stack = [[root, (), 0]]
        while stack:
            tree, suffix, position = stack[-1]
            if tree.single_path:
                path_items = cp.asnumpy(tree.items[1:]).tolist()
                path_counts = cp.asnumpy(tree.counts[1:]).tolist()
                for length in range(1, len(path_items) + 1):
                    for selected in combinations(range(len(path_items)), length):
                        ids = suffix + tuple(path_items[index] for index in selected)
                        support = min(path_counts[index] for index in selected)
                        self._storePattern(ids, support, labels)
                stack.pop()
                continue
            if position >= len(tree.header):
                stack.pop()
                continue
            stack[-1][2] += 1
            item, support = tree.header[position]
            next_suffix = suffix + (item,)
            if self._storePattern(next_suffix, support, labels) is False:
                continue
            conditional = _fptree.conditional_tree(cp, tree, item, len(labels), self._minSup)
            if conditional is not None:
                stack.append([conditional, next_suffix, 0])


FPGrowth = cuFPGrowth


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit("Usage: python cuFPGrowth.py <inputFile> <outputFile> <minSup> [sep]")
    try:
        minSup = int(sys.argv[3])
    except ValueError:
        minSup = float(sys.argv[3])
    sep = sys.argv[4] if len(sys.argv) == 5 else None
    miner = cuFPGrowth(sys.argv[1], minSup, sep)
    miner.mine()
    miner.save(sys.argv[2])
    miner.printResults()
