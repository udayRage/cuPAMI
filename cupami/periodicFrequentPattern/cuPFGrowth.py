"""Local FP-tree search augmented with GPU temporal occurrence bitsets.

Based on Han et al.'s FP-growth search, with GPU periodicity checks.
See README.md#references for sources.
"""
import time
from ._temporal import TemporalInput
from ..frequentPattern.cuFPGrowth import cuFPGrowth
from ..frequentPattern import _fptree


class cuPFGrowth(TemporalInput, cuFPGrowth):
    """Conditional FP-tree mining with temporal checks before expansion."""
    def mine(self):
        self._startTime = time.perf_counter()
        cp = self._getCudaModule()
        with cp.cuda.using_allocator(cp.get_default_memory_pool().malloc):
            labels, rows, columns = self._prepare(cp)
            rows, columns = _fptree.unique_pairs(cp, rows, columns)
            weights = cp.ones(self._transactionCount, dtype=cp.uint64)
            root = _fptree.build_tree(cp, rows, columns, weights, len(labels), self._minSup)
            if root is not None:
                self._search(cp, root, labels)
            cp.cuda.get_current_stream().synchronize()
            self._endTime = time.perf_counter()

    def _storePattern(self, item_ids, support, labels):
        pattern = tuple(sorted(labels[index] for index in item_ids))
        measured, period = self._evaluate(self._getCudaModule(), [pattern])[0]
        return self._accept(pattern, measured, period)
