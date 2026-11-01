"""Local CUDA Apriori adaptation with periodicity pruning; see README.md#references.
GPU counting reference: PAMI gPFMinerBit, supportAndPeriod kernel.
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/periodicFrequentPattern/cuda/gPFMinerBit.py
Follows batched bitset intersections with per-candidate support/gap scans.
Local variant uses breadth-first Apriori pruning and row-index bitsets plus
int64 timestamps, rather than dense bits indexed by elapsed time.
Related GPU temporal work: Tarun et al., IEEE Big Data 2022,
doi:10.1109/BigData55660.2022.10020216 (gPF-Miner's flattened arrays).
This file does not implement that paper's flattened-array algorithm.
See docs/algorithm-references.md.
"""
import time
from ._temporal import TemporalInput


class cuPFApriori(TemporalInput):
    """Level-wise periodic-frequent mining; GPU support and maximum gaps."""
    _BATCH_SIZE = 4096

    def mine(self):
        self._startTime = time.perf_counter()
        cp = self._getCudaModule()
        labels, _, _ = self._prepare(cp)
        candidates = [(label,) for label in labels]
        while candidates:
            level = []
            for start in range(0, len(candidates), self._BATCH_SIZE):
                batch = candidates[start:start + self._BATCH_SIZE]
                for pattern, (support, period) in zip(batch, self._evaluate(cp, batch)):
                    if self._accept(pattern, support, period):
                        level.append(pattern)
            candidates = self._generateCandidates(level)
        cp.cuda.get_current_stream().synchronize()
        self._endTime = time.perf_counter()
