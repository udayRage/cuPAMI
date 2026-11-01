"""CUDA periodic-support counting with PAMI-compatible thresholds.

Reference API: PAMI.partialPeriodicPattern.basic.PPP_ECLAT.
Only gaps between distinct occurrence timestamps count; boundary gaps do not.

GPU scan reference: PAMI gPPMiner, supportAndPeriod.
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/partialPeriodicPattern/cuda/gPPMiner.py
Local changes exclude the start gap, use explicit int64 timestamps, and
accept support >= minPS as in PPP_ECLAT. See docs/algorithm-references.md.
"""
import math
import numbers
from functools import lru_cache

import numpy as np

from ..periodicFrequentPattern._temporal import TemporalInput
from ..frequentPattern.cuApriori import cuApriori

_SOURCE = r'''
extern "C" __global__ void partial_counts(
    const unsigned int* bits, unsigned long long words,
    const unsigned int* candidates, unsigned int width, unsigned int count,
    const long long* timestamps, unsigned long long n, double period,
    unsigned long long* supports) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= count) return;
    unsigned long long support = 0;
    long long previous = 0;
    bool seen = false;
    for (unsigned long long w = 0; w < words; ++w) {
        unsigned int value = 0xffffffffu;
        for (unsigned int j = 0; j < width; ++j)
            value &= bits[(unsigned long long)candidates[(unsigned long long)c * width + j] * words + w];
        while (value) {
            unsigned int bit = __ffs(value) - 1;
            unsigned long long row = w * 32 + bit;
            if (row < n) {
                long long ts = timestamps[row];
                if (!seen || ts != previous) {
                    if (seen && (double)(ts - previous) <= period) ++support;
                    previous = ts;
                    seen = true;
                }
            }
            value &= value - 1;
        }
    }
    supports[c] = support;
}
'''


@lru_cache(maxsize=None)
def _kernel(device):
    import cupy as cp
    return cp.RawKernel(_SOURCE, 'partial_counts', options=('--std=c++11',))


class PartialInput(TemporalInput):
    """Temporal input with periodic support rather than maximum-gap support.

    Integer thresholds are absolute. Floats (including decimal strings) are
    multiplied by the transaction count, as in PAMI. Duplicate timestamps
    are merged by taking the union of their baskets before mining.
    """
    def __init__(self, iFile, minPS, period, sep=None):
        self.minPS = minPS
        self.period = period
        super().__init__(iFile, 1, 0, sep)
        self._minPS = self._threshold(minPS, 'minPS')
        self._period = self._threshold(period, 'period')

    def _threshold(self, value, name):
        if isinstance(value, str):
            try:
                value = float(value) if any(c in value.lower() for c in '.e') else int(value)
            except ValueError:
                raise ValueError(f'{name} must be a nonnegative finite number') from None
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be a nonnegative finite number')
        result = value * self._transactionCount if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral) else value
        if not math.isfinite(result):
            raise ValueError(f'{name} is too large')
        return result

    def _prepare(self, cp):
        # PAMI uses timestamp sets: items in separate rows at the same time
        # must be merged before intersections, not just deduplicated afterward.
        transactions, timestamps = self._Database, self._timestamps
        merged = {}
        for ts, basket in zip(timestamps, transactions):
            merged.setdefault(ts, set()).update(basket)
        self._Database = [sorted(items) for items in merged.values()]
        self._timestamps = list(merged)
        count = self._transactionCount
        self._transactionCount = len(merged)
        try:
            return super()._prepare(cp)
        finally:
            self._Database, self._timestamps = transactions, timestamps
            self._transactionCount = count

    def _evaluate(self, cp, candidates):
        if not candidates:
            return []
        indexes = cp.asarray([[self._itemIndexes[item] for item in p] for p in candidates], dtype=cp.uint32)
        supports = cp.empty(len(candidates), dtype=cp.uint64)
        _kernel(cp.cuda.runtime.getDevice())(((len(candidates) + 127) // 128,), (128,), (
            self._temporalBits, np.uint64(self._temporalBits.shape[1]), indexes,
            np.uint32(indexes.shape[1]), np.uint32(len(candidates)), self._deviceTimestamps,
            np.uint64(len(set(self._timestamps))), np.float64(self._period), supports))
        return [(int(s), 0) for s in cp.asnumpy(supports)]

    def _accept(self, pattern, support, period):
        if support >= self._minPS:
            self._finalPatterns[pattern] = int(support)
            return True
        return False

    getPatterns = cuApriori.getPatterns
    save = cuApriori.save

    def getPatternsAsDataFrame(self):
        import pandas as pd
        return pd.DataFrame(self.getPatterns().items(), columns=['Patterns', 'periodicSupport'])
