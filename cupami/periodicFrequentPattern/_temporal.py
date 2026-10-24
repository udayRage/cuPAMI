"""Shared temporal input and local CUDA periodicity implementation.

Periodicity follows Kiran, Kitsuregawa and Reddy (JSS, 2016), Section 1:
maximum inter-arrival time including database start/end gaps. This kernel
is local code; it does not implement that paper's greedy pruning algorithm.
"""
import csv
import io
import math
import numbers
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

from ..frequentPattern.cuApriori import cuApriori
from ..frequentPattern import _bitsets

_SOURCE = r'''
extern "C" __global__ void periodic_counts(
    const unsigned int* bits, unsigned long long words,
    const unsigned int* candidates, unsigned int width, unsigned int candidate_count,
    const long long* timestamps, unsigned long long n, long long end,
    unsigned long long* supports, long long* periods) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= candidate_count) return;
    unsigned long long support = 0;
    long long previous = 0, period = 0;
    for (unsigned long long w = 0; w < words; ++w) {
        unsigned int value = 0xffffffffu;
        for (unsigned int j = 0; j < width; ++j)
            value &= bits[(unsigned long long)candidates[(unsigned long long)c * width + j] * words + w];
        while (value) {
            unsigned int bit = __ffs(value) - 1;
            unsigned long long row = w * 32 + bit;
            if (row < n) {
                long long ts = timestamps[row];
                period = max(period, ts - previous);
                previous = ts;
                ++support;
            }
            value &= value - 1;
        }
    }
    supports[c] = support;
    periods[c] = max(period, end - previous);
}
'''


@lru_cache(maxsize=None)
def _kernel(device):
    import cupy as cp
    return cp.RawKernel(_SOURCE, 'periodic_counts', options=('--std=c++11',))


class TemporalInput(cuApriori):
    """Timestamp-first CSV, temporal BIN/Parquet, or iterable input.

    Equal timestamps remain separate transactions; duplicate items within a
    basket count once. Empty baskets retain their timestamp and count in the
    support denominator. Rows are sorted by timestamp. Timeline starts at 0
    and ends at the largest input timestamp (or explicit endTime).
    """
    def __init__(self, iFile, minSup, maxPer, sep=None, *, endTime=None):
        loading_start = time.perf_counter()
        self._timestamps = []
        self.endTime = endTime
        self.maxPer = maxPer
        super().__init__(iFile, minSup, sep, use_gds=False, use_gpu_io=False)
        if isinstance(minSup, bool) or not isinstance(minSup, numbers.Real) or not math.isfinite(minSup) or minSup <= 0:
            raise ValueError('minSup must be a positive finite number')
        if isinstance(maxPer, bool) or not isinstance(maxPer, numbers.Real) or not math.isfinite(maxPer) or maxPer < 0:
            raise ValueError('maxPer must be a nonnegative finite number')
        # Fractions refer to elapsed timestamp units, not number of records.
        self._maxPer = maxPer * self._endTimestamp if isinstance(maxPer, float) and 0 < maxPer <= 1 else maxPer
        self._periods = {}
        self._loadingTime = time.perf_counter() - loading_start

    @staticmethod
    def _timestamp(value):
        if isinstance(value, (bool, float)):
            raise ValueError('timestamps must be nonnegative int64 integers')
        text = str(value).strip()
        if not text or any(c not in '0123456789' for c in text):
            raise ValueError('timestamps must be nonnegative int64 integers')
        timestamp = int(text)
        if timestamp > np.iinfo(np.int64).max:
            raise ValueError('timestamp exceeds int64')
        return timestamp

    def _readTransactions(self, source):
        if isinstance(source, (str, Path)):
            suffix = Path(str(source)).suffix.lower()
            is_url = urlparse(str(source)).scheme in ('http', 'https')
            if suffix in ('.bin', '.parquet'):
                if is_url:
                    raise ValueError('Temporal BIN and Parquet input requires a local file')
                from ..extras import _temporal_io
                rows = (_temporal_io.read_binary(source) if suffix == '.bin'
                        else _temporal_io.read_parquet(source))
                return self._cleanTemporalRows(rows)
            if is_url:
                with urlopen(str(source), timeout=30) as response:
                    text = response.read().decode('utf-8-sig')
            else:
                text = Path(source).read_text(encoding='utf-8-sig')
            separator = self.sep
            if separator is None:
                first = next((line for line in text.splitlines() if line.strip()), '')
                separator = '\t' if '\t' in first else ',' if ',' in first else None
            rows = csv.reader(io.StringIO(text), delimiter=separator) if separator else (line.split() for line in text.splitlines())
        else:
            rows = source
        return self._cleanTemporalRows(rows)

    def _cleanTemporalRows(self, rows):
        records = []
        for row in rows:
            row = list(row)
            if not row or not any(str(value).strip() for value in row):
                continue
            records.append((self._timestamp(row[0]), self._cleanTransaction(row[1:])))
        records.sort(key=lambda record: record[0])
        self._timestamps = [record[0] for record in records]
        last = self._timestamps[-1] if records else 0
        self._endTimestamp = last if self.endTime is None else self._timestamp(self.endTime)
        if self._endTimestamp < last:
            raise ValueError('endTime must be at least the last timestamp')
        self._inputBackend = {'backend': 'cpu', 'reason': 'Temporal reader; mining uses CUDA'}
        return [record[1] for record in records]

    def _prepare(self, cp):
        labels, rows, columns = self._getDevicePairs(cp)
        self._labels = labels
        self._itemIndexes = {label: i for i, label in enumerate(labels)}
        self._temporalBits = _bitsets.build_bitsets(cp, rows, columns, self._transactionCount, len(labels))
        self._deviceTimestamps = cp.asarray(self._timestamps, dtype=cp.int64)
        self._finalPatterns = {}
        self._periods = {}
        return labels, rows, columns

    def _evaluate(self, cp, candidates):
        if not candidates:
            return []
        indexes = cp.asarray([[self._itemIndexes[item] for item in pattern] for pattern in candidates], dtype=cp.uint32)
        supports = cp.empty(len(candidates), dtype=cp.uint64)
        periods = cp.empty(len(candidates), dtype=cp.int64)
        _kernel(cp.cuda.runtime.getDevice())(((len(candidates) + 127) // 128,), (128,), (
            self._temporalBits, np.uint64(self._temporalBits.shape[1]), indexes,
            np.uint32(indexes.shape[1]), np.uint32(len(candidates)), self._deviceTimestamps,
            np.uint64(self._transactionCount), np.int64(self._endTimestamp), supports, periods))
        return list(zip(cp.asnumpy(supports).tolist(), cp.asnumpy(periods).tolist()))

    def _accept(self, pattern, support, period):
        if support >= self._minSup and support > 0 and period <= self._maxPer:
            self._finalPatterns[pattern] = int(support)
            self._periods[pattern] = int(period)
            return True
        return False

    def getPatterns(self):
        return {'\t'.join(pattern): (support, self._periods[pattern]) for pattern, support in self._finalPatterns.items()}

    def getPatternsAsDataFrame(self):
        import pandas as pd
        return pd.DataFrame([('\t'.join(p), s, self._periods[p]) for p, s in self._finalPatterns.items()],
                            columns=['Patterns', 'Support', 'Periodicity'])

    def save(self, outputFile):
        with Path(outputFile).open('w', encoding='utf-8') as output:
            for pattern, support in sorted(self._finalPatterns.items()):
                output.write('{} : {} : {}\n'.format('\t'.join(pattern), support, self._periods[pattern]))
