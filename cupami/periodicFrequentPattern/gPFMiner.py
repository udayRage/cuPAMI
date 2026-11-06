# gPFMiner discovers periodic-frequent patterns with GPU timestamp bitsets.
#
# **Importing this algorithm into a python program**
#
#             import cupami as alg
#
#             obj = alg.gPFMiner('temporal.csv', minSup=10, maxPer=30)
#             obj.mine()
#             patterns = obj.getPatterns()
#             print('Total number of patterns:', len(patterns))
#             obj.save('patterns.txt')
#             Df = obj.getPatternsAsDataFrame()
#             print('Memory USS:', obj.getMemoryUSS())
#             print('Memory RSS:', obj.getMemoryRSS())
#             print('Mining time in seconds:', obj.getRuntime())
#
"""gPF-Miner: a CuPy adaptation of PAMI's public timestamp-bitset variant.

Paper: Sreepada Tarun, Rage Uday Kiran, Yutaka Watanobe, Kazuo Goda,
"A Novel GPU-Accelerated Algorithm to Discover Periodic-Frequent Patterns
in Temporal Databases", IEEE Big Data 2022, pp. 121–126.
https://doi.org/10.1109/BigData55660.2022.10020216

Source references (Tarun Sreepada, supervised by Rage Uday Kiran):
https://github.com/UdayLab/PAMI/blob/2d4e7bcc2879bec8c12473e8fb82b0ac22aa3918/PAMI/periodicFrequentPattern/cuda/cuGPFMiner.py
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/periodicFrequentPattern/cuda/gPFMinerBit.py

The paper's flattened array consists of packed item bitsets. This port retains
that layout and complete-key GPU intersections, support/period scans and prefix
expansion from the public code. It uses
CuPy, iterative depth-first scheduling, set-bit scans, bounded candidate batches,
exact start/end gaps and a timeline allocation limit. Original source credits
and license apply. See docs/algorithm-references.md.
"""

__copyright__ = """
Copyright (C) 2021 Rage Uday Kiran

This program is free software: you can redistribute it and/or modify it under
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
"""

from functools import lru_cache
from numbers import Integral
from pathlib import Path
import sys
import time

import numpy as np

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'cupami.periodicFrequentPattern'

from ._temporal import TemporalInput


_SOURCE = r'''
// Adapted from Tarun Sreepada's PAMI supportAndPeriod kernels (GPL-3.0-or-later).
// Timestamp zero is represented explicitly; padding bits are never counted.
extern "C" __global__ void supportAndPeriod(
    const unsigned long long* bits, unsigned long long words,
    const unsigned int* candidates, const unsigned long long* offsets,
    unsigned int count, long long endTime, double maxPeriod,
    unsigned long long* supports, long long* periods) {
    unsigned int job = blockIdx.x * blockDim.x + threadIdx.x;
    if (job >= count) return;
    unsigned long long support = 0;
    long long previous = 0, maximum = 0;
    for (unsigned long long word = 0; word < words; ++word) {
        unsigned long long intersection = ~0ULL;
        for (unsigned long long k = offsets[job]; k < offsets[job + 1]; ++k)
            intersection &= bits[(unsigned long long)candidates[k] * words + word];
        while (intersection) {
            unsigned int bit = __ffsll((long long)intersection) - 1;
            long long timestamp = (long long)(word * 64 + bit);
            if (timestamp > endTime) break;
            long long gap = timestamp - previous;
            maximum = gap > maximum ? gap : maximum;
            // Periodicity cannot decrease as this scan progresses.
            if ((double)maximum > maxPeriod) {
                supports[job] = support;
                periods[job] = maximum;
                return;
            }
            previous = timestamp;
            ++support;
            intersection &= intersection - 1;
        }
        // Even an occurrence at the next word's first timestamp would be late.
        long long boundary = (long long)((word + 1) * 64);
        if (boundary > endTime) boundary = endTime;
        if ((double)(boundary - previous) > maxPeriod) {
            supports[job] = support;
            periods[job] = boundary - previous;
            return;
        }
    }
    long long tail = endTime - previous;
    supports[job] = support;
    periods[job] = tail > maximum ? tail : maximum;
}
'''


@lru_cache(maxsize=None)
def _kernel(device):
    import cupy as cp
    with cp.cuda.Device(device):
        return cp.RawKernel(_SOURCE, 'supportAndPeriod')


class gPFMiner(TemporalInput):
    """
    **About this algorithm**

    :**Description**: gPF-Miner's public bitset variant uses GPU intersections
        over a dense timestamp timeline. A CPU depth-first prefix traversal
        batches candidate patterns; one GPU thread evaluates each candidate's
        support and maximum gap, including the start and end gaps.

    :**Reference**: Sreepada Tarun, Rage Uday Kiran, Yutaka Watanobe, Kazuo Goda:
        A Novel GPU-Accelerated Algorithm to Discover Periodic-Frequent Patterns
        in Temporal Databases. IEEE Big Data 2022, pp. 121–126.
        https://doi.org/10.1109/BigData55660.2022.10020216

    :**Parameters**: - **iFile** (*str, pathlib.Path or iterable*) -- Timestamp-first
                       rows, CSV, temporal BIN/Parquet, or a CSV HTTP(S) URL.
                     - **minSup** (*int or float*) -- Integer absolute support;
                       floats in (0, 1] are fractions of input transaction count.
                     - **maxPer** (*int or float*) -- Maximum gap in timestamp
                       units; floats in (0, 1] are fractions of the timeline.
                     - **sep** (*str or None*) -- Delimiter, detected when None.
                     - **endTime** (*int or None*) -- Observation end, defaulting
                       to the largest input timestamp, including empty rows.
                     - **maxTimelineBytes** (*int*) -- Maximum singleton timeline
                       bitset size in bytes; default 512 MiB. Checked before allocation.

    :**Attributes**: - **_timelineBits** (*cupy.ndarray or None*) -- Timestamp
                       singleton bitsets used for complete-key intersections.
                     - **_finalPatterns** (*dict*) -- Patterns and support counts.
                     - **_periods** (*dict*) -- Maximum pattern gaps.
                     - **_startTime**, **_endTime** (*float*) -- Mining timings.

    **Execution methods**

    **Terminal command**

    .. code-block:: console

        python3 -m cupami.periodicFrequentPattern.gPFMiner temporal.csv patterns.txt 10 30

    **Calling from a python program**

    .. code-block:: python

        from cupami import gPFMiner

        obj = gPFMiner('temporal.csv', minSup=10, maxPer=30)
        obj.mine()
        print(obj.getPatterns())
        obj.save('patterns.txt')
        Df = obj.getPatternsAsDataFrame()
        print(obj.getMemoryUSS(), obj.getMemoryRSS())
        print(obj.getRuntime())

    .. note:: Equal timestamps are unioned: support counts distinct occurrence
              timestamps, matching the timestamp-bitset representation. Relative
              support still uses the input row count. Large sparse timelines may
              exceed the memory limit; cuPFEclat uses row-based bitsets instead.
              Local differences from the pseudocode include lexical item order,
              set-bit scans and corrected boundary handling. The paper ranks items
              by descending support. Mining needs CuPy and a CUDA GPU.

    **Credits**

    Original PAMI programs by Tarun Sreepada under the supervision of Professor
    Rage Uday Kiran. cuPAMI integration and corrections by Mithun Thangaraj under
    the supervision of Professor Rage Uday Kiran. Source links and adaptation
    differences are recorded in the module documentation.
    """

    _BATCH_SIZE = 4096

    def __init__(self, iFile, minSup, maxPer, sep=None, *, endTime=None,
                 maxTimelineBytes=512 * 1024 * 1024):
        if isinstance(maxTimelineBytes, bool) or not isinstance(maxTimelineBytes, Integral) or maxTimelineBytes < 1:
            raise ValueError('maxTimelineBytes must be a positive integer')
        self.maxTimelineBytes = int(maxTimelineBytes)
        self._timelineBits = None
        super().__init__(iFile, minSup, maxPer, sep, endTime=endTime)

    def _prepareTimeline(self, cp):
        occurrences = {}
        for timestamp, basket in zip(self._timestamps, self._Database):
            for item in basket:
                occurrences.setdefault(item, set()).add(timestamp)
        labels, times = [], []
        for label in sorted(occurrences):
            points = sorted(occurrences[label])
            gaps = (b - a for a, b in zip([0, *points], [*points, self._endTimestamp]))
            period = max(gaps)
            if self._accept((label,), len(points), period):
                labels.append(label)
                times.append(points)
        words = self._endTimestamp // 64 + 1
        required = len(labels) * words * np.dtype(np.uint64).itemsize
        if required > self.maxTimelineBytes:
            raise ValueError(f'Timestamp bitsets need {required} bytes, above maxTimelineBytes={self.maxTimelineBytes}; use cuPFEclat for sparse timelines')
        if not labels:
            return []
        bits = np.zeros((len(labels), words), dtype=np.uint64)
        for index, points in enumerate(times):
            stamps = np.asarray(points, dtype=np.int64)
            masks = np.left_shift(np.uint64(1), (stamps % 64).astype(np.uint64))
            np.bitwise_or.at(bits[index], stamps // 64, masks)
        self._timelineBits = cp.asarray(bits)
        return labels

    def _evaluateTimeline(self, cp, candidates):
        if not candidates:
            return []
        flat = cp.asarray([index for pattern in candidates for index in pattern], dtype=cp.uint32)
        offsets = cp.asarray(np.cumsum([0] + [len(pattern) for pattern in candidates], dtype=np.uint64))
        supports = cp.empty(len(candidates), dtype=cp.uint64)
        periods = cp.empty(len(candidates), dtype=cp.int64)
        _kernel(cp.cuda.runtime.getDevice())(((len(candidates) + 127) // 128,), (128,), (
            self._timelineBits, np.uint64(self._timelineBits.shape[1]), flat, offsets,
            np.uint32(len(candidates)), np.int64(self._endTimestamp), np.float64(self._maxPer),
            supports, periods))
        return list(zip(cp.asnumpy(supports).tolist(), cp.asnumpy(periods).tolist()))

    def mine(self):
        """Evaluate complete candidate keys over singleton timestamp bitsets."""
        self._startTime = time.perf_counter()
        self._finalPatterns, self._periods = {}, {}
        self._timelineBits = None
        cp = self._getCudaModule()
        try:
            labels = self._prepareTimeline(cp)
            stack = [[(), list(range(len(labels))), 0]]
            while stack:
                prefix, extensions, position = stack[-1]
                if position >= len(extensions) - 1:
                    stack.pop()
                    continue
                stack[-1][2] += 1
                child = prefix + (extensions[position],)
                tails = extensions[position + 1:]
                kept = []
                for start in range(0, len(tails), self._BATCH_SIZE):
                    candidates = [child + (tail,) for tail in tails[start:start + self._BATCH_SIZE]]
                    for candidate, (support, period) in zip(candidates, self._evaluateTimeline(cp, candidates)):
                        pattern = tuple(labels[index] for index in candidate)
                        if self._accept(pattern, support, period):
                            kept.append(candidate[-1])
                if len(kept) > 1:
                    stack.append([child, kept, 0])
            cp.cuda.get_current_stream().synchronize()
        finally:
            self._timelineBits = None
            self._endTime = time.perf_counter()


if __name__ == '__main__':
    if len(sys.argv) not in (5, 6):
        sys.exit('Usage: python gPFMiner.py <inputFile> <outputFile> <minSup> <maxPer> [sep]')
    def threshold(text):
        return float(text) if any(c in text.lower() for c in '.e') else int(text)
    miner = gPFMiner(sys.argv[1], threshold(sys.argv[3]), threshold(sys.argv[4]),
                     sep=sys.argv[5] if len(sys.argv) == 6 else None)
    miner.mine()
    miner.save(sys.argv[2])
    miner.printResults()
