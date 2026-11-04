# cuEclat discovers frequent itemsets with depth-first prefix search. CUDA kernels intersect
# transaction bitsets and count support for candidate extensions.
#
# **Importing this algorithm into a python program**
#
#             import cupami as alg
#
#             iFile = 'sampleDB.csv'
#             oFile = 'patterns.txt'
#             minSup = 10  # absolute support; 0.1 means 10% of transactions
#
#             obj = alg.cuEclat(iFile=iFile, minSup=minSup)
#             obj.mine()
#
#             frequentPatterns = obj.getPatterns()
#             print("Total number of patterns:", len(frequentPatterns))
#
#             obj.save(oFile)
#             Df = obj.getPatternsAsDataFrame()  # requires pandas
#
#             print("Total Memory in USS:", obj.getMemoryUSS())
#             print("Total Memory in RSS:", obj.getMemoryRSS())
#             print("Mining time in seconds:", obj.getRuntime())
#             print("Loading time in seconds:", obj.getLoadingTime())
#             print("Total time in seconds:", obj.getTotalRuntime())
#

"""CUDA Eclat using vertical bitsets and prefix equivalence classes.

Algorithm: Zaki, Parthasarathy, Ogihara and Li, "New Algorithms for Fast
Discovery of Association Rules," KDD 1997, pp. 283-286.
https://www.cs.rpi.edu/~zaki/PaperDir/KDD97.pdf
Expanded treatment: Zaki, "Scalable Algorithms for Association Mining,"
IEEE TKDE 12(3), 2000, pp. 372-390. doi:10.1109/69.846291.

This is a local CUDA/bitset adaptation, not the authors' published C++ code.
It batches pair preprocessing, then intersects retained vertical tidsets in
depth-first prefix classes. See README.md#references for implementation sources.

CUDA implementation reference: PAMI cuEclatBit (GPL-3.0).
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/frequentPattern/cuda/cuEclatBit.py
Follows vertical GPU AND/popcount joins. cuPAMI retains child intersections
in depth-first prefix classes and uses a block reduction per extension.
GPU word-intersection/reduction context: Silvestri and Orlando, gpuDCI,
PDP 2012, Section 4. doi:10.1109/PDP.2012.94.
https://www.dsi.unive.it/~orlando/PUB/gpudci2012.pdf
This implements Eclat search, not DCI's search. See docs/algorithm-references.md.
"""

__copyright__ = """
Copyright (C)  2026 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program. If not, see <https://www.gnu.org/licenses/>.
"""


from itertools import combinations, islice
from pathlib import Path
import sys
import time

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "cupami.frequentPattern"

from .cuApriori import cuApriori
from . import _bitsets


class cuEclat(cuApriori):
    """
    **About this algorithm**

    :**Description**: cuEclat discovers frequent itemsets with depth-first prefix search. CUDA kernels intersect transaction bitsets and count support for candidate extensions.

    :**Reference**: Zaki, M. J.: Scalable Algorithms for Association Mining. IEEE TKDE 12(3), pp. 372–390 (2000). https://doi.org/10.1109/69.846291
                    Silvestri, C., Orlando, S.: gpuDCI: Exploiting GPUs in Frequent Itemset Mining. PDP (2012). https://doi.org/10.1109/PDP.2012.94

    :**Parameters**:    - **iFile** (*str or pathlib.Path or iterable*) -- Transaction rows or a CSV, BIN, or Parquet input file. CSV HTTP(S) URLs are also supported.
                            - **minSup** (*int or float*) -- Positive minimum support. Integers are absolute support; floats in (0, 1] are fractions of the transaction count. Larger floats are absolute support.
                            - **sep** (*str or None*) -- Item delimiter. None detects supported delimiters from the input.
                            - **use_gds** (*str or bool*) -- "auto", True, or False. Required GDS supports local BIN input only.
                            - **use_gpu_io** (*str or bool*) -- "auto", True, or False. Controls supported GPU input loading paths.

    :**Attributes**:    - **_startTime** (*float*) -- Start time of the latest mining call.
                            - **_endTime** (*float*) -- Completion time of the latest mining call.
                            - **_finalPatterns** (*mapping*) -- Patterns and supports retained by the miner.
                            - **_loadingTime** (*float*) -- Input loading time in seconds.
                            - **_transactionCount** (*int*) -- Number of input transactions used by the miner.
                            - **_Database** (*list or None*) -- Parsed transactions, or None when retained by a GPU loading path.

    **Execution methods**

    **Terminal command**

    .. code-block:: console

        Format:

        python3 -m cupami.frequentPattern.cuEclat <inputFile> <outputFile> <minSup> [sep]

        Example Usage:

        python3 -m cupami.frequentPattern.cuEclat sampleDB.csv patterns.txt 10

    .. note:: Mining requires CuPy and an NVIDIA CUDA GPU. Memory methods report
              process RSS/USS in bytes when psutil is available, otherwise zero.
              getRuntime() reports mining time; getTotalRuntime() adds loading.
              Parquet support, where available, requires PyArrow. Dataframe output
              requires pandas. See docs/algorithm-references.md for source credits
              and differences from the referenced algorithms.

    **Calling from a python program**

    .. code-block:: python

        import cupami as alg

        iFile = 'sampleDB.csv'
        oFile = 'patterns.txt'
        minSup = 10  # absolute support; 0.1 means 10% of transactions

        obj = alg.cuEclat(iFile=iFile, minSup=minSup)
        obj.mine()

        frequentPatterns = obj.getPatterns()
        print("Total number of patterns:", len(frequentPatterns))

        obj.save(oFile)
        Df = obj.getPatternsAsDataFrame()  # requires pandas

        print("Total Memory in USS:", obj.getMemoryUSS())
        print("Total Memory in RSS:", obj.getMemoryRSS())
        print("Mining time in seconds:", obj.getRuntime())
        print("Loading time in seconds:", obj.getLoadingTime())
        print("Total time in seconds:", obj.getTotalRuntime())

    **Credits**

    The cuPAMI implementation was written by Mithun Thangaraj under the supervision
    of Professor Rage Uday Kiran. Adapted source credits and copyright notices are
    retained in the module documentation and docs/algorithm-references.md.

    **Implementation notes**

    Mine all frequent itemsets using Eclat rather than Apriori search.

    Shares only input handling, support interpretation, output and reporting
    utilities with cuApriori. CSV/BIN/Parquet, text URLs and transaction
    iterables use the same constructor arguments and fallback controls.
    GPU intersections/popcount are combined with CPU prefix scheduling.
    """

    _PAIR_BATCH_SIZE = 65536
    _INTERSECTION_BATCH_BYTES = 64 * 1024 * 1024

    def mine(self):
        self._startTime = time.perf_counter()
        cp = self._getCudaModule()
        self._finalPatterns = {}
        items, rows, columns = self._getDevicePairs(cp)
        bitsets = _bitsets.build_bitsets(cp, rows, columns, self._transactionCount, len(items))
        supports = _bitsets.count_supports(cp, bitsets, [[i] for i in range(len(items))])
        frequent = []
        for index, support in enumerate(supports):
            if support >= self._minSup:
                frequent.append(index)
                self._finalPatterns[(items[index],)] = int(support)

        # Pair preprocessing forms the prefix classes described in the Eclat
        # papers. Its cost is included in mine(), not amortized away.
        classes = {}
        pairs = self._candidatePairs(frequent, items)
        while True:
            batch = list(islice(pairs, self._PAIR_BATCH_SIZE))
            if not batch:
                break
            pairSupports = _bitsets.count_supports(cp, bitsets, batch)
            for (left, right), support in zip(batch, pairSupports):
                if support >= self._minSup:
                    self._finalPatterns[(items[left], items[right])] = int(support)
                    classes.setdefault(left, []).append(right)

        for left, extensions in classes.items():
            if len(extensions) < 2:
                continue
            labels = [items[index] for index in extensions]
            vertical = bitsets[cp.asarray(extensions, dtype=cp.uint32)] & bitsets[left]
            self._searchClass(cp, (items[left],), labels, vertical)

        cp.cuda.get_current_stream().synchronize()
        self._endTime = time.perf_counter()

    def _candidatePairs(self, frequent, items):
        return combinations(frequent, 2)

    def _extensionBatch(self, cp, prefix, labels, bitsets, start, stop):
        return labels[start:stop], bitsets[start:stop]

    def _searchClass(self, cp, prefix, labels, bitsets):
        # Explicit DFS frames avoid Python's recursion limit and keep only
        # classes along the active search path, rather than all sibling trees.
        stack = [[prefix, labels, bitsets, 0]]
        while stack:
            frame = stack[-1]
            currentPrefix, currentLabels, vertical, position = frame
            if position >= len(currentLabels) - 1:
                stack.pop()
                continue
            frame[3] += 1
            nextPrefix = currentPrefix + (currentLabels[position],)
            names, chunks = [], []
            words = vertical.shape[1]
            batchSize = max(1, self._INTERSECTION_BATCH_BYTES // max(1, 4 * words))
            for start in range(position + 1, len(currentLabels), batchSize):
                stop = min(len(currentLabels), start + batchSize)
                batchLabels, batchBits = self._extensionBatch(
                    cp, nextPrefix, currentLabels, vertical, start, stop
                )
                if not batchLabels:
                    continue
                intersections, supports = _bitsets.intersect_extensions(cp, vertical[position], batchBits)
                keep = []
                for offset, support in enumerate(supports):
                    if support >= self._minSup:
                        name = batchLabels[offset]
                        self._finalPatterns[nextPrefix + (name,)] = int(support)
                        names.append(name)
                        keep.append(offset)
                if keep:
                    chunks.append(intersections[cp.asarray(keep, dtype=cp.int64)])
            if len(names) > 1:
                childBits = chunks[0] if len(chunks) == 1 else cp.concatenate(chunks)
                stack.append([nextPrefix, names, childBits, 0])


Eclat = cuEclat


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit("Usage: python cuEclat.py <inputFile> <outputFile> <minSup> [sep]")
    try:
        minSup = int(sys.argv[3])
    except ValueError:
        minSup = float(sys.argv[3])
    sep = sys.argv[4] if len(sys.argv) == 5 else None
    miner = cuEclat(sys.argv[1], minSup, sep)
    miner.mine()
    miner.save(sys.argv[2])
    miner.printResults()
