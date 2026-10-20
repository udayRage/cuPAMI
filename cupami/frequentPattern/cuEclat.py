"""CUDA Eclat using vertical bitsets and prefix equivalence classes.

Algorithm: Zaki, Parthasarathy, Ogihara and Li, "New Algorithms for Fast
Discovery of Association Rules," KDD 1997, pp. 283-286.
https://www.cs.rpi.edu/~zaki/PaperDir/KDD97.pdf
Expanded treatment: Zaki, "Scalable Algorithms for Association Mining,"
IEEE TKDE 12(3), 2000, pp. 372-390. doi:10.1109/69.846291.

This is a local CUDA/bitset adaptation, not the authors' published C++ code.
It batches pair preprocessing, then intersects retained vertical tidsets in
depth-first prefix classes. See docs/references.md for implementation sources.
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
    """Mine all frequent itemsets using Eclat rather than Apriori search.

    Shares only input handling, support interpretation, output and reporting
    utilities with cuApriori. CSV/BIN/Parquet, text URLs and transaction
    iterables use the same constructor arguments and fallback controls.
    GPU intersections/popcount are combined with CPU prefix scheduling.
    """

    _PAIR_BATCH_SIZE = 65536
    _INTERSECTION_BATCH_BYTES = 64 * 1024 * 1024

    def mine(self):
        cp = self._getCudaModule()
        self._startTime = time.perf_counter()
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
        pairs = combinations(frequent, 2)
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
                intersections, supports = _bitsets.intersect_extensions(
                    cp, vertical[position], vertical[start:stop]
                )
                keep = []
                for offset, support in enumerate(supports):
                    if support >= self._minSup:
                        name = currentLabels[start + offset]
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
