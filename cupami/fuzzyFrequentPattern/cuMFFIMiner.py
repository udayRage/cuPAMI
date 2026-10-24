"""CUDA multiple fuzzy frequent itemset mining.

Lin, Li, Fournier-Viger, Hong, Wu and Zhan, "Efficient Mining of Multiple
Fuzzy Frequent Itemsets," IJFS 19(4), 2017, 1032–1040 (online 2016).
DOI:10.1007/s40815-016-0246-1. All frequent regions are retained; a pattern
contains at most one region of each base item. GPU vertical adaptation;
Based on PAMI cuMFFIMiner (GPL-3.0), Copyright (C) 2026 Rage Uday Kiran.
PAMI and paper credits are in README.md#references.
"""
from pathlib import Path
import sys

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'cupami.fuzzyFrequentPattern'

from .cuFFIMiner import cuFFIMiner


class cuMFFIMiner(cuFFIMiner):
    """Retain all frequent fuzzy regions with at most one region per base item.

    Shares CSV/text/URL input, float64 support, thresholds, timing and output
    methods with cuFFIMiner. Search uses batched GPU minimum intersections.
    """
    def _selectLabels(self, supports):
        return sorted((i for i in range(len(self._labels)) if self._frequent(supports[i])),
                      key=lambda i: (supports[i], self._labels[i]))

    def _extensionBatch(self, cp, prefix, names, memberships, start, stop):
        used = {label.rsplit('.', 1)[0] for label in prefix}
        indexes = [i for i in range(start, stop) if names[i].rsplit('.', 1)[0] not in used]
        if len(indexes) == stop - start:
            return super()._extensionBatch(cp, prefix, names, memberships, start, stop)
        if not indexes:
            return [], None
        return [names[i] for i in indexes], memberships[cp.asarray(indexes, dtype=cp.int64)]


if __name__ == '__main__':
    if len(sys.argv) not in (4, 5):
        sys.exit('Usage: python cuMFFIMiner.py <inputFile> <outputFile> <minSup> [sep]')
    try:
        support = int(sys.argv[3])
    except ValueError:
        support = float(sys.argv[3])
    miner = cuMFFIMiner(sys.argv[1], support, sep=sys.argv[4] if len(sys.argv) == 5 else None)
    miner.mine()
    miner.save(sys.argv[2])
    miner.printResults()
