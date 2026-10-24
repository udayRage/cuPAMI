"""CUDA fuzzy frequent itemset mining from transformed CSV data.

Lin, Li, Fournier-Viger and Hong, "A fast Algorithm for mining fuzzy frequent
itemsets," JIFS 29(6), 2015, 2373–2379. DOI:10.3233/IFS-151936.
Maximum scalar cardinality and minimum-membership intersections follow the
paper and PAMI FFIMiner. Local GPU vertical search; README.md#references.
"""
from array import array
import csv
import io
import math
from numbers import Integral, Real
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
from urllib.request import urlopen
import numpy as np

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'cupami.fuzzyFrequentPattern'

from ..frequentPattern.cuApriori import cuApriori
from ._fuzzy import singleton_supports, intersect_memberships


class cuFFIMiner(cuApriori):
    """CSV/text input with rows labels:memberships; one region per base item.

    Memberships are finite values in [0, 1]. The suffix after a label's last
    dot denotes its region. Keep each base item's largest total membership
    region, resolving ties lexically. Supports are sums of per-row minima.
    Float minSup in (0, 1] is a fraction, without rounding; other positive
    numeric values are absolute fuzzy support. Parsing is CPU, mining CUDA.
    """
    _INTERSECTION_BATCH_BYTES = 64 * 1024 * 1024

    def __init__(self, iFile, minSup, sep=None):
        loading_start = time.perf_counter()
        if isinstance(minSup, bool) or not isinstance(minSup, Real) or not math.isfinite(minSup) or minSup <= 0:
            raise ValueError('minSup must be a positive finite number')
        if sep is not None and (not isinstance(sep, str) or len(sep) != 1 or sep in ':\r\n'):
            raise ValueError('sep must be a single item delimiter other than colon or newline')
        self.iFile, self.minSup, self.sep = iFile, minSup, sep
        self._startTime = self._endTime = 0.0
        self._finalPatterns = {}
        self._inputBackend = {'backend': 'cpu', 'reason': 'Fuzzy CSV parsing; mining uses CUDA'}
        self._readCsv(iFile)
        self._minSup = float(minSup)
        if not isinstance(minSup, Integral) and minSup <= 1:
            self._minSup *= self._transactionCount
        self._tolerance = 1e-12 * max(1.0, self._minSup)
        self._loadingTime = time.perf_counter() - loading_start

    def _readCsv(self, source):
        if not isinstance(source, (str, Path)):
            raise TypeError('cuFFIMiner requires a fuzzy CSV filename or text URL')
        is_url = urlparse(str(source)).scheme in ('http', 'https')
        suffix = Path(urlparse(str(source)).path if is_url else source).suffix.lower()
        if suffix in ('.bin', '.parquet'):
            raise ValueError('cuFFIMiner currently accepts fuzzy CSV/text only')
        if is_url:
            with urlopen(str(source), timeout=60) as response:
                with io.TextIOWrapper(response, encoding='utf-8-sig', newline='') as stream:
                    self._parseRows(stream)
        else:
            with Path(source).open(encoding='utf-8-sig', newline='') as stream:
                self._parseRows(stream)

    def _parseRows(self, stream):
        labels, indexes = [], {}
        rows, columns, memberships = array('Q'), array('I'), array('d')
        count = 0
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.strip().split(':')
            if len(parts) != 2:
                raise ValueError(f'Row {number}: expected items:memberships')
            separator = self.sep
            if separator is None:
                separator = '\t' if '\t' in line else ',' if ',' in line else None
            try:
                tokens = [next(csv.reader([part.strip()], delimiter=separator, strict=True))
                          if part.strip() and separator else part.split() for part in parts]
            except csv.Error as error:
                raise ValueError(f'Row {number}: invalid fuzzy CSV') from error
            items, values = [item.strip() for item in tokens[0]], tokens[1]
            if len(items) != len(values):
                raise ValueError(f'Row {number}: item and membership counts differ')
            if any(not item or any(c in item for c in '\t\r\n') for item in items):
                raise ValueError(f'Row {number}: invalid fuzzy label')
            if len(set(items)) != len(items):
                raise ValueError(f'Row {number}: duplicate fuzzy label')
            for item, text in zip(items, values):
                if '.' in item and not all(item.rsplit('.', 1)):
                    raise ValueError(f'Row {number}: expected item.Region label')
                try:
                    value = float(text)
                except ValueError as error:
                    raise ValueError(f'Row {number}: invalid membership') from error
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'Row {number}: memberships must be finite values in [0, 1]')
                if item not in indexes:
                    indexes[item] = len(labels)
                    labels.append(item)
                rows.append(count)
                columns.append(indexes[item])
                memberships.append(value)
            count += 1
        self._transactionCount, self._labels = count, labels
        self._rows = np.asarray(rows, dtype=np.uint64)
        self._columns = np.asarray(columns, dtype=np.uint32)
        self._memberships = np.asarray(memberships, dtype=np.float64)

    def _frequent(self, support):
        return support > 0 and support + self._tolerance >= self._minSup

    def _selectLabels(self, supports):
        best = {}
        for index in sorted(range(len(self._labels)), key=lambda i: self._labels[i]):
            base = self._labels[index].rsplit('.', 1)[0]
            previous = best.get(base)
            if previous is None or (supports[index] > supports[previous] and
                                    not math.isclose(supports[index], supports[previous], rel_tol=1e-12, abs_tol=1e-12)):
                best[base] = index
        return sorted((i for i in best.values() if self._frequent(supports[i])),
                          key=lambda i: (supports[i], self._labels[i]))

    def _extensionBatch(self, cp, prefix, names, memberships, start, stop):
        return names[start:stop], memberships[start:stop]

    def _singletons(self, cp):
        if not self._labels:
            return [], None
        columns, values = cp.asarray(self._columns), cp.asarray(self._memberships)
        supports = singleton_supports(cp, columns, values, len(self._labels))
        selected = self._selectLabels(supports)
        labels = [self._labels[i] for i in selected]
        if not labels:
            return [], None
        mapping = cp.full(len(self._labels), -1, dtype=cp.int64)
        mapping[cp.asarray(selected, dtype=cp.int64)] = cp.arange(len(selected))
        mapped = mapping[columns]
        keep = mapped >= 0
        matrix = cp.zeros((len(labels), self._transactionCount), dtype=cp.float64)
        matrix[mapped[keep], cp.asarray(self._rows)[keep]] = values[keep]
        for label, support in zip(labels, cp.asnumpy(matrix.sum(axis=1, dtype=cp.float64)).tolist()):
            self._finalPatterns[(label,)] = float(support)
        return labels, matrix

    def mine(self):
        self._startTime = time.perf_counter()
        cp = self._getCudaModule()
        with cp.cuda.using_allocator(cp.get_default_memory_pool().malloc):
            self._finalPatterns = {}
            labels, matrix = self._singletons(cp)
            if labels:
                self._search(cp, labels, matrix)
            cp.cuda.get_current_stream().synchronize()
            self._endTime = time.perf_counter()

    def _search(self, cp, labels, matrix):
        stack = [[(), labels, matrix, 0]]
        while stack:
            prefix, names, memberships, position = stack[-1]
            if position >= len(names) - 1:
                stack.pop()
                continue
            stack[-1][3] += 1
            child = prefix + (names[position],)
            kept_names, chunks = [], []
            batch_size = max(1, self._INTERSECTION_BATCH_BYTES // max(1, 8 * self._transactionCount))
            for start in range(position + 1, len(names), batch_size):
                stop = min(len(names), start + batch_size)
                batch_names, extensions = self._extensionBatch(cp, child, names, memberships, start, stop)
                if not batch_names:
                    continue
                intersections, supports = intersect_memberships(cp, memberships[position], extensions)
                keep = []
                for offset, support in enumerate(supports):
                    if self._frequent(support):
                        label = batch_names[offset]
                        self._finalPatterns[tuple(sorted(child + (label,)))] = float(support)
                        kept_names.append(label)
                        keep.append(offset)
                if keep:
                    chunks.append(intersections[cp.asarray(keep, dtype=cp.int64)])
            if len(kept_names) > 1:
                child_matrix = chunks[0] if len(chunks) == 1 else cp.concatenate(chunks)
                stack.append([child, kept_names, child_matrix, 0])


if __name__ == '__main__':
    if len(sys.argv) not in (4, 5):
        sys.exit('Usage: python cuFFIMiner.py <inputFile> <outputFile> <minSup> [sep]')
    try:
        support = int(sys.argv[3])
    except ValueError:
        support = float(sys.argv[3])
    miner = cuFFIMiner(sys.argv[1], support, sep=sys.argv[4] if len(sys.argv) == 5 else None)
    miner.mine()
    miner.save(sys.argv[2])
    miner.printResults()
