"""Spatial Eclat with CUDA vertical-bitset support counting.

Neighbour-constrained prefix search follows PAMI SpatialECLAT (B. Sai Chitra,
Copyright 2021 Rage Uday Kiran, GPL-3.0). Spatial model: Kiran et al.,
SIGSPATIAL 2020, doi:10.1145/3397536.3422206. See README.md#references.
"""
import csv
import io
import math
import time
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from ..frequentPattern.cuEclat import cuEclat


class cuSpatialEclat(cuEclat):
    """Mine frequent patterns using PAMI's neighbour-constrained Eclat search.

    iFile is an item-only CSV; nFile rows contain an item followed by its
    neighbours. Both accept local paths or HTTP(S) URLs. sep=None detects
    tabs or commas separately for each file. No timestamp column is inferred.
    Neighbours are used as listed, following PAMI SpatialECLAT.
    """

    def __init__(self, iFile, nFile, minSup, sep=None):
        start = time.perf_counter()
        if isinstance(minSup, bool) or not math.isfinite(float(minSup)) or float(minSup) <= 0:
            raise ValueError("minSup must be positive and finite")
        self.nFile = nFile
        super().__init__(iFile, minSup, sep=sep, use_gds=False, use_gpu_io=False)
        if self._minSup < 1 and self._transactionCount:
            raise ValueError("Absolute minSup must be at least 1")
        items = {item for row in self._Database for item in row}
        self._neighbours = {item: set() for item in items}
        for row in self._csvRows(nFile):
            row = [value.strip() for value in row if value.strip()]
            if row and row[0] in items:
                self._neighbours[row[0]] = set(row[1:]) & items
        self._inputBackend = {"backend": "cpu", "reason": "Spatial CSV parsing and neighbour preparation"}
        self._loadingTime = time.perf_counter() - start

    @staticmethod
    @contextmanager
    def _openCsv(source):
        if not isinstance(source, (str, Path)) or urlparse(str(source)).path.lower().endswith((".bin", ".parquet")):
            raise ValueError("cuSpatialEclat accepts CSV paths or URLs only")
        if urlparse(str(source)).scheme in ("http", "https"):
            with urlopen(str(source), timeout=60) as response:
                with io.TextIOWrapper(response, encoding="utf-8-sig", newline="") as stream:
                    yield stream
        else:
            with Path(source).open(encoding="utf-8-sig", newline="") as stream:
                yield stream

    def _csvRows(self, source):
        with self._openCsv(source) as stream:
            first = stream.readline()
            delimiter = self.sep or ("\t" if "\t" in first else ",")
            yield from csv.reader(chain((first,), stream), delimiter=delimiter)

    def _readTransactions(self, source):
        return self._cleanTransactions(self._csvRows(source))

    def _candidatePairs(self, frequent, items):
        for position, left in enumerate(frequent):
            neighbours = self._neighbours[items[left]]
            for right in frequent[position+1:]:
                if items[right] in neighbours:
                    yield left, right

    def _extensionBatch(self, cp, prefix, labels, bitsets, start, stop):
        allowed = self._neighbours[prefix[0]].copy()
        for item in prefix[1:]:
            allowed.intersection_update(self._neighbours[item])
        selected = [i for i in range(start, stop) if labels[i] in allowed]
        if not selected:
            return [], None
        return [labels[i] for i in selected], bitsets[cp.asarray(selected, dtype=cp.int64)]
