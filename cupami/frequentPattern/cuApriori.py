# cuApriori finds frequent patterns in transactional databases using CUDA.
# Apriori uses the downward closure property to prune candidate patterns.
# Algorithm: Agrawal and Srikant, "Fast Algorithms for Mining Association
# Rules," VLDB 1994, pp. 487-499, Section 2.1.
# https://www.vldb.org/conf/1994/P487.PDF
# CUDA adaptations and GDS API provenance: README.md#references.
#
# Example:
#
#     from cupami import cuApriori
#
#     miner = cuApriori(iFile, minSup)
#     miner.mine()
#     patterns = miner.getPatterns()
#     miner.save("patterns.txt")

from __future__ import annotations

import csv
import io
import math
import struct
import sys
import time
from contextlib import nullcontext
from itertools import chain, combinations
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen


if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "cupami.frequentPattern"


class cuApriori:
    def __init__(self, iFile, minSup, sep=None, use_gds="auto", use_gpu_io="auto"):
        if use_gds not in ("auto", True, False):
            raise ValueError('use_gds must be "auto", True, or False')
        if use_gpu_io not in ("auto", True, False):
            raise ValueError('use_gpu_io must be "auto", True, or False')
        self.iFile = iFile
        self.minSup = minSup
        self.sep = sep
        self.use_gds = use_gds
        self.use_gpu_io = use_gpu_io
        self._binaryDevice = None
        self._deviceTransactions = None
        self._inputBackend = {"backend": "cpu", "reason": "Text or Parquet input"}
        if use_gds is True and (not isinstance(iFile, (str, Path))
                                or Path(iFile).suffix.lower() != ".bin"
                                or urlparse(str(iFile)).scheme in ("http", "https")):
            raise ValueError("Required GDS input currently supports local BIN files only")
        self._Database = self._readTransactions(iFile)
        if self._Database is not None:
            self._transactionCount = len(self._Database)
        self._minSup = self._getSupportCount(minSup, self._transactionCount)
        self._finalPatterns: Dict[Tuple[str, ...], int] = {}
        self._startTime = 0.0
        self._endTime = 0.0

    def _readTransactions(self, source):
        if isinstance(source, str) and urlparse(source).scheme in ("http", "https"):
            with urlopen(source, timeout=30) as response:
                inputStream = io.TextIOWrapper(response, encoding="utf-8", newline="")
                return self._readDelimitedFile(inputStream)

        if isinstance(source, (str, Path)):
            suffix = Path(source).suffix.lower()
            if suffix == ".bin":
                return self._readBinaryFile(Path(source))
            if suffix == ".parquet":
                if self._tryGpuFile(source, parquet=True):
                    return None
                return self._readParquetFile(source)
            if self._tryGpuFile(source):
                return None
            with Path(source).open("r", newline="", encoding="utf-8") as inputFile:
                return self._readDelimitedFile(inputFile)

        return self._cleanTransactions(source)

    def _tryGpuFile(self, path, parquet=False):
        if self.use_gpu_io is False:
            return False
        from . import _gpu_io

        try:
            self._getCudaModule()
            ioContext = nullcontext()
            if self.use_gds is False:
                import kvikio
                import kvikio.defaults
                ioContext = kvikio.defaults.set("compat_mode", kvikio.CompatMode.ON)
            with ioContext:
                if parquet:
                    pairs = _gpu_io.read_parquet(path)
                else:
                    with Path(path).open("r", encoding="utf-8") as source:
                        firstLine = source.readline()
                    separator = self.sep or ("\t" if "\t" in firstLine else ",")
                    pairs = _gpu_io.read_text(path, separator)
        except (ImportError, RuntimeError, _gpu_io.UnsupportedTextLayout) as error:
            if self.use_gpu_io is True:
                raise RuntimeError(f"GPU input unavailable: {error}") from error
            self._inputBackend = {"backend": "cpu", "reason": str(error)}
            return False
        self._deviceTransactions = pairs[:3]
        self._transactionCount = pairs[3]
        self._inputBackend = {
            "backend": "cudf", "reason": "GPU parsing and transaction expansion",
            "direct_gds_verified": False,
        }
        return True

    def _readBinaryFile(self, path):
        """Read the transactional generator's CUPAMI01 format."""
        with path.open("rb") as source:
            header = source.read(32)
            if len(header) != 32:
                raise ValueError("Invalid BIN file: truncated header")
            magic, count, total, universe = struct.unpack("<8sQQQ", header)
            if magic != b"CUPAMI01":
                raise ValueError("Invalid BIN file: expected CUPAMI01 format")
            if not 1 <= universe <= 0xFFFFFFFF:
                raise ValueError("Invalid BIN file: item universe must fit in uint32")
            if path.stat().st_size != 32 + 8 * (count + 1) + 4 * total:
                raise ValueError("Invalid BIN file: size does not match header")
            if self._tryBinaryDeviceRead(path, count, total, universe):
                return None
            if self._tryBinaryBulkRead(path, count, total, universe):
                return None
            offsets = [value[0] for value in struct.iter_unpack(
                "<Q", source.read(8 * (count + 1))
            )]
            if (offsets[0] != 0 or offsets[-1] != total
                    or any(left > right for left, right in zip(offsets, offsets[1:]))):
                raise ValueError("Invalid BIN file: invalid transaction offsets")
            transactions = []
            for start, end in zip(offsets, offsets[1:]):
                items = [value[0] for value in struct.iter_unpack(
                    "<I", source.read(4 * (end - start))
                )]
                if any(item > universe for item in items):
                    raise ValueError("Invalid BIN file: item ID outside configured universe")
                transaction = self._cleanTransaction(items)
                if transaction:
                    transactions.append(transaction)
            return transactions

    def _tryBinaryBulkRead(self, path, count, total, universe):
        """Read typed arrays in bulk and transfer them without Python baskets."""
        if self.use_gpu_io is False:
            return False
        try:
            cp = self._getCudaModule()
        except (ImportError, RuntimeError) as error:
            if self.use_gpu_io is True:
                raise RuntimeError(f"GPU BIN input unavailable: {error}") from error
            return False
        import numpy as np

        with path.open("rb") as source:
            source.seek(32)
            offsets = np.fromfile(source, dtype="<u8", count=count + 1)
            ids = np.fromfile(source, dtype="<u4", count=total)
        if offsets.size != count + 1 or ids.size != total:
            raise ValueError("Invalid BIN file: truncated arrays")
        if (offsets[0] != 0 or offsets[-1] != total
                or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError("Invalid BIN file: invalid transaction offsets")
        if np.any(ids > universe):
            raise ValueError("Invalid BIN file: item ID outside configured universe")
        self._transactionCount = int(np.count_nonzero(offsets[1:] > offsets[:-1]))
        self._binaryDevice = (cp.asarray(offsets), cp.asarray(ids))
        cp.cuda.get_current_stream().synchronize()
        self._inputBackend = {
            "backend": "numpy-cuda",
            "reason": "Bulk numeric BIN reads and GPU transfers; no Python transactions",
            "gds_fallback_reason": self._inputBackend["reason"],
            "direct_gds_verified": False,
        }
        return True

    def _tryBinaryDeviceRead(self, path, count, total, universe):
        """Use NVIDIA cuFile through RAPIDS KvikIO; see README.md#references.

        This is local API integration, not a GDS algorithm from the Apriori paper.
        """
        if self.use_gds is False:
            self._inputBackend = {"backend": "cpu", "reason": "GDS disabled"}
            return False
        try:
            import kvikio
            import kvikio.defaults
            import kvikio.cufile_driver
            cp = self._getCudaModule()
            if not kvikio.cufile_driver.get("is_gds_available"):
                raise RuntimeError("cuFile reports GDS unavailable on this system")
            size = path.stat().st_size
            data = cp.empty(size, dtype=cp.uint8)
            # Disable KvikIO's POSIX path and small-read threshold. cuFile may
            # still fall back internally if its configuration permits it.
            with kvikio.defaults.set({"compat_mode": kvikio.CompatMode.OFF,
                                      "gds_threshold": 0}):
                with kvikio.CuFile(path, "r") as source:
                    if source.pread(data).get() != size:
                        raise RuntimeError("Short cuFile read")
            cp.cuda.get_current_stream().synchronize()
        except (ImportError, RuntimeError, OSError) as error:
            self._inputBackend = {"backend": "cpu", "reason": str(error)}
            if self.use_gds is True:
                raise RuntimeError(f"GDS input requested but unavailable: {error}") from error
            return False

        offsets = data[32:32 + 8 * (count + 1)].view(cp.uint64)
        ids = data[32 + 8 * (count + 1):].view(cp.uint32)
        if (int(offsets[0]) != 0 or int(offsets[-1]) != total
                or bool(cp.any(offsets[1:] < offsets[:-1]))):
            raise ValueError("Invalid BIN file: invalid transaction offsets")
        if bool(cp.any(ids > universe)):
            raise ValueError("Invalid BIN file: item ID outside configured universe")
        self._transactionCount = int(cp.count_nonzero(offsets[1:] > offsets[:-1]))
        self._binaryDevice = (offsets, ids)
        self._inputBackend = {
            "backend": "cufile",
            "reason": "BIN arrays loaded to GPU; cuFile internal fallback may still apply",
            "cufile_allows_compatibility": bool(
                kvikio.cufile_driver.get("allow_compat_mode")
            ),
        }
        return True

    def getInputBackend(self):
        """Report the selected input path without claiming verified direct DMA."""
        return dict(self._inputBackend)

    def _getDevicePairs(self, cp):
        if self._deviceTransactions is not None:
            return self._deviceTransactions
        if self._binaryDevice is not None:
            offsets, ids = self._binaryDevice
            unique = cp.unique(ids)
            numericItems = cp.asnumpy(unique).tolist()
            items = sorted(map(str, numericItems))
            itemIndexes = {item: index for index, item in enumerate(items)}
            # Empty baskets are omitted, matching the text and CPU readers.
            originalRows = cp.searchsorted(
                offsets[1:], cp.arange(ids.size, dtype=cp.uint64), side="right"
            )
            activeRows = cp.cumsum(offsets[1:] > offsets[:-1]) - 1
            rows = activeRows[originalRows]
            columnMap = cp.asarray([itemIndexes[str(item)] for item in numericItems],
                                   dtype=cp.uint32)
            columns = columnMap[cp.searchsorted(unique, ids)]
            return items, rows, columns
        items = sorted({item for row in self._Database for item in row})
        itemIndexes = {item: index for index, item in enumerate(items)}
        # Transfer all pairs in bulk, rather than launching one assignment per row.
        rows, columns = [], []
        for rowIndex, transaction in enumerate(self._Database):
            rows.extend([rowIndex] * len(transaction))
            columns.extend(itemIndexes[item] for item in transaction)
        return items, cp.asarray(rows, dtype=cp.uint64), cp.asarray(columns, dtype=cp.uint32)

    def _readParquetFile(self, path):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("Parquet input requires pyarrow: pip install pyarrow") from error
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        if "items" not in schema.names:
            raise ValueError("Invalid Parquet file: missing items column")
        itemType = schema.field("items").type
        if not pa.types.is_list(itemType) or not pa.types.is_uint32(itemType.value_type):
            raise ValueError("Invalid Parquet file: items must be a list of uint32 IDs")
        transactions = []
        for batch in parquet.iter_batches(columns=["items"]):
            rows = batch.column(0).to_pylist()
            if any(row is None or any(item is None for item in row) for row in rows):
                raise ValueError("Invalid Parquet file: null transactions or item IDs")
            transactions.extend(self._cleanTransactions(rows))
        return transactions

    def _readDelimitedFile(self, inputStream):
        firstLine = inputStream.readline()
        delimiter = self.sep
        if delimiter is None:
            delimiter = "\t" if "\t" in firstLine else ","
        rows = csv.reader(chain((firstLine,), inputStream), delimiter=delimiter)
        return self._cleanTransactions(rows)

    def _cleanTransactions(self, rows):
        transactions = []
        for row in rows:
            transaction = self._cleanTransaction(row)
            if transaction:
                transactions.append(transaction)
        return transactions

    @staticmethod
    def _cleanTransaction(transaction):
        items = []
        for value in transaction:
            item = str(value).strip()
            if item and item not in items:
                items.append(item)
        return tuple(items)

    @staticmethod
    def _getSupportCount(minSup, transactionCount):
        if isinstance(minSup, float) and 0 < minSup <= 1:
            return math.ceil(minSup * transactionCount)
        return int(minSup)

    @staticmethod
    def _getCudaModule():
        try:
            import cupy as cp
        except ImportError as error:
            raise ImportError("GPU mining requires CuPy and a CUDA device.") from error

        try:
            deviceCount = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as error:
            raise RuntimeError("GPU mining could not access CUDA.") from error
        if deviceCount == 0:
            raise RuntimeError("GPU mining requires CuPy and a CUDA device; no CUDA device was found.")
        return cp

    def _generateCandidates(self, previousLevel):
        if not previousLevel:
            return []

        patternSize = len(previousLevel[0]) + 1
        previousPatterns = set(previousLevel)
        candidates = set()
        orderedPatterns = sorted(previousLevel)

        for leftIndex, leftPattern in enumerate(orderedPatterns):
            for rightPattern in orderedPatterns[leftIndex + 1:]:
                if leftPattern[:-1] != rightPattern[:-1]:
                    continue

                candidate = leftPattern + (rightPattern[-1],)
                subsets = combinations(candidate, patternSize - 1)
                if all(tuple(subset) in previousPatterns for subset in subsets):
                    candidates.add(candidate)

        return sorted(candidates)

    def _countCandidateSupports(self, cp, bitsets, itemIndexes, candidates):
        from ._bitsets import count_supports

        return count_supports(cp, bitsets, [
            [itemIndexes[item] for item in candidate] for candidate in candidates
        ])

    def mine(self):
        cp = self._getCudaModule()
        from ._bitsets import build_bitsets, count_supports
        self._startTime = time.perf_counter()

        items, rows, columns = self._getDevicePairs(cp)
        itemIndexes = {item: index for index, item in enumerate(items)}
        bitsets = build_bitsets(cp, rows, columns, self._transactionCount, len(items))
        self._finalPatterns = {}

        itemSupports = count_supports(cp, bitsets, [[index] for index in range(len(items))])
        currentLevel = []
        for itemIndex, support in enumerate(itemSupports):
            if support >= self._minSup:
                pattern = (items[itemIndex],)
                currentLevel.append(pattern)
                self._finalPatterns[pattern] = int(support)

        while currentLevel:
            candidates = self._generateCandidates(currentLevel)
            if not candidates:
                break

            candidateSupports = self._countCandidateSupports(
                cp, bitsets, itemIndexes, candidates
            )
            nextLevel = []
            for candidate, support in zip(candidates, candidateSupports):
                if support >= self._minSup:
                    nextLevel.append(candidate)
                    self._finalPatterns[candidate] = int(support)
            currentLevel = nextLevel

        cp.cuda.get_current_stream().synchronize()
        self._endTime = time.perf_counter()

    def getPatterns(self):
        return {"\t".join(pattern): support
                for pattern, support in self._finalPatterns.items()}

    def getPatternsAsDataFrame(self):
        import pandas as pd

        rows = [("\t".join(pattern), support)
                for pattern, support in self._finalPatterns.items()]
        return pd.DataFrame(rows, columns=["Patterns", "Support"])

    def save(self, outputFile):
        with Path(outputFile).open("w", encoding="utf-8") as file:
            for pattern, support in sorted(self._finalPatterns.items()):
                patternText = "\t".join(pattern)
                file.write(f"{patternText} : {support}\n")

    def getRuntime(self):
        return self._endTime - self._startTime

    def getMemoryRSS(self):
        try:
            import psutil
        except ImportError:
            return 0
        return int(psutil.Process().memory_info().rss)

    def getMemoryUSS(self):
        try:
            import psutil
        except ImportError:
            return 0
        return int(psutil.Process().memory_full_info().uss)

    def printResults(self):
        print("Total number of Frequent Patterns:", len(self._finalPatterns))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS:", self.getMemoryRSS())
        print("Total ExecutionTime in seconds:", self.getRuntime())


Apriori = cuApriori


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit("Usage: python cuApriori.py <inputFile> <outputFile> <minSup> [sep]")

    try:
        minSup = int(sys.argv[3])
    except ValueError:
        minSup = float(sys.argv[3])

    sep = sys.argv[4] if len(sys.argv) == 5 else None
    miner = cuApriori(sys.argv[1], minSup, sep)
    miner.mine()
    miner.save(sys.argv[2])
    miner.printResults()
