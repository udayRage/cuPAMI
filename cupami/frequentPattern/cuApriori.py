# cuApriori finds frequent patterns in transactional databases using CUDA.
# Apriori uses the downward closure property to prune candidate patterns.
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
import sys
import time
from itertools import chain, combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen


class cuApriori:
    def __init__(self, iFile, minSup, sep=None):
        self.iFile = iFile
        self.minSup = minSup
        self.sep = sep
        self._Database = self._readTransactions(iFile)
        self._minSup = self._getSupportCount(minSup, len(self._Database))
        self._finalPatterns: Dict[Tuple[str, ...], int] = {}
        self._startTime = 0.0
        self._endTime = 0.0

    def _readTransactions(self, source):
        if isinstance(source, str) and urlparse(source).scheme in ("http", "https"):
            with urlopen(source, timeout=30) as response:
                inputStream = io.TextIOWrapper(response, encoding="utf-8", newline="")
                return self._readDelimitedFile(inputStream)

        if isinstance(source, (str, Path)):
            with Path(source).open("r", newline="", encoding="utf-8") as inputFile:
                return self._readDelimitedFile(inputFile)

        return self._cleanTransactions(source)

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
            raise ImportError("cuApriori requires CuPy and a CUDA device") from error

        try:
            deviceCount = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as error:
            raise RuntimeError("cuApriori could not access CUDA") from error
        if deviceCount == 0:
            raise RuntimeError("cuApriori requires a CUDA device; none were found")
        return cp

    def _createTransactionMatrix(self, cp, items):
        itemIndexes = {item: index for index, item in enumerate(items)}
        matrix = cp.zeros((len(self._Database), len(items)), dtype=cp.bool_)

        for transactionIndex, transaction in enumerate(self._Database):
            columns = [itemIndexes[item] for item in transaction]
            if columns:
                matrix[transactionIndex, columns] = True

        return matrix, itemIndexes

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

    def _countCandidateSupports(self, cp, matrix, itemIndexes, candidates):
        candidateIndexes = cp.asarray(
            [[itemIndexes[item] for item in candidate] for candidate in candidates],
            dtype=cp.int64,
        )
        return cp.asnumpy(
            matrix[:, candidateIndexes].all(axis=2).sum(axis=0)
        ).tolist()

    def mine(self):
        cp = self._getCudaModule()
        self._startTime = time.perf_counter()

        items = sorted({item for transaction in self._Database for item in transaction})
        matrix, itemIndexes = self._createTransactionMatrix(cp, items)
        self._finalPatterns = {}

        itemSupports = cp.asnumpy(matrix.sum(axis=0)).tolist()
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
                cp, matrix, itemIndexes, candidates
            )
            nextLevel = []
            for candidate, support in zip(candidates, candidateSupports):
                if support >= self._minSup:
                    nextLevel.append(candidate)
                    self._finalPatterns[candidate] = int(support)
            currentLevel = nextLevel

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
