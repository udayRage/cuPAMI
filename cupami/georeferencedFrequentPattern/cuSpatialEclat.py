# cuSpatialEclat discovers spatial frequent patterns using neighbour-constrained Eclat
# search. Neighbour compatibility is checked on the CPU and support is counted with CUDA
# bitset intersections.
#
# **Importing this algorithm into a python program**
#
#             import cupami as alg
#
#             iFile = 'sampleDB.csv'
#             nFile = 'neighbours.csv'
#             oFile = 'patterns.txt'
#             minSup = 10  # absolute support; 0.1 means 10% of transactions
#
#             obj = alg.cuSpatialEclat(iFile=iFile, nFile=nFile, minSup=minSup)
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

"""Spatial Eclat with CUDA vertical-bitset support counting.

Neighbour-constrained prefix search follows PAMI SpatialECLAT (B. Sai Chitra,
Copyright 2021 Rage Uday Kiran, GPL-3.0). Spatial model: Kiran et al.,
SIGSPATIAL 2020, doi:10.1145/3397536.3422206. See README.md#references.

Implementation references: PAMI SpatialECLAT for neighbour pruning,
and PAMI cuEclatBit for CUDA vertical intersections.
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/georeferencedFrequentPattern/basic/SpatialECLAT.py
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/frequentPattern/cuda/cuEclatBit.py
Local hybrid checks neighbour compatibility on the CPU and counts support
on the GPU; no published CUDA SpatialECLAT implementation is claimed.
See docs/algorithm-references.md.
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
    """
    **About this algorithm**

    :**Description**: cuSpatialEclat discovers spatial frequent patterns using neighbour-constrained Eclat search. Neighbour compatibility is checked on the CPU and support is counted with CUDA bitset intersections.

    :**Reference**: Kiran, R. U., Shrivastava, S., Fournier-Viger, P., Zettsu, K., Toyoda, M., Kitsuregawa, M.: Discovering Frequent Spatial Patterns in Very Large Spatiotemporal Databases. SIGSPATIAL, pp. 445–448 (2020). https://doi.org/10.1145/3397536.3422206

    :**Parameters**:    - **iFile** (*str or pathlib.Path*) -- Item-only transaction CSV, provided as a local path or HTTP(S) URL.
                            - **nFile** (*str or pathlib.Path*) -- Neighbour CSV: each row contains an item followed by its neighbours. Local paths and HTTP(S) URLs are supported.
                            - **minSup** (*int or float*) -- Positive minimum support. Integers are absolute support; floats in (0, 1] are fractions of the transaction count. Larger floats are absolute support.
                            - **sep** (*str or None*) -- Item delimiter. None detects supported delimiters from the input.

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

        Run the following Python invocation from a terminal:

        Example Usage:

        python3 -c "import cupami as alg; iFile = 'sampleDB.csv'; nFile = 'neighbours.csv'; oFile = 'patterns.txt'; minSup = 10; obj = alg.cuSpatialEclat(iFile=iFile, nFile=nFile, minSup=minSup); obj.mine(); obj.save(oFile)"

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
        nFile = 'neighbours.csv'
        oFile = 'patterns.txt'
        minSup = 10  # absolute support; 0.1 means 10% of transactions

        obj = alg.cuSpatialEclat(iFile=iFile, nFile=nFile, minSup=minSup)
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

    Mine frequent patterns using PAMI's neighbour-constrained Eclat search.

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
