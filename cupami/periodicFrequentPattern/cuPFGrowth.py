# cuPFGrowth combines conditional FP-tree search with GPU temporal occurrence checks.
# Patterns must satisfy minimum support and maximum periodicity.
#
# **Importing this algorithm into a python program**
#
#             import cupami as alg
#
#             iFile = 'sampleDB.csv'
#             oFile = 'patterns.txt'
#             minSup = 10  # absolute support; 0.1 means 10% of transactions
#             maxPer = 30
#
#             obj = alg.cuPFGrowth(iFile=iFile, minSup=minSup, maxPer=maxPer)
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

"""Local FP-tree search augmented with GPU temporal occurrence bitsets.

Based on Han et al.'s FP-growth search, with GPU periodicity checks.
See README.md#references for sources.

GPU references: Arour and Belkahla, CIT 2014, Section 5.3,
doi:10.2498/cit.1002361; PAMI gPFMinerBit for temporal GPU checks.
https://hrcak.srce.hr/en/clanak/190853
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/periodicFrequentPattern/cuda/gPFMinerBit.py
Local hybrid combines weighted conditional FP-tree arrays with bitset gap
checks. It is not a reproduction of flattened-array gPF-Miner.
See docs/algorithm-references.md for the mapping and limitations.
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

import time
from ._temporal import TemporalInput
from ..frequentPattern.cuFPGrowth import cuFPGrowth
from ..frequentPattern import _fptree


class cuPFGrowth(TemporalInput, cuFPGrowth):
    """
    **About this algorithm**

    :**Description**: cuPFGrowth combines conditional FP-tree search with GPU temporal occurrence checks. Patterns must satisfy minimum support and maximum periodicity.

    :**Reference**: Kiran, R. U., Kitsuregawa, M., Reddy, P. K.: Efficient Discovery of Periodic-Frequent Patterns in Very Large Databases. Journal of Systems and Software 112, pp. 110–121 (2016).
                    Related GPU research: Tarun, S., Kiran, R. U., Watanobe, Y., Goda, K.: A Novel GPU-Accelerated Algorithm to Discover Periodic-Frequent Patterns in Temporal Databases. IEEE Big Data, pp. 121–126 (2022). https://doi.org/10.1109/BigData55660.2022.10020216

    :**Parameters**:    - **iFile** (*str or pathlib.Path or iterable*) -- Timestamp-first transaction rows, temporal CSV/BIN/Parquet, or a CSV HTTP(S) URL.
                            - **minSup** (*int or float*) -- Positive minimum support. Integers are absolute support; floats in (0, 1] are fractions of the transaction count. Larger floats are absolute support.
                            - **maxPer** (*int or float*) -- Maximum gap including start and end gaps. A float in (0, 1] is a fraction of the timeline; other values are timestamp units.
                            - **sep** (*str or None*) -- Item delimiter. None detects supported delimiters from the input.
                            - **endTime** (*int or None*) -- Optional observation end timestamp; defaults to the largest input timestamp.

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

        python3 -c "import cupami as alg; iFile = 'sampleDB.csv'; oFile = 'patterns.txt'; minSup = 10; maxPer = 30; obj = alg.cuPFGrowth(iFile=iFile, minSup=minSup, maxPer=maxPer); obj.mine(); obj.save(oFile)"

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
        maxPer = 30

        obj = alg.cuPFGrowth(iFile=iFile, minSup=minSup, maxPer=maxPer)
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

    Conditional FP-tree mining with temporal checks before expansion.
    """
    def mine(self):
        self._startTime = time.perf_counter()
        cp = self._getCudaModule()
        with cp.cuda.using_allocator(cp.get_default_memory_pool().malloc):
            labels, rows, columns = self._prepare(cp)
            rows, columns = _fptree.unique_pairs(cp, rows, columns)
            weights = cp.ones(self._transactionCount, dtype=cp.uint64)
            root = _fptree.build_tree(cp, rows, columns, weights, len(labels), self._minSup)
            if root is not None:
                self._search(cp, root, labels)
            cp.cuda.get_current_stream().synchronize()
            self._endTime = time.perf_counter()

    def _storePattern(self, item_ids, support, labels):
        pattern = tuple(sorted(labels[index] for index in item_ids))
        measured, period = self._evaluate(self._getCudaModule(), [pattern])[0]
        return self._accept(pattern, measured, period)
