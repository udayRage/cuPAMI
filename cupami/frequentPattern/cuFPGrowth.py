# cuFPGrowth discovers frequent itemsets using FP-tree search. GPU tree arrays and
# conditional projections support mining without Apriori candidate generation.
#
# **Importing this algorithm into a python program**
#
#             import cupami as alg
#
#             iFile = 'sampleDB.csv'
#             oFile = 'patterns.txt'
#             minSup = 10  # absolute support; 0.1 means 10% of transactions
#
#             obj = alg.cuFPGrowth(iFile=iFile, minSup=minSup)
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

"""CUDA FP-growth using GPU FP-trees and weighted conditional pattern bases.

Algorithm: Jiawei Han, Jian Pei and Yiwen Yin, "Mining Frequent Patterns
without Candidate Generation," SIGMOD 2000, pp. 1-12.
https://doi.org/10.1145/342009.335372
Algorithms 1 and 2 define tree construction and FP-growth; Lemma 3.2 gives
the single-path optimization. This is a local GPU adaptation, not the
authors' source code. See README.md#references.

GPU reference: Arour and Belkahla, CIT 22(3), 2014, Section 5.3,
doi:10.2498/cit.1002361. https://hrcak.srce.hr/en/clanak/190853
Follows GPU array representations and projected conditional mining.
Local adaptation uses parent/item/count arrays, CuPy level grouping and
weighted ancestor kernels; it does not reproduce the paper's OpenCL
FP-array/list layout. CPU schedules conditional trees.
See docs/algorithm-references.md for the technique-to-code mapping.
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


from itertools import combinations
from pathlib import Path
import sys
import time

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "cupami.frequentPattern"

from .cuApriori import cuApriori
from . import _fptree


class cuFPGrowth(cuApriori):
    """
    **About this algorithm**

    :**Description**: cuFPGrowth discovers frequent itemsets using FP-tree search. GPU tree arrays and conditional projections support mining without Apriori candidate generation.

    :**Reference**: Han, J., Pei, J., Yin, Y.: Mining Frequent Patterns without Candidate Generation. SIGMOD, pp. 1–12 (2000). https://doi.org/10.1145/342009.335372
                    Arour, K., Belkahla, A.: Frequent Pattern-growth Algorithm on Multi-core CPU and GPU Processors. CIT 22(3), pp. 159–169 (2014). https://doi.org/10.2498/cit.1002361

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

        python3 -m cupami.frequentPattern.cuFPGrowth <inputFile> <outputFile> <minSup> [sep]

        Example Usage:

        python3 -m cupami.frequentPattern.cuFPGrowth sampleDB.csv patterns.txt 10

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

        obj = alg.cuFPGrowth(iFile=iFile, minSup=minSup)
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

    Mine frequent itemsets through FP-trees, without Apriori candidates.

    Shares readers and output APIs with cuApriori. Weighted supports, sorted
    baskets, FP-tree arrays, and conditional projections are computed using
    CuPy on CUDA. CPU code schedules the search and formats output patterns.
    A positive minimum support count is required for nonempty input.
    """

    def mine(self):
        self._startTime = time.perf_counter()
        cp = self._getCudaModule()
        # cuDF can change CuPy's process-wide allocator during import. Keep
        # tree allocation consistent across readers without changing that
        # global choice for the caller or for cuDF itself.
        with cp.cuda.using_allocator(cp.get_default_memory_pool().malloc):
            self._mineWithPool(cp)

    def _mineWithPool(self, cp):
        self._finalPatterns = {}
        labels, rows, columns = self._getDevicePairs(cp)
        if labels and self._minSup < 1:
            raise ValueError("FP-growth requires a positive minimum support count")
        rows, columns = _fptree.unique_pairs(cp, rows, columns)
        weights = cp.ones(self._transactionCount, dtype=cp.uint64)
        root = _fptree.build_tree(cp, rows, columns, weights, len(labels), self._minSup)
        if root is not None:
            self._search(cp, root, labels)
        cp.cuda.get_current_stream().synchronize()
        self._endTime = time.perf_counter()

    def _storePattern(self, item_ids, support, labels):
        pattern = tuple(sorted(labels[index] for index in item_ids))
        self._finalPatterns[pattern] = int(support)

    def _search(self, cp, root, labels):
        # Explicit frames preserve depth-first scheduling without relying on
        # Python's recursion limit. Each frame owns its conditional FP-tree.
        stack = [[root, (), 0]]
        while stack:
            tree, suffix, position = stack[-1]
            if tree.single_path:
                path_items = cp.asnumpy(tree.items[1:]).tolist()
                path_counts = cp.asnumpy(tree.counts[1:]).tolist()
                for length in range(1, len(path_items) + 1):
                    for selected in combinations(range(len(path_items)), length):
                        ids = suffix + tuple(path_items[index] for index in selected)
                        support = min(path_counts[index] for index in selected)
                        self._storePattern(ids, support, labels)
                stack.pop()
                continue
            if position >= len(tree.header):
                stack.pop()
                continue
            stack[-1][2] += 1
            item, support = tree.header[position]
            next_suffix = suffix + (item,)
            if self._storePattern(next_suffix, support, labels) is False:
                continue
            conditional = _fptree.conditional_tree(cp, tree, item, len(labels), self._minSup)
            if conditional is not None:
                stack.append([conditional, next_suffix, 0])


FPGrowth = cuFPGrowth


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit("Usage: python cuFPGrowth.py <inputFile> <outputFile> <minSup> [sep]")
    try:
        minSup = int(sys.argv[3])
    except ValueError:
        minSup = float(sys.argv[3])
    sep = sys.argv[4] if len(sys.argv) == 5 else None
    miner = cuFPGrowth(sys.argv[1], minSup, sep)
    miner.mine()
    miner.save(sys.argv[2])
    miner.printResults()
