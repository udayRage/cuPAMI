# cuPFEclat discovers periodic frequent patterns using depth-first search over timestamp
# occurrence bitsets. GPU checks enforce support and maximum-gap constraints.
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
#             obj = alg.cuPFEclat(iFile=iFile, minSup=minSup, maxPer=maxPer)
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

"""Local CUDA vertical prefix search with periodicity pruning.

Uses Eclat's depth-first equivalence classes and intersects timestamp
occurrence bitsets on the GPU. This is a local adaptation, not author code.

GPU reference: PAMI gPFMinerBit and cuGPFMiner.
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/periodicFrequentPattern/cuda/gPFMinerBit.py
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/periodicFrequentPattern/cuda/cuGPFMiner.py
Follows depth-first prefix search and GPU temporal intersections. Local
counting uses transaction-row bitsets and explicit int64 timestamps.
Tarun et al., IEEE Big Data 2022, doi:10.1109/BigData55660.2022.10020216,
provides related flattened-array gPF-Miner; this is the bitset variant.
See docs/algorithm-references.md for differences and boundary semantics.
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


class cuPFEclat(TemporalInput):
    """
    **About this algorithm**

    :**Description**: cuPFEclat discovers periodic frequent patterns using depth-first search over timestamp occurrence bitsets. GPU checks enforce support and maximum-gap constraints.

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

        python3 -c "import cupami as alg; iFile = 'sampleDB.csv'; oFile = 'patterns.txt'; minSup = 10; maxPer = 30; obj = alg.cuPFEclat(iFile=iFile, minSup=minSup, maxPer=maxPer); obj.mine(); obj.save(oFile)"

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

        obj = alg.cuPFEclat(iFile=iFile, minSup=minSup, maxPer=maxPer)
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

    Depth-first periodic-frequent mining over vertical occurrence bitsets.
    """
    _BATCH_SIZE = 4096

    def mine(self):
        self._startTime = time.perf_counter()
        cp = self._getCudaModule()
        labels, _, _ = self._prepare(cp)
        singles = [(label,) for label in labels]
        frequent = []
        for start in range(0, len(singles), self._BATCH_SIZE):
            batch = singles[start:start + self._BATCH_SIZE]
            for pattern, (support, period) in zip(batch, self._evaluate(cp, batch)):
                if self._accept(pattern, support, period):
                    frequent.append(pattern[0])
        stack = [[(), frequent, 0]]
        while stack:
            prefix, extensions, position = stack[-1]
            if position >= len(extensions) - 1:
                stack.pop()
                continue
            stack[-1][2] += 1
            child = prefix + (extensions[position],)
            tails = extensions[position + 1:]
            kept = []
            for start in range(0, len(tails), self._BATCH_SIZE):
                batch = [child + (item,) for item in tails[start:start + self._BATCH_SIZE]]
                for pattern, (support, period) in zip(batch, self._evaluate(cp, batch)):
                    if self._accept(pattern, support, period):
                        kept.append(pattern[-1])
            if len(kept) > 1:
                stack.append([child, kept, 0])
        cp.cuda.get_current_stream().synchronize()
        self._endTime = time.perf_counter()
