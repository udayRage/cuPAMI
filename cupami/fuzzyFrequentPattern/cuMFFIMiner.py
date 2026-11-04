# cuMFFIMiner discovers multiple fuzzy frequent patterns using shared base-item traversal
# (SBT). Regions of the same base item share transaction matching while retaining separate
# supports. Each pattern contains at most one region of a base item.
#
# **Importing this algorithm into a python program**
#
#             import cupami as alg
#
#             iFile = 'fuzzy.csv'
#             oFile = 'patterns.txt'
#             minSup = 10  # absolute support; 0.1 means 10% of transactions
#
#             obj = alg.cuMFFIMiner(iFile=iFile, minSup=minSup)
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

"""Shared base-item traversal for multiple fuzzy frequent pattern mining.

Accompanies "cuMFFP-Miner: GPU-Accelerated Multiple Fuzzy Frequent Pattern
Mining via Shared Base-Item Traversal" by Mithun Thangaraj and Rage Uday Kiran,
The University of Aizu, Japan.
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

from pathlib import Path
import sys

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'cupami.fuzzyFrequentPattern'

from ._multi_api import _MultiMiner, _main


class cuMFFIMiner(_MultiMiner):
    """
    **About this algorithm**

    :**Description**: cuMFFIMiner discovers multiple fuzzy frequent patterns using shared base-item traversal (SBT). Regions of the same base item share transaction matching while retaining separate supports. Each pattern contains at most one region of a base item.

    :**Reference**: Thangaraj, M., Kiran, R. U.: cuMFFP-Miner: GPU-Accelerated Multiple Fuzzy Frequent Pattern Mining via Shared Base-Item Traversal. The University of Aizu, Japan.
                    Fuzzy support model: Lin et al.: Efficient Mining of Multiple Fuzzy Frequent Itemsets. IJFS 19(4), pp. 1032–1040 (2017; online 2016). https://doi.org/10.1007/s40815-016-0246-1

    :**Parameters**:    - **iFile** (*str or pathlib.Path*) -- Fuzzy CSV/text file with labels:memberships rows. Local files only.
                            - **minSup** (*int or float*) -- Positive minimum support. Integers are absolute support; floats in (0, 1] are fractions of the transaction count. Larger floats are absolute support.
                            - **sep** (*str or None*) -- Item delimiter. None detects supported delimiters from the input.
                            - **supportScale** (*int or float*) -- Positive divisor applied to minSup; default 1. Fractional support is used only with supportScale=1.
                            - **membershipScale** (*int*) -- Membership representation scale; default 10 (tenths). Choose a scale matching input precision.
                            - **repeat** (*int*) -- Physical transaction repetition count; default 1.
                            - **device** (*int*) -- CUDA device index; default 0.
                            - **inputMode** (*str*) -- "auto", "gpu", or "python"; selects the fuzzy input parser.
                            - **duplicatePolicy** (*str*) -- "max" keeps the maximum membership for duplicate labels; "error" rejects duplicates.
                            - **allocator** (*str*) -- "fresh" or "reuse"; controls CuPy allocation policy.

    :**Attributes**:    - **_startTime** (*float*) -- Start time of the latest mining call.
                            - **_endTime** (*float*) -- Completion time of the latest mining call.
                            - **_finalPatterns** (*mapping*) -- Patterns and supports retained by the miner.
                            - **_loadingTime** (*float*) -- Input loading time in seconds.
                            - **_transactionCount** (*int*) -- Number of input transactions used by the miner.

    **Execution methods**

    **Terminal command**

    .. code-block:: console

        Format:

        python3 -m cupami.fuzzyFrequentPattern.cuMFFIMiner <inputFile> <outputFile> <minSup> [sep]

        Example Usage:

        python3 -m cupami.fuzzyFrequentPattern.cuMFFIMiner sampleDB.csv patterns.txt 10

    .. note:: Mining requires CuPy and an NVIDIA CUDA GPU. Memory methods report
              process RSS/USS in bytes when psutil is available, otherwise zero.
              getRuntime() reports mining time; getTotalRuntime() adds loading.
              Parquet support, where available, requires PyArrow. Dataframe output
              requires pandas. See docs/algorithm-references.md for source credits
              and differences from the referenced algorithms.

    **Calling from a python program**

    .. code-block:: python

        import cupami as alg

        iFile = 'fuzzy.csv'
        oFile = 'patterns.txt'
        minSup = 10  # absolute support; 0.1 means 10% of transactions

        obj = alg.cuMFFIMiner(iFile=iFile, minSup=minSup)
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
    """
    _grouped = True


if __name__ == '__main__':
    _main(cuMFFIMiner)
