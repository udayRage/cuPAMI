# cuPPPEclat discovers partial-periodic patterns with depth-first GPU occurrence
# intersections. Periodic support counts consecutive distinct occurrence gaps within the
# period threshold, excluding boundary gaps.
#
# **Importing this algorithm into a python program**
#
#             import cupami as alg
#
#             iFile = 'sampleDB.csv'
#             oFile = 'patterns.txt'
#             minPS = 5
#             period = 10
#
#             obj = alg.cuPPPEclat(iFile=iFile, minPS=minPS, period=period)
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

"""Depth-first CUDA partial-periodic pattern mining.
GPU reference: PAMI gPPMiner, bitset intersections and temporal scans.
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/partialPeriodicPattern/cuda/gPPMiner.py
CPU model/API: PAMI PPP_ECLAT; depth-first prefix extension.
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/partialPeriodicPattern/basic/PPP_ECLAT.py
Local variant uses row bitsets plus timestamps and counts only consecutive
occurrence gaps, excluding boundary gaps. See docs/algorithm-references.md.
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

from ._partial import PartialInput
from ..periodicFrequentPattern.cuPFEclat import cuPFEclat


class cuPPPEclat(PartialInput):
    """
    **About this algorithm**

    :**Description**: cuPPPEclat discovers partial-periodic patterns with depth-first GPU occurrence intersections. Periodic support counts consecutive distinct occurrence gaps within the period threshold, excluding boundary gaps.

    :**Reference**: Periodic-support model and GPU scan references: PAMI PPP_ECLAT and gPPMiner. The implementation counts consecutive distinct occurrence gaps and excludes boundary gaps; see docs/algorithm-references.md.
                    Partial-periodic research paper: https://www.tkl.iis.u-tokyo.ac.jp/new/uploads/publication_file/file/774/JSS_2017.pdf

    :**Parameters**:    - **iFile** (*str or pathlib.Path or iterable*) -- Timestamp-first rows or temporal CSV/BIN/Parquet. Equal timestamps are merged for mining.
                            - **minPS** (*int or float or str*) -- Minimum qualifying consecutive occurrence gaps. Integers are absolute; floats and decimal strings are multiplied by the input transaction count.
                            - **period** (*int or float or str*) -- Largest gap counted toward periodic support. Integers are timestamp units; floats and decimal strings are multiplied by the input transaction count.
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

        python3 -c "import cupami as alg; iFile = 'sampleDB.csv'; oFile = 'patterns.txt'; minPS = 5; period = 10; obj = alg.cuPPPEclat(iFile=iFile, minPS=minPS, period=period); obj.mine(); obj.save(oFile)"

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
        minPS = 5
        period = 10

        obj = alg.cuPPPEclat(iFile=iFile, minPS=minPS, period=period)
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

    Mine with (iFile, minPS, period, sep=None); return periodic supports.
    """
    _BATCH_SIZE = 4096
    mine = cuPFEclat.mine
