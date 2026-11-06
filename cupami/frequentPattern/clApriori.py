# clApriori discovers frequent patterns on OpenCL GPUs, including supported AMD GPUs.
#
# **Importing this algorithm into a python program**
#
#             import cupami as alg
#
#             obj = alg.clApriori('transactions.csv', minSup=10, vendor='AMD')
#             obj.mine()
#             print('Total number of patterns:', len(obj.getPatterns()))
#             obj.save('patterns.txt')
#             Df = obj.getPatternsAsDataFrame()
#             print('Memory USS:', obj.getMemoryUSS())
#             print('Memory RSS:', obj.getMemoryRSS())
#             print('Mining time in seconds:', obj.getRuntime())
#
"""Experimental OpenCL Apriori for AMD and other OpenCL GPU implementations.

Uses cuPAMI's CPU readers and Apriori pruning with OpenCL packed-bitset kernels.
No CuPy, CUDA, NVIDIA GDS, or OpenCL CPU fallback is used by this miner.
AMD performance has not been validated on this project's NVIDIA test machine.
OpenCL API reference: https://documen.tician.de/pyopencl/
Source and paper mapping: docs/algorithm-references.md.
"""

__copyright__ = """
Copyright (C) 2026 Rage Uday Kiran

This program is free software: you can redistribute it and/or modify it under
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
"""

import math
from numbers import Integral, Real
from pathlib import Path
import sys
import time

import numpy as np

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'cupami.frequentPattern'

from .cuApriori import cuApriori
from ._opencl_bitsets import OpenCLBitsets, list_devices


class clApriori(cuApriori):
    """
    **About this algorithm**

    :**Description**: Apriori with OpenCL GPU bitset construction and support
        counting. One work-group evaluates each candidate using contiguous word
        access, popcount, and a local-memory sum reduction. Work-group size is
        chosen from the device/kernel limits, preferred execution multiple and
        bitset length, with no fixed NVIDIA warp-size assumption.

    :**Reference**: Agrawal, R., Srikant, R.: Fast Algorithms for Mining Association
        Rules. VLDB 1994, pp. 487–499. https://www.vldb.org/conf/1994/P487.PDF
        GPU support-counting design: Zhang, F., Zhang, Y., Bakos, J.: GPApriori:
        GPU-Accelerated Frequent Itemset Mining. CLUSTER 2011, pp. 590–594.
        https://doi.org/10.1109/CLUSTER.2011.61
        This is a local OpenCL adaptation, not an AMD implementation supplied by
        those papers. AMD hardware performance is not yet measured.

    :**Parameters**: - **iFile** (*str, pathlib.Path or iterable*) -- Transactions,
                       CSV, BIN, Parquet, or a CSV HTTP(S) URL; read on the CPU.
                     - **minSup** (*int or float*) -- Positive absolute support
                       or float fraction in (0, 1]; thresholds round upward.
                     - **sep** (*str or None*) -- CSV delimiter; None detects tabs
                       or commas, matching the cuPAMI transactional reader.
                     - **device** (*int*) -- Index in listDevices(vendor), default 0.
                     - **vendor** (*str or None*) -- Optional vendor filter, e.g.
                       'AMD'. None prefers AMD GPUs, then other OpenCL GPUs.
                     - **workGroupSize** (*int or None*) -- Optional power-of-two
                       group size up to the selected kernel/device limit (max 256).

    :**Attributes**: - **_Database** (*list*) -- CPU-parsed transactions.
                     - **_finalPatterns** (*dict*) -- Item tuples and supports.
                     - **_backend** (*OpenCLBitsets or None*) -- Lazy GPU context.
                     - **_startTime**, **_endTime** (*float*) -- Mining timings.

    **Execution methods**

    **Terminal command**

    .. code-block:: console

        python3 -m cupami.frequentPattern.clApriori transactions.csv patterns.txt 10 --vendor AMD

    **Calling from a python program**

    .. code-block:: python

        from cupami import clApriori

        print(clApriori.listDevices())
        obj = clApriori('transactions.csv', minSup=10, vendor='AMD')
        obj.mine()
        print(obj.getPatterns())
        print(obj.getComputeBackend())
        obj.save('patterns.txt')
        print(obj.getRuntime())

    .. note:: Requires PyOpenCL and a vendor OpenCL GPU runtime supporting
              OpenCL C 1.2. AMD compatibility depends on the GPU and driver.
              NVIDIA OpenCL can also run these kernels. Only GPU devices are
              selected. No automatic CPU fallback or CUDA dependency is used.
              First-call context creation and compilation are included in mining
              time. Pandas and PyArrow are needed for dataframe output and
              Parquet input, respectively.

    **Credits**

    OpenCL port by Mithun Thangaraj, supervised by Professor Rage Uday Kiran.
    CUDA bitset source: udayRage/cuda_pami, revision 2ed352a, GPL-3.0-or-later.
    Existing cuPAMI reader and search credits are retained in their modules.
    """

    _BATCH_SIZE = 4096

    def __init__(self, iFile, minSup, sep=None, *, device=0, vendor=None, workGroupSize=None):
        if isinstance(device, bool) or not isinstance(device, Integral) or device < 0:
            raise ValueError('device must be a nonnegative integer')
        if vendor is not None and (not isinstance(vendor, str) or not vendor.strip()):
            raise ValueError('vendor must be a nonempty string or None')
        if workGroupSize is not None and (isinstance(workGroupSize, bool) or
                not isinstance(workGroupSize, Integral) or workGroupSize < 1 or
                workGroupSize > 256 or workGroupSize & (workGroupSize - 1)):
            raise ValueError('workGroupSize must be a power of two from 1 to 256, or None')
        self.device = int(device)
        self.vendor = vendor.strip() if vendor is not None else None
        self.workGroupSize = int(workGroupSize) if workGroupSize is not None else None
        self._backend = None
        # Both flags prevent the inherited readers from attempting CUDA or GDS.
        super().__init__(iFile, minSup, sep, use_gds=False, use_gpu_io=False)
        self._inputBackend = {'backend': 'cpu', 'reason': 'CPU parsing; OpenCL GPU mining'}

    @staticmethod
    def _getSupportCount(minSup, transactionCount):
        if isinstance(minSup, bool) or not isinstance(minSup, Real) or not math.isfinite(minSup) or minSup <= 0:
            raise ValueError('minSup must be a positive finite number')
        if not isinstance(minSup, Integral) and minSup <= 1:
            return math.ceil(minSup * transactionCount)
        return math.ceil(minSup)

    @staticmethod
    def listDevices(vendor=None):
        """List OpenCL GPUs without creating a context or using CUDA."""
        if vendor is not None and (not isinstance(vendor, str) or not vendor.strip()):
            raise ValueError('vendor must be a nonempty string or None')
        return list_devices(vendor.strip() if vendor is not None else None)

    def getComputeBackend(self):
        """Report the actual OpenCL GPU and selected work-group size."""
        return self._backend.metadata() if self._backend is not None else {'backend': 'opencl', 'device': None}

    def mine(self):
        """Build bitsets on the OpenCL GPU and evaluate Apriori candidate levels."""
        self._startTime = time.perf_counter()
        self._finalPatterns = {}
        try:
            if self._backend is None:
                self._backend = OpenCLBitsets(self.device, self.vendor, self.workGroupSize)
            backend = self._backend
            labels = sorted({item for basket in self._Database for item in basket})
            if not labels:
                return
            indexes = {label: i for i, label in enumerate(labels)}
            rows, columns = [], []
            for row, basket in enumerate(self._Database):
                rows.extend([row] * len(basket))
                columns.extend(indexes[item] for item in basket)
            backend.build(np.asarray(rows, dtype=np.uint64), np.asarray(columns, dtype=np.uint32),
                          self._transactionCount, len(labels))
            candidates = [(label,) for label in labels]
            while candidates:
                frequent = []
                for start in range(0, len(candidates), self._BATCH_SIZE):
                    batch = candidates[start:start + self._BATCH_SIZE]
                    supports = backend.count([[indexes[item] for item in pattern] for pattern in batch])
                    for pattern, support in zip(batch, supports):
                        if support > 0 and support >= self._minSup:
                            self._finalPatterns[pattern] = int(support)
                            frequent.append(pattern)
                candidates = self._generateCandidates(frequent)
        finally:
            if self._backend is not None:
                self._backend.clear()
            self._endTime = time.perf_counter()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mine frequent patterns on an OpenCL GPU')
    parser.add_argument('inputFile')
    parser.add_argument('outputFile')
    parser.add_argument('minSup', type=float)
    parser.add_argument('--sep', default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--vendor', default=None)
    args = parser.parse_args()
    support = int(args.minSup) if args.minSup.is_integer() else args.minSup
    miner = clApriori(args.inputFile, support, args.sep, device=args.device, vendor=args.vendor)
    miner.mine()
    miner.save(args.outputFile)
    miner.printResults()
    print('OpenCL GPU:', miner.getComputeBackend())
