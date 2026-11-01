import math
from collections.abc import Mapping
from decimal import Decimal
from numbers import Integral, Real
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse

from ..frequentPattern.cuApriori import cuApriori
from . import _multi


class _PatternView(Mapping):
    def __init__(self, results):
        self._results = results
        self._index = None

    def __len__(self):
        return len(self._results)

    def items(self):
        for pattern, support in self._results.items():
            yield tuple(f"{item}.{region}" for item, region in pattern), support

    def __iter__(self):
        for pattern, _ in self.items():
            yield pattern

    def __getitem__(self, pattern):
        if self._index is None:
            self._index = dict(self.items())
        return self._index[pattern]


class _MultiMiner(cuApriori):
    _grouped = True

    def __init__(self, iFile, minSup, sep=None, *, supportScale=1,
                 membershipScale=10, repeat=1, device=0, inputMode='auto',
                 duplicatePolicy='max', allocator='fresh'):
        if not isinstance(iFile, (str, Path)):
            raise TypeError('Expected a local fuzzy CSV/text filename')
        if urlparse(str(iFile)).scheme in ('http', 'https'):
            raise ValueError('These miners require a local fuzzy CSV/text file')
        if Path(iFile).suffix.lower() in ('.bin', '.parquet'):
            raise ValueError('These miners accept fuzzy CSV/text, not BIN or Parquet')
        if not Path(iFile).is_file():
            raise FileNotFoundError(iFile)
        if isinstance(minSup, bool) or not isinstance(minSup, Real) or not math.isfinite(minSup) or minSup <= 0:
            raise ValueError('minSup must be a positive finite number')
        if isinstance(supportScale, bool) or not isinstance(supportScale, Real) or not math.isfinite(supportScale) or supportScale <= 0:
            raise ValueError('supportScale must be a positive finite number')
        for name, value, lower, upper in (('repeat', repeat, 1, 2147483647),
                                          ('membershipScale', membershipScale, 1, 32767),
                                          ('device', device, 0, 2147483647)):
            if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
                raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')
        if sep not in (None, ' ', '\t', ','):
            raise ValueError('Supported separators are whitespace, tab and comma')
        if inputMode not in ('auto', 'gpu', 'python'):
            raise ValueError('inputMode must be auto, gpu or python')
        if duplicatePolicy not in ('max', 'error'):
            raise ValueError('duplicatePolicy must be max or error')
        if allocator not in ('fresh', 'reuse'):
            raise ValueError('allocator must be fresh or reuse')
        self.iFile, self.minSup, self.sep = iFile, minSup, sep
        self.supportScale, self.membershipScale = supportScale, int(membershipScale)
        self.repeat, self.device = int(repeat), int(device)
        self.inputMode, self.duplicatePolicy, self.allocator = inputMode, duplicatePolicy, allocator
        self._relativeSupport = supportScale == 1 and not isinstance(minSup, Integral) and minSup <= 1
        self._startTime = self._endTime = self._loadingTime = 0.0
        self._transactionCount = 0
        self._finalPatterns = {}
        self._statistics = {}
        self._inputBackend = {'backend': 'deferred', 'reason': 'File reading and parsing are included in mine()'}

    def _usesGpuInput(self):
        if self.inputMode != 'auto':
            return self.inputMode == 'gpu'
        with Path(self.iFile).open('rb') as stream:
            for line in stream:
                if not line.strip():
                    continue
                labels = line.split(b':', 1)[0].strip()
                return not labels or re.fullmatch(
                    rb'[0-9]+\.[A-Z]{1,6}(?:[ \t\r,]+[0-9]+\.[A-Z]{1,6})*[ \t\r,]*', labels) is not None
        return True

    def mine(self):
        cp = self._getCudaModule()
        with cp.cuda.Device(self.device), cp.cuda.Stream.null:
            cp.cuda.runtime.free(0)
            cp.cuda.Stream.null.synchronize()
            if self.allocator == 'fresh':
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            self._finalPatterns = {}
            self._statistics = {}
            self._startTime = time.perf_counter()
            self._endTime = self._startTime
            threshold = float(self.minSup / self.supportScale)
            use_gpu = self._usesGpuInput()
            if use_gpu:
                results, stats = _multi.mine_file_gpu(
                    self.iFile, threshold, cp, self._grouped, self.repeat,
                    self.membershipScale, self.duplicatePolicy,
                    relative_support=self._relativeSupport)
            else:
                database, transactions = _multi.read_database(
                    self.iFile, repeat=self.repeat, duplicate_policy=self.duplicatePolicy)
                if self._relativeSupport:
                    threshold = Decimal(str(self.minSup)) * transactions
                if not transactions:
                    results, stats = {}, {'levels': [0], 'candidates': 0, 'list_matches': 0}
                else:
                    results, stats = _multi.mine(database, threshold, cp, self._grouped, self.membershipScale)
                stats.update(transactions=transactions,
                             base_items=len({term[0] for term in database}),
                             timing_scope='full file read + parse/packing + mining + result download')
            cp.cuda.Stream.null.synchronize()
            self._endTime = time.perf_counter()
        self._transactionCount = stats['transactions']
        self._statistics = {**stats, 'allocation_policy': self.allocator,
                            'support_scale': self.supportScale,
                            'membership_scale': self.membershipScale,
                            'relative_support': self._relativeSupport,
                            'mode': 'grouped' if self._grouped else 'naive'}
        self._finalPatterns = _PatternView(results)
        self._inputBackend = {'backend': 'cuda' if use_gpu else 'cpu',
                              'reason': 'GPU CSV parsing' if use_gpu else 'Python fuzzy CSV parsing; mining uses CUDA'}

    def getStatistics(self):
        return dict(self._statistics)


def _main(miner_class):
    if len(sys.argv) not in (4, 5):
        sys.exit(f'Usage: python {miner_class.__name__}.py <inputFile> <outputFile> <minSup> [sep]')
    try:
        support = int(sys.argv[3])
    except ValueError:
        support = float(sys.argv[3])
    miner = miner_class(sys.argv[1], support, sep=sys.argv[4] if len(sys.argv) == 5 else None)
    miner.mine()
    miner.save(sys.argv[2])
    miner.printResults()
