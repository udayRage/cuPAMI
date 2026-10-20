"""Local CUDA Apriori adaptation with periodicity pruning; see references.md."""
import time
from ._temporal import TemporalInput


class cuPFApriori(TemporalInput):
    """Level-wise periodic-frequent mining; GPU support and maximum gaps."""
    _BATCH_SIZE = 4096

    def mine(self):
        cp = self._getCudaModule()
        self._startTime = time.perf_counter()
        labels, _, _ = self._prepare(cp)
        candidates = [(label,) for label in labels]
        while candidates:
            level = []
            for start in range(0, len(candidates), self._BATCH_SIZE):
                batch = candidates[start:start + self._BATCH_SIZE]
                for pattern, (support, period) in zip(batch, self._evaluate(cp, batch)):
                    if self._accept(pattern, support, period):
                        level.append(pattern)
            candidates = self._generateCandidates(level)
        cp.cuda.get_current_stream().synchronize()
        self._endTime = time.perf_counter()
