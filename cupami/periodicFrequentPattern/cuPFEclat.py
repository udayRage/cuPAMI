"""Local CUDA vertical prefix search with periodicity pruning.

Uses Eclat's depth-first equivalence classes and intersects timestamp
occurrence bitsets on the GPU. This is a local adaptation, not author code.
"""
import time
from ._temporal import TemporalInput


class cuPFEclat(TemporalInput):
    """Depth-first periodic-frequent mining over vertical occurrence bitsets."""
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
