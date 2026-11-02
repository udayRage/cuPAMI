cuPFEclat

Find periodic frequent patterns using depth-first search and GPU intersections.

Input rows start with a timestamp followed by items. Use an iterable of rows, a timestamp-first CSV, or a temporal BIN or Parquet file.

`minSup` is an integer support count or a float fraction in (0, 1]. `maxPer` bounds gaps between occurrences, including the start and end gaps. A float `maxPer` in (0, 1] is a fraction of the timeline. `endTime` optionally sets the timeline end.

Mining requires CuPy and an NVIDIA CUDA GPU. Parquet input additionally requires PyArrow.

```python
from cupami import cuPFEclat

miner = cuPFEclat([[1, "a", "b"], [2, "a"], [3, "a", "b"]], minSup=2, maxPer=2)
miner.mine()
print(miner.getPatterns())
miner.save("patterns.txt")
```

`getPatterns()` maps item tuples to `(support, periodicity)` values. `save()` writes patterns to a text file.

Use `getLoadingTime()`, `getMiningTime()`, and `getTotalRuntime()` for timings in seconds.
