cuPPPEclat

Find partial-periodic patterns using depth-first search and GPU intersections.

Input is timestamp-first transaction rows, a CSV, or a temporal BIN or Parquet file. Rows with the same timestamp are merged before mining.

`period` is the largest gap counted toward periodic support. `minPS` is the minimum number of qualifying consecutive occurrence gaps. Integer values are absolute; float values for either parameter are multiplied by the transaction count.

Mining requires CuPy and an NVIDIA CUDA GPU. Parquet input additionally requires PyArrow.

```python
from cupami import cuPPPEclat

miner = cuPPPEclat([[1, "a", "b"], [2, "a"], [3, "a", "b"]], minPS=1, period=2)
miner.mine()
print(miner.getPatterns())
miner.save("patterns.txt")
```

`getPatterns()` maps item tuples to periodic support counts. `save()` writes patterns to a text file.

Use `getLoadingTime()`, `getMiningTime()`, and `getTotalRuntime()` for timings in seconds.
