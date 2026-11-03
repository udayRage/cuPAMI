# cuFPGrowth

Find frequent itemsets using an FP-tree and GPU preparation of conditional trees.

Input can be an iterable of transactions or a CSV, BIN, or Parquet file. Each transaction contains item identifiers. Duplicate items within a transaction count once.

`minSup` is an absolute count when given as an integer; a float in (0, 1] is a fraction of transactions. `sep` selects the CSV delimiter.

Mining requires CuPy and an NVIDIA CUDA GPU. Parquet input additionally requires PyArrow.

```python
from cupami import cuFPGrowth

miner = cuFPGrowth([["a", "b"], ["a", "c"], ["a", "b", "c"]], minSup=2)
miner.mine()
print(miner.getPatterns())
miner.save("patterns.txt")
```

`getPatterns()` returns a mapping from item tuples to support counts. `save()` writes patterns to a text file.

Use `getLoadingTime()`, `getMiningTime()`, and `getTotalRuntime()` for timings in seconds.
