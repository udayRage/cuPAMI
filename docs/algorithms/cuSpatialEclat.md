cuSpatialEclat

Find frequent patterns constrained by item neighbourhoods. Neighbour compatibility is checked on the CPU and support is counted on the GPU.

`iFile` is an item-only transaction CSV. `nFile` is a neighbour CSV where each row starts with an item followed by its neighbours. Both accept local paths or HTTP(S) URLs. Neighbours are used as listed.

`minSup` is an integer count or a float fraction in (0, 1]. `sep` sets the delimiter for both files; otherwise tabs or commas are detected separately.

Mining requires CuPy and an NVIDIA CUDA GPU. Parquet input additionally requires PyArrow.

```python
from cupami import cuSpatialEclat

miner = cuSpatialEclat("transactions.csv", "neighbours.csv", minSup=0.01)
miner.mine()
print(miner.getPatterns())
miner.save("patterns.txt")
```

`getPatterns()` maps item tuples to support counts. `save()` writes patterns to a text file.

Use `getLoadingTime()`, `getMiningTime()`, and `getTotalRuntime()` for timings in seconds.
