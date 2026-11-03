# cuFFIMiner

Find fuzzy frequent patterns after selecting the region with the largest total membership for each base item.

Input is a fuzzy CSV/text file with rows such as `1.L\t2.H:0.8\t0.6`. Labels contain an item and region separated by a dot; membership values range from 0 to 1. Support is the sum of minimum memberships across matching transactions.

`minSup` is an integer absolute fuzzy support or a float fraction in (0, 1]. `sep` selects the item delimiter. Local paths and HTTP(S) URLs are supported.

Mining requires CuPy and an NVIDIA CUDA GPU. Parquet input additionally requires PyArrow.

```python
from cupami import cuFFIMiner

miner = cuFFIMiner("fuzzy.csv", minSup=0.01)
miner.mine()
print(miner.getPatterns())
miner.save("patterns.txt")
```

`getPatterns()` maps tuples of region labels to fuzzy support values. `save()` writes patterns to a text file.

Use `getLoadingTime()`, `getMiningTime()`, and `getTotalRuntime()` for timings in seconds.
