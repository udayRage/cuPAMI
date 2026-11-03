# cuMultiFuzzyBaseline

These implementations accompany the paper “cuMFFP-Miner: GPU-Accelerated Multiple Fuzzy Frequent Pattern Mining via Shared Base-Item Traversal” by Mithun Thangaraj and Rage Uday Kiran, The University of Aizu, Japan.

Multiple fuzzy frequent pattern mining keeps all frequent fuzzy regions of an item, such as `1.L` and `1.H`, available for different patterns. Here, `1` is the base item and `L` or `H` is its fuzzy region. A pattern contains at most one region of each base item. A pattern is frequent when the sum of its minimum membership in each transaction reaches `minSup`.

`cuMultiFuzzyBaseline` is the flat GPU baseline. It processes region combinations as separate traversal jobs, which can repeat transaction matching for regions of the same base item.

Input is a fuzzy CSV/text file with rows such as `1.L\t2.H:0.8\t0.6`. Labels contain an item and region separated by a dot; membership values range from 0 to 1. Support is the sum of minimum memberships across matching transactions.

`minSup` is an integer absolute fuzzy support or a float fraction in (0, 1]. `sep` selects the item delimiter. These miners require a local file. `device` selects the CUDA device, `inputMode` selects `auto`, `gpu`, or `python` parsing, and `membershipScale` controls membership precision (default 10). The default duplicate policy keeps the maximum membership per label.

Mining requires CuPy and an NVIDIA CUDA GPU.

```python
from cupami import cuMultiFuzzyBaseline

miner = cuMultiFuzzyBaseline("fuzzy.csv", minSup=0.01)
miner.mine()
print(miner.getPatterns())
miner.save("patterns.txt")
```

`getPatterns()` maps tuples of region labels to fuzzy support values. `save()` writes patterns to a text file.

Use `getLoadingTime()`, `getMiningTime()`, and `getTotalRuntime()` for timings in seconds.
