# Fuzzy CSV input

```python
import cupami as cuPAMI

miner = cuPAMI.cuFFIMiner("fuzzy.csv", minSup=0.01)
miner.mine()
patterns = miner.getPatterns()
```

CSV files and text URLs contain one transformed transaction per line, without
a header. The colon separates labels from their membership values:

```text
A.L,A.H,B.H:0.4,0.6,1
A.L,A.H,B.H:0.7,0.3,0.5
```

Tabs, commas, and whitespace are detected; `sep` selects another delimiter.
Labels must be unique per row, and membership counts must match. Memberships
are finite values in [0, 1]. Missing labels have membership zero. Blank lines
are ignored, while `:` counts as an empty transaction.

The suffix after the last dot identifies a fuzzy region. The region with the
largest total membership is retained for each base item; ties use lexical order.
In this example `A.L` totals 1.1 and `A.H` totals 0.9, so `A.L` is retained.
Pattern support sums the minimum membership per transaction. For `{A.L, B.H}`,
support is `min(0.4, 1) + min(0.7, 0.5) = 0.9`.

An integer `minSup` is absolute support. A float in `(0, 1]` is a fraction of
the transaction count, without rounding; a larger float is absolute fuzzy
support. Float64 reductions use threshold tolerance
`1e-12 * max(1, absolute_minimum_support)`.

Parsing uses the CPU; CUDA handles support reductions and minimum-membership
intersections, with depth-first scheduling on the CPU. Initial GPU storage is
8 bytes per transaction per retained frequent region. Intersection batches
target 64 MiB, with additional memory for retained search classes.

The miner shares pattern output, saving, memory reporting, and the separate
loading/mining/total timing methods with the other miners. BIN, Parquet,
quantitative-to-fuzzy conversion, and fuzzy generation are outside this version.
Paper and PAMI source credits are in the [README](../README.md#references).
