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

## Multiple fuzzy regions

```python
miner = cuPAMI.cuMFFIMiner("fuzzy.csv", minSup=0.01)
miner.mine()
patterns = miner.getPatterns()
```

`cuFFIMiner` keeps the strongest region per item. `cuMFFIMiner` keeps every
frequent region. A multiple-region pattern still contains at most one region
of each base item: `{A.L, B.H}` is allowed, while `{A.L, A.H}` is excluded.
`cuMFFIMiner` uses shared base-item traversal (SBT). `cuMultiFuzzyBaseline`
uses flat traversal, processing region combinations separately. Both require a
local CSV/text file; URLs are supported only by `cuFFIMiner`.

An integer `minSup` is absolute fuzzy support. A float in `(0, 1]` is a fraction
of the transaction count; larger floats are absolute fuzzy support.

The multiple-region miners use `membershipScale=10` by default, representing
memberships in tenths; choose a scale matching your input precision. Duplicate
labels keep their maximum membership by default (`duplicatePolicy='max'`).
`inputMode` selects `auto`, `gpu`, or `python` parsing. GPU parsing is selected
automatically for supported numeric base-item labels; Python parsing supports
other labels. File reading and parsing are included in `mine()` timing.
`cuFFIMiner` uses CPU parsing and float64 membership intersections.

See the [SBT guide](algorithms/cuMFFIMiner.md) and
[flat baseline guide](algorithms/cuMultiFuzzyBaseline.md) for examples and paper credits.
