# Spatial frequent patterns

`cuSpatialEclat` follows PAMI's `SpatialECLAT` interface:

```python
import cupami as cuPAMI

miner = cuPAMI.cuSpatialEclat("transactions.csv", "neighbours.csv", minSup=0.01)
miner.mine()
print("Patterns:", len(miner.getPatterns()))
print("Loading (s):", miner.getLoadingTime())
print("Mining (s):", miner.getMiningTime())
print("Total (s):", miner.getTotalRuntime())
```

Both inputs are headerless CSV files (local paths or HTTP(S) URLs). Commas
and tabs are detected automatically; use `sep` for an explicit separator.
BIN, Parquet, and in-memory databases are not supported by this miner.

Transaction CSV:

```text
a,b,c
a,b
b,c
```

Neighbour CSV, with the main item first and its neighbours after it:

```text
a,b
b,a,c
c,b
```

At `minSup=2`, the result is `a:2`, `b:3`, `c:2`, `a b:2`, `b c:2`.
`a c` and `a b c` are excluded by the neighbour rule. Every newly added
item must be a listed neighbour of every preceding item in the pattern.
Use symmetric neighbour lists for ordinary undirected spatial relationships;
the miner uses the lists as supplied, matching PAMI.

An integer `minSup` is a support count; a float in `(0, 1]` is a fraction
of the nonempty transaction count. Repeated items within a row count once.
The usual `getPatterns()`, `getPatternsAsDataFrame()`, `save()` and timing
methods are available. CUDA performs bitset intersections and support
counting; CSV parsing and neighbour pruning run on the CPU.

## Spatiotemporal source data

The [spatiotemporal T10I4D100K CSV](https://u-aizu.ac.jp/~udayrage/datasets/spatioTemporalDatabases/spatiotemporal_T10I4D100K.csv)
contains a timestamp followed by `(x y)` locations. Remove the timestamp
column before passing transactions to either PAMI `SpatialECLAT` or
`cuSpatialEclat`; timestamps are not items and this miner does not impose
a temporal constraint.

As in PAMI, prepare a separate neighbourhood file for the desired distance.
Neighbour labels must exactly match the location labels in the transaction
CSV, including parentheses and spacing. PAMI's `FindNeighboursUsingEuclidean`
can calculate Euclidean neighbours from the original temporal source;
its saved `Point(...)` labels may need formatting to match `(x y)` labels.
Distance is a preparation setting, not a parameter of `SpatialECLAT`.

The implementation follows PAMI's neighbour-constrained search and uses
cuPAMI's CUDA Eclat infrastructure. Paper and code credits are in
[README](../README.md#references).
