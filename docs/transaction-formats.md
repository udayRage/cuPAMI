# Synthetic transactional data

Generate a database once, then save the same transactions in three formats:

```python
from cupami.extras.syntheticDatabaseGenerator import createSyntheticTransaction

generator = createSyntheticTransaction(
    transactions=1000, items=100, min_length=2, max_length=10, seed=42
)
generator.create()
generator.save("transactions.csv")
generator.save("transactions.bin")
generator.save("transactions.parquet")
```

Parquet export requires `pip install pyarrow`. CSV and BIN use the Python standard library.

For direct generation without retaining the database in memory:

```python
generator.createTransactionalDatabase("transactions.parquet")
```

The file extension selects the format (case insensitive). Direct generation advances the random generator; use separate generators with identical parameters and seeds to reproduce the same transactions across formats. `save()` reuses the database from `create()`.

Generated transactions contain sorted, unique item IDs in `1..items`. Duplicate baskets across different transactions are allowed. Zero transactions produces a valid empty file/database in each format. BIN and Parquet support item IDs up to `2**32 - 1`.

## CSV

UTF-8, no header, one transaction per row, comma separated integer item IDs. Rows can have different lengths. This is a basket layout, not a one-hot matrix.

```text
2,5,9
1,5
2,9
```

The existing `.txt`, `.dat`, and extensionless exports also retain this CSV layout.

## cuPAMI binary transaction format, version 1

This is a cuPAMI format, not a universal `.bin` standard or IBM's binary format. All integers are unsigned and little endian. There is no padding or compression.

| Byte position | Type | Meaning |
|---|---|---|
| 0 | 8 bytes | ASCII `CUPAMI01`, identifying format and version |
| 8 | uint64 | Transaction count N |
| 16 | uint64 | Total item occurrences M |
| 24 | uint64 | Configured item universe size U |
| 32 | uint64[N + 1] | Transaction offsets measured in item entries |
| 32 + 8 × (N + 1) | uint32[M] | Concatenated item IDs |

Transaction i is `items[offsets[i]:offsets[i + 1]]` for zero based i. Offsets start at zero, are nondecreasing, and end at M. Item IDs are in `1..U`; U can include IDs absent from the generated database. File size is exactly `32 + 8*(N+1) + 4*M` bytes. An empty database has one zero offset and no item entries.

For the CSV example above:

```text
N = 3, M = 7
Offsets = [0, 3, 5, 7]
Items   = [2, 5, 9, 1, 5, 2, 9]
```

Contiguous offsets and item arrays avoid padding variable length baskets. This is the storage representation; mining algorithms may convert it to another GPU representation. No performance advantage has been benchmarked yet.

## Parquet

One row per transaction. The only column is `items`, a non-null list of non-null uint32 item IDs. Lists are stored as native Parquet lists, not strings such as `"[2,5,9]"`. Files use Snappy compression. Direct generation writes batches of up to 8,192 transactions.

```python
import pyarrow.parquet as pq
transactions = pq.read_table("transactions.parquet")["items"].to_pylist()
```

## Current scope

These exports are for the transactional generator. The temporal generator retains CSV output, with timestamps in its first field; BIN and Parquet exports reject temporal data until their timestamp schemas are defined.

`cuApriori` currently reads delimited text, URLs pointing to delimited text, or transaction iterables. Direct BIN and Parquet mining input is not implemented yet. The README diagram shows the intended inputs.
