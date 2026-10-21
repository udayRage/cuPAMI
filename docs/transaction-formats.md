# Data generation and file formats

See [algorithm and implementation references](../README.md#references) for the original
Apriori paper, adapted CUDA code, and the official GDS/KvikIO sources.

## Convert an existing dataset

Text datasets can contain tabs or commas regardless of their `.csv` extension.
Both layouts use one transaction per line without a header. cuApriori detects
tabs or commas automatically; set `sep` explicitly for other single-character
delimiters.

```python
from cupami import convertFile

url = "https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/Transactional_T10I4D100K.csv"
convertFile(url, "transactions.bin", dataType="transactional")
convertFile("transactions.bin", "transactions.parquet", dataType="transactional")
convertFile("transactions.parquet", "transactions.csv", dataType="transactional")
```

The converter accepts local CSV/BIN/Parquet files, text URLs, or transaction
iterables. It uses the same readers and writers as cuApriori and the generator.
The shared entry point lives in `cupami/extras/fileConverter.py`; `dataType`
selects the database layout and the output extension selects the storage format.
`dataType="transactional"` is the default; use `dataType="temporal"` for
timestamped data. `convertTransactions` remains a transactional shorthand.
It requires no GPU and retains the database in CPU memory. Parquet requires
PyArrow. Item IDs must be nonnegative uint32 integers (zero is preserved); conversion normalizes their
numeric spelling and removes duplicate IDs within a basket. Transactional
conversion omits empty baskets; temporal conversion preserves them and retains
separate rows with repeated timestamps, sorted by timestamp.
CSV output uses commas. The returned dictionary reports transaction and item
occurrence counts. The existing dataset is converted; no new data is generated.

## Generate synthetic transactions

Use the public package import and pass generated data directly to a miner:

```python
import cupami as cuPAMI

generator = cuPAMI.TransactionalDatabase(
    transactions=10000,
    items=50,
    min_length=2,
    max_length=10,
    seed=42,
)
data = generator.create()

miner = cuPAMI.cuApriori(data, minSup=0.01)
miner.mine()
patterns = miner.getPatterns()
```

`create()` returns the transactions and keeps a copy in the generator for
optional export. `cuEclat` and `cuFPGrowth` accept the same data.

### Optional export

After the example above, save the same generated transactions in any format:

```python
generator.save("transactions.csv")
generator.save("transactions.bin")
generator.save("transactions.parquet")
```

The file extension selects the format. Parquet export requires PyArrow;
CSV and BIN export use the Python standard library. Each `save()` exports
the data from the most recent `create()` call without generating new data.

Generated transactions contain sorted, unique item IDs in `1..items`. Duplicate baskets across different transactions are allowed. Zero transactions produces a valid empty file/database in each format. BIN and Parquet support item IDs up to `2**32 - 1`.

## Generate temporal data

```python
import cupami as cuPAMI

data = cuPAMI.TemporalDatabase(
    transactions=10000,
    items=50,
    min_length=2,
    max_length=10,
    seed=42,
).create()

miner = cuPAMI.cuPFEclat(data, minSup=0.02, maxPer=100)
miner.mine()
patterns = miner.getPatterns()
```

Use `cuPFApriori` or `cuPFGrowth` with the same input and parameters.
Temporal results map each pattern to `(support, periodicity)`. Integer `maxPer`
is measured in timestamp units; a float in `(0, 1]` is a fraction of the
observation window. Fractional `minSup` uses the number of records, rounded up.

`TemporalDatabase` accepts `repeatChance` (0–1) to repeat the previous
timestamp and `mergeTimestamps=True` to combine rows sharing a timestamp.
Merging keeps unique items and may reduce the number of output rows.

Timestamps must be nonnegative int64 integers. Rows are sorted by timestamp;
repeated timestamps remain separate transactions, and empty baskets retain
their timestamp. Periodicity includes gaps from 0 to the first occurrence
and from the last occurrence to the largest input timestamp. Set `endTime`
to extend the observation window.

Temporal miners accept iterable rows, timestamp-first CSV/text, temporal BIN,
temporal Parquet, and text URLs. Local file formats are selected by extension.
File parsing uses the CPU and mining uses CUDA; temporal GDS is not implemented.
BIN and Parquet preserve repeated timestamps and empty baskets.

Keep the generator to export the same data:

```python
generator = cuPAMI.TemporalDatabase(
    transactions=10000,
    items=50,
    min_length=2,
    max_length=10,
    repeatChance=0.2,
    mergeTimestamps=False,
)
data = generator.create()
generator.save("temporal.csv")
generator.save("temporal.bin")
generator.save("temporal.parquet")

miner = cuPAMI.cuPFEclat("temporal.bin", minSup=0.02, maxPer=100)
miner.mine()
```

The same files work with `cuPFApriori` and `cuPFGrowth`.
To convert an existing timestamp-first dataset:

```python
cuPAMI.convertFile("temporal.csv", "temporal.bin", dataType="temporal")
cuPAMI.convertFile("temporal.bin", "temporal.parquet", dataType="temporal")
```

## CSV

UTF-8, no header, one transaction per row, comma separated integer item IDs. Rows can have different lengths. This is a basket layout, not a one-hot matrix.

```text
2,5,9
1,5
2,9
```

The existing `.txt`, `.dat`, and extensionless exports also retain this CSV layout.

## Transactional BIN format, version 1

This is a cuPAMI format, not a universal `.bin` standard or IBM's binary format. All integers are unsigned and little endian. There is no padding or compression.

| Byte position | Type | Meaning |
|---|---|---|
| 0 | 8 bytes | ASCII `CUPAMI01`, identifying format and version |
| 8 | uint64 | Transaction count N |
| 16 | uint64 | Total item occurrences M |
| 24 | uint64 | Item ID upper bound U (configured item count for synthetic data) |
| 32 | uint64[N + 1] | Transaction offsets measured in item entries |
| 32 + 8 × (N + 1) | uint32[M] | Concatenated item IDs |

Transaction i is `items[offsets[i]:offsets[i + 1]]` for zero based i. Offsets start at zero, are nondecreasing, and end at M. Item IDs are in `0..U`; the synthetic generator uses `1..U`, while converted external datasets may also contain zero. U is at least 1 and can include IDs absent from the database. File size is exactly `32 + 8*(N+1) + 4*M` bytes. An empty database has one zero offset and no item entries.

For the CSV example above:

```text
N = 3, M = 7
Offsets = [0, 3, 5, 7]
Items   = [2, 5, 9, 1, 5, 2, 9]
```

Contiguous offsets and item arrays avoid padding variable length baskets. This is the storage representation; mining algorithms may convert it to another GPU representation. Loading performance depends on the dataset, reader, and storage device.

## Transactional Parquet

One row per transaction. The only column is `items`, a non-null list of non-null uint32 item IDs. Lists are stored as native Parquet lists, not strings such as `"[2,5,9]"`. Files use Snappy compression. Direct generation writes batches of up to 8,192 transactions.

```python
import pyarrow.parquet as pq
transactions = pq.read_table("transactions.parquet")["items"].to_pylist()
```

## Temporal BIN format, version 1

Little-endian header and arrays, with timestamps stored separately:

| Byte position | Type | Meaning |
|---|---|---|
| 0 | 8 bytes | ASCII `CUPAMIT1` |
| 8 | uint64 | Number of rows N |
| 16 | uint64 | Number of item occurrences M |
| 24 | uint64 | Upper item ID bound U |
| 32 | int64[N] | Nonnegative timestamps |
| 32 + 8*N | uint64[N + 1] | Basket offsets |
| 32 + 8*N + 8*(N + 1) | uint32[M] | Item IDs |

Offsets start at zero, never decrease, and end at M. IDs are in `0..U`,
where U is at least 1 and fits uint32. The exact size is
`32 + 8*N + 8*(N + 1) + 4*M` bytes. Equal adjacent offsets represent empty
baskets. Duplicate timestamps are retained. Readers validate the header,
array sizes, offsets, timestamps, and item bounds before mining.
The distinct header prevents transactional BIN files being read as temporal data.

## Temporal Parquet

Two non-null columns: `timestamp: int64` and `items: list<uint32>` with
non-null members. Timestamps are nonnegative. Empty lists and repeated timestamps
are retained. Files use Snappy compression and generation writes batches of
up to 8192 rows. PyArrow is required for reading and writing these files.

## Current scope

Both generators export CSV, BIN, and Parquet. Transactional files store baskets;
temporal files also store a timestamp per basket. Use the matching miner for
each dataset type. Temporal Parquet requires PyArrow.

`cuApriori` reads the generator's CSV, BIN, and Parquet files directly, selecting the reader by file extension (case insensitive). Parquet CPU input requires `pyarrow`; GPU input uses cuDF. Delimited text URLs and transaction iterables are also supported; URL input remains text only.
`cuEclat` shares these readers and constructor options, including GPU loading
and GDS controls. Its prefix-class intersection search replaces Apriori's
level-wise candidate generation; output patterns retain the same interface.
`cuFPGrowth` also shares the readers and output methods. It prepares an FP-tree
instead of bitsets, with conditional trees built during mining.

```python
from cupami import cuApriori

miner = cuApriori("transactions.parquet", minSup=0.1)  # also .csv or .bin
miner.mine()  # requires CuPy and a CUDA device
patterns = miner.getPatterns()
```

## GPU loading and mining

CSV and Parquet input default to `use_gpu_io="auto"`. When cuDF and CUDA are
available, the loader parses and expands transactions on the GPU using the
load/split/explode approach from
[cuda_pami](https://github.com/udayRage/cuda_pami/blob/main/src/algorithms/frequent/fpminer-bitset.py).
The Parquet loader accepts this generator's `items: list<uint32>` schema.
CSV GPU loading accepts unquoted nonnegative integer baskets; other CSV
layouts retain the general CPU reader. Both paths remove repeated items within
a transaction and omit empty baskets.

BIN input uses NumPy bulk array reads and transfers the numeric arrays to CUDA
when direct GDS is unavailable. This `numpy-cuda` backend avoids constructing
Python transactions and converting each occurrence into a string. Offsets and
item IDs are validated before transfer; distinct labels alone are formatted
for pattern output. GDS remains the preferred BIN path when available.

Use `use_gpu_io=False` to force CPU parsing (and `use_gds=False` to disable BIN
GDS), or `use_gpu_io=True` to require GPU loading. Automatic mode falls back
if the GPU dependencies are unavailable.
`getInputBackend()` identifies the selected reader. GPU parsing does not itself
prove that GDS was used. `use_gds=False` disables KvikIO's direct cuFile path
for cuDF reads as well as disabling the BIN GDS reader.

`cuApriori` and `cuEclat` use packed uint32 bitsets and CUDA AND/popcount support counting,
adapted from cuda_pami's bitset miner. Each item uses one bit per transaction,
rather than a byte per transaction. `cuApriori` candidate generation retains subset
pruning. Support counting avoids materializing transaction-by-candidate-by-item
temporary arrays. CPU-loaded transactions are transferred to the GPU in bulk.

For the local `jhub` environment, disable user packages to avoid NumPy
version conflicts:

```bash
PYTHONNOUSERSITE=1 conda run -n jhub python your_script.py
```

## Optional GDS input

Local BIN input defaults to `use_gds="auto"`. When KvikIO, CUDA, and
`kvikio.cufile_driver.get("is_gds_available")` are available, cuApriori reads
the BIN file into GPU memory through cuFile and builds transaction bitsets
on the GPU. Only the header and distinct item labels need CPU processing.

```python
miner = cuApriori("transactions.bin", 0.02, use_gds="auto")
print(miner.getInputBackend())
miner.mine()
```

Use `use_gds=False` to disable the direct cuFile reader; bulk NumPy reads
and CUDA transfers remain available. Set `use_gpu_io=False` as well to
force CPU transaction parsing. Use `use_gds=True` to require
the cuFile path and raise an error when GDS is reported unavailable or the
read fails. Automatic mode falls back to bulk numeric reads and GPU transfers
when CUDA is available, or to the CPU transaction reader otherwise.
`getInputBackend()` records the selected path and fallback reason.

The `cufile` backend identifies the API used, **not proof of direct DMA**.
KvikIO's POSIX fallback and small-read threshold are disabled for that path,
but cuFile itself may still use CPU compatibility I/O if its configuration
allows it. For a strict deployment check, configure a separate process with
`CUFILE_ENV_PATH_JSON` pointing to a cuFile configuration with
`properties.allow_compat_mode=false`, and run with `use_gds=True`.
Use NVIDIA's `gdscheck -p` and storage-specific diagnostics as well.
The library does not change system driver or cuFile configuration.

On the inspected host, `nvidia_fs` is loaded. Strict direct reads were verified
on `/tmp`'s NVMe/ext4 filesystem after `data=ordered` became visible in the
mount options. The workspace's `/home` RAID10 volume (`/dev/md0`) failed strict
cuFile registration. Keep datasets on the verified NVMe filesystem for direct
GDS experiments. Driver availability alone does not prove that a particular
file used direct DMA.
