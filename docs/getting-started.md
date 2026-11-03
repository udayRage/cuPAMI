# Getting started

## Install

```bash
pip install cupami
```

Mining requires an NVIDIA GPU and a CuPy installation compatible with your CUDA
version. For example, for CUDA 12:

```bash
pip install cupy-cuda12x
```

Parquet conversion requires `pyarrow`; DataFrame results require `pandas`.

## Mine frequent patterns

```python
from cupami import BinaryTransactionalDatabase, cuApriori

generator = BinaryTransactionalDatabase(100, 10, avg_length=3, seed=42)
transactions = generator.create()
miner = cuApriori(transactions, minSup=0.1)
miner.mine()
print(miner.getPatterns())
print("Loading (s):", miner.getLoadingTime())
print("Mining (s):", miner.getMiningTime())
```

## Convert a dataset

```python
from cupami import convertFile

convertFile("transactions.csv", "transactions.bin", dataType="binary")
convertFile("transactions.bin", "transactions.parquet", dataType="binary")
```

Conversion and generation run on the CPU. See [data formats](transaction-formats.md)
for quantitative and temporal datasets, or the [algorithm guides](algorithms/README.md)
for fuzzy, periodic, partial-periodic, and spatial mining.

More examples are available in the
[quickstart notebook](https://github.com/udayRage/cuPAMI/blob/main/quickstart.ipynb).
