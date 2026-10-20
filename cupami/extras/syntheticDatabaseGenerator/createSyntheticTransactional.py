import csv
import random
import shutil
import struct
import tempfile
from itertools import islice
from pathlib import Path


class createSyntheticTransaction:
    def __init__(self, transactions, items, min_length=1, max_length=None, seed=None):
        if transactions < 0:
            raise ValueError("transactions must be non-negative")
        if items < 1:
            raise ValueError("items must be at least 1")

        if max_length is None:
            max_length = items
        if min_length < 1 or max_length < min_length or max_length > items:
            raise ValueError(
                "transaction lengths must satisfy 1 <= min_length <= max_length <= items"
            )

        self.transactions = transactions
        self.items = items
        self.min_length = min_length
        self.max_length = max_length
        self._random = random.Random(seed)
        self._database = None

    def create(self):
        self._database = list(self._generateTransactions())

    def getTransactions(self):
        if self._database is None:
            raise RuntimeError("Call create() before getTransactions()")
        return [transaction.copy() for transaction in self._database]

    def save(self, outputFile):
        """Save the created database as CSV, BIN, or Parquet by extension."""
        if self._database is None:
            raise RuntimeError("Call create() before save()")
        self._writeDatabase(outputFile, iter(self._database))

    def _generateTransactions(self):
        itemIds = range(1, self.items + 1)
        for _ in range(self.transactions):
            length = self._random.randint(self.min_length, self.max_length)
            yield sorted(self._random.sample(itemIds, length))

    def createTransactionalDatabase(self, outputFile):
        """Generate directly to a file without retaining the full database."""
        self._writeDatabase(outputFile, self._generateTransactions())

    def _writeDatabase(self, outputFile, transactions):
        suffix = Path(outputFile).suffix.lower()
        if suffix == ".bin":
            self._writeBinary(outputFile, transactions)
        elif suffix == ".parquet":
            self._writeParquet(outputFile, transactions)
        elif suffix in ("", ".csv", ".txt", ".dat"):
            with Path(outputFile).open("w", newline="", encoding="utf-8") as output:
                csv.writer(output).writerows(transactions)
        else:
            raise ValueError("Use a .csv, .bin, or .parquet output file")

    def _writeBinary(self, outputFile, transactions):
        if self.items > 0xFFFFFFFF:
            raise ValueError("BIN item IDs must fit in uint32")
        # Temporary streams keep memory bounded during direct generation.
        with tempfile.TemporaryFile() as offsets, tempfile.TemporaryFile() as items:
            count = total = 0
            offsets.write(struct.pack("<Q", 0))
            for transaction in transactions:
                items.write(struct.pack(f"<{len(transaction)}I", *transaction))
                count += 1
                total += len(transaction)
                offsets.write(struct.pack("<Q", total))
            offsets.seek(0)
            items.seek(0)
            with Path(outputFile).open("wb") as output:
                output.write(struct.pack("<8sQQQ", b"CUPAMI01", count, total, self.items))
                shutil.copyfileobj(offsets, output)
                shutil.copyfileobj(items, output)

    def _writeParquet(self, outputFile, transactions):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("Parquet export requires pyarrow: pip install pyarrow") from error
        if self.items > 0xFFFFFFFF:
            raise ValueError("Parquet item IDs must fit in uint32")
        schema = pa.schema([pa.field(
            "items", pa.list_(pa.field("item", pa.uint32(), nullable=False)),
            nullable=False,
        )])
        with pq.ParquetWriter(outputFile, schema, compression="snappy") as writer:
            while True:
                batch = list(islice(transactions, 8192))
                if not batch:
                    break
                table = pa.Table.from_arrays(
                    [pa.array(batch, type=schema.field("items").type)], schema=schema
                )
                writer.write_table(table)
