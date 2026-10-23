import csv
import math
import numbers
import time
import random
import shutil
import struct
import tempfile
from contextlib import contextmanager
from itertools import islice
from pathlib import Path


class TransactionalDatabase:
    """Generate unique-item baskets with a configurable expected average length.

    Lengths are sampled symmetrically around avg_length and stochastically
    rounded to integers in [1, items]. The realized mean can vary. Omitting
    avg_length uses (items + 1) / 2. Measurements describe the latest create()
    or direct-to-file generation; save() does not replace them.
    """
    def __init__(self, transactions, items, avg_length=None, seed=None):
        if isinstance(transactions, bool) or not isinstance(transactions, int):
            raise TypeError("transactions must be an integer")
        if isinstance(items, bool) or not isinstance(items, int):
            raise TypeError("items must be an integer")
        if transactions < 0:
            raise ValueError("transactions must be non-negative")
        if items < 1:
            raise ValueError("items must be at least 1")
        if avg_length is None:
            avg_length = (items + 1) / 2
        if isinstance(avg_length, bool) or not isinstance(avg_length, numbers.Real):
            raise TypeError("avg_length must be a number")
        if not math.isfinite(avg_length) or not 1 <= avg_length <= items:
            raise ValueError("avg_length must satisfy 1 <= avg_length <= items")

        self.transactions = transactions
        self.items = items
        self.avg_length = float(avg_length)
        self._random = random.Random(seed)
        self._database = None
        self._runtime = 0.0
        self._memoryRSS = 0
        self._memoryUSS = 0

    @contextmanager
    def _measureGeneration(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._runtime = time.perf_counter() - start
            self._memoryRSS = self._memoryUSS = 0
            try:
                import psutil
            except ImportError:
                pass
            else:
                process = psutil.Process()
                try:
                    self._memoryRSS = int(process.memory_info().rss)
                    self._memoryUSS = int(process.memory_full_info().uss)
                except (psutil.AccessDenied, AttributeError):
                    pass

    def getRuntime(self):
        """Elapsed seconds for the most recent generation (including file writes)."""
        return self._runtime

    def getMemoryRSS(self):
        """Process RSS in bytes after generation; 0 when psutil is unavailable."""
        return self._memoryRSS

    def getMemoryUSS(self):
        """Process USS in bytes after generation; 0 when unavailable."""
        return self._memoryUSS

    def create(self):
        """Generate and return transactions ready to pass to a miner."""
        with self._measureGeneration():
            self._database = list(self._generateTransactions())
            result = self.getTransactions()
        return result

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
        radius = min(self.avg_length - 1, self.items - self.avg_length)
        for _ in range(self.transactions):
            # A symmetric uniform draw has the requested mean. Stochastic
            # rounding preserves that mean for fractional average lengths.
            target = self._random.uniform(self.avg_length - radius,
                                          self.avg_length + radius)
            length = math.floor(target)
            if self._random.random() < target - length:
                length += 1
            yield sorted(self._random.sample(itemIds, length))

    def createTransactionalDatabase(self, outputFile):
        """Generate directly to a file without retaining the full database."""
        with self._measureGeneration():
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
