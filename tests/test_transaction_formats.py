import struct
import tempfile
import unittest
from collections import Counter
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from cupami import TransactionalDatabase, cuApriori


def expected_patterns(rows, threshold):
    supports = {}
    for row in rows:
        items = sorted({str(item).strip() for item in row if str(item).strip()})
        for size in range(1, len(items) + 1):
            for pattern in combinations(items, size):
                key = "\t".join(pattern)
                supports[key] = supports.get(key, 0) + 1
    return {key: count for key, count in supports.items() if count >= threshold}


class TransactionFormatsTest(unittest.TestCase):
    def test_requested_10000_transaction_dataset(self):
        try:
            cuApriori._getCudaModule()
        except (ImportError, RuntimeError) as error:
            self.skipTest(str(error))
        generator = TransactionalDatabase(
            10000, 50, min_length=2, max_length=10, seed=42
        )
        generator.create()
        rows = generator.getTransactions()
        singles = Counter(str(item) for row in rows for item in row)
        pairs = Counter(pair for row in rows for pair in combinations(row, 2))
        # No pair reaches support 200, so no larger pattern can reach it either.
        self.assertLess(max(pairs.values()), 200)
        expected = {item: count for item, count in singles.items() if count >= 200}
        with tempfile.TemporaryDirectory() as folder:
            for extension in ("csv", "bin", "parquet"):
                with self.subTest(extension=extension):
                    path = Path(folder) / f"transactions.{extension}"
                    generator.save(path)
                    miner = cuApriori(path, 0.02)
                    miner.mine()
                    self.assertEqual(miner.getPatterns(), expected)

    def test_gds_unavailable_fallback_and_required_mode(self):
        try:
            import kvikio.cufile_driver as driver
            cuApriori._getCudaModule()
        except (ImportError, RuntimeError) as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "transactions.bin"
            generator = TransactionalDatabase(10, 8, seed=42)
            generator.create()
            generator.save(path)
            with patch.object(driver, "get", return_value=False):
                miner = cuApriori(path, 1)
                self.assertEqual(miner.getInputBackend()["backend"], "numpy-cuda")
                self.assertIn("unavailable", miner.getInputBackend()["gds_fallback_reason"])
                with self.assertRaisesRegex(RuntimeError, "GDS input requested"):
                    cuApriori(path, 1, use_gds=True)

    def test_cufile_device_pipeline_with_compatibility_fallback(self):
        try:
            import kvikio.cufile_driver as driver
            cuApriori._getCudaModule()
            originalGet = driver.get
            originalGet("allow_compat_mode")
        except (ImportError, RuntimeError) as error:
            self.skipTest(str(error))
        # Exercise real CuFile I/O and GPU matrix construction even on a host
        # without nvidia-fs. This is NOT proof that direct DMA is available.
        def reportAvailable(key):
            return True if key == "is_gds_available" else originalGet(key)

        with tempfile.TemporaryDirectory() as folder:
            for count in (0, 1, 100):
                with self.subTest(count=count):
                    path = Path(folder) / "transactions.bin"
                    generator = TransactionalDatabase(count, 12, seed=42)
                    generator.create()
                    generator.save(path)
                    baseline = cuApriori(path, 0.2, use_gds=False)
                    baseline.mine()
                    with patch.object(driver, "get", side_effect=reportAvailable):
                        miner = cuApriori(path, 0.2, use_gds=True)
                    self.assertEqual(miner.getInputBackend()["backend"], "cufile")
                    self.assertIsNone(miner._Database)
                    miner.mine()
                    self.assertEqual(miner.getPatterns(), baseline.getPatterns())

    def test_cuda_patterns_match_across_formats(self):
        try:
            cuApriori._getCudaModule()
        except (ImportError, RuntimeError) as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as folder:
            generator = TransactionalDatabase(
                100, 8, min_length=1, max_length=5, seed=42
            )
            generator.create()
            expected = expected_patterns(generator.getTransactions(), 20)
            for extension in ("csv", "bin", "parquet"):
                with self.subTest(extension=extension):
                    path = Path(folder) / f"transactions.{extension}"
                    generator.save(path)
                    miner = cuApriori(path, 20, use_gds=False, use_gpu_io=False)
                    miner.mine()
                    self.assertEqual(miner.getPatterns(), expected)

    def test_identical_transactions_and_patterns(self):
        try:
            cuApriori._getCudaModule()
        except (ImportError, RuntimeError) as error:
            self.skipTest(str(error))
        for count in (0, 1, 100):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as folder:
                generator = TransactionalDatabase(
                    count, 8, min_length=1, max_length=5, seed=42
                )
                generator.create()
                rows = generator.getTransactions()
                expected = [tuple(map(str, row)) for row in rows]
                supports = {}
                for row in expected:
                    for size in range(1, len(row) + 1):
                        for pattern in combinations(sorted(row), size):
                            key = "\t".join(pattern)
                            supports[key] = supports.get(key, 0) + 1
                threshold = max(1, int(count * 0.2))
                expectedPatterns = {key: value for key, value in supports.items()
                                    if value >= threshold}
                for extension in ("csv", "bin", "parquet", "BIN", "PARQUET"):
                    path = Path(folder) / f"transactions.{extension}"
                    generator.save(path)
                    miner = cuApriori(path, threshold, use_gds=False, use_gpu_io=False)
                    self.assertEqual(miner._Database, expected)
                    miner.mine()
                    self.assertEqual(miner.getPatterns(), expectedPatterns)

    def test_gpu_loaders_and_bitsets_match_independent_supports(self):
        try:
            import cudf
            cuApriori._getCudaModule()
        except (ImportError, RuntimeError) as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as folder:
            # Straddle bitset word boundaries and require multiple mining levels.
            for count in (0, 1, 31, 32, 33, 65):
                generator = TransactionalDatabase(count, 8, max_length=5, seed=42)
                generator.create()
                threshold = max(1, count // 8)
                expected = expected_patterns(generator.getTransactions(), threshold)
                for extension in ("csv", "parquet"):
                    with self.subTest(count=count, extension=extension):
                        path = Path(folder) / f"transactions.{extension}"
                        generator.save(path)
                        miner = cuApriori(path, threshold, use_gpu_io=True)
                        self.assertEqual(miner.getInputBackend()["backend"], "cudf")
                        miner.mine()
                        self.assertEqual(miner.getPatterns(), expected)

    def test_tabbed_text_uses_gpu_loader(self):
        try:
            import cudf
            cuApriori._getCudaModule()
        except (ImportError, RuntimeError) as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "transactions.csv"
            path.write_bytes(b"0\t1\t2\r\n1\t2\r\n0\t2")
            miner = cuApriori(path, 1, use_gpu_io=True)
            self.assertEqual(miner.getInputBackend()["backend"], "cudf")
            miner.mine()
            self.assertEqual(miner.getPatterns(), expected_patterns(
                [[0, 1, 2], [1, 2], [0, 2]], 1
            ))

    def test_binary_bulk_loading_without_gds(self):
        try:
            cuApriori._getCudaModule()
        except (ImportError, RuntimeError) as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "transactions.bin"
            for rows in ([], [[0]], [[1, 1, 2], [], [0, 2], [4294967295, 2]]):
                with self.subTest(rows=rows):
                    # Write raw baskets so empty rows and duplicate entries
                    # reach the bulk loader rather than converter cleanup.
                    offsets = [0]
                    flat = []
                    for row in rows:
                        flat.extend(row)
                        offsets.append(len(flat))
                    universe = max(1, max(flat, default=1))
                    path.write_bytes(
                        struct.pack("<8sQQQ", b"CUPAMI01", len(rows), len(flat), universe)
                        + struct.pack(f"<{len(offsets)}Q", *offsets)
                        + struct.pack(f"<{len(flat)}I", *flat)
                    )
                    miner = cuApriori(path, 1, use_gds=False, use_gpu_io=True)
                    self.assertEqual(miner.getInputBackend()["backend"], "numpy-cuda")
                    self.assertIsNone(miner._Database)
                    miner.mine()
                    self.assertEqual(miner.getPatterns(), expected_patterns(rows, 1))

    def test_duplicates_empty_baskets_and_text_fallback(self):
        try:
            cuApriori._getCudaModule()
        except (ImportError, RuntimeError) as error:
            self.skipTest(str(error))
        rows = [[1, 1, 2], [], [2, 3], [1, 2, 3]]
        expected = expected_patterns(rows, 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicates.parquet"
            pq.write_table(pa.table({"items": pa.array(rows, type=pa.list_(pa.uint32()))}), path)
            miner = cuApriori(path, 1)
            miner.mine()
            self.assertEqual(miner.getPatterns(), expected)
            path = Path(folder) / "quoted.csv"
            path.write_text('"apple,pear",banana\nbanana\n', encoding="utf-8")
            miner = cuApriori(path, 1)
            self.assertEqual(miner.getInputBackend()["backend"], "cpu")
            miner.mine()
            self.assertEqual(miner.getPatterns(), {
                "apple,pear": 1, "banana": 2, "apple,pear\tbanana": 1,
            })

    def test_invalid_binary_files(self):
        cases = [
            b"short",
            struct.pack("<8sQQQ", b"INVALID!", 0, 0, 8),
            struct.pack("<8sQQQ", b"CUPAMI01", 1, 1, 8),
            struct.pack("<8sQQQ2QI", b"CUPAMI01", 1, 1, 8, 1, 1, 2),
            struct.pack("<8sQQQ3Q2I", b"CUPAMI01", 2, 2, 8, 0, 3, 2, 1, 2),
            struct.pack("<8sQQQ2QI", b"CUPAMI01", 1, 1, 8, 0, 1, 9),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.bin"
            for data in cases:
                with self.subTest(data=data):
                    path.write_bytes(data)
                    with self.assertRaises(ValueError):
                        cuApriori(path, 1)

    def test_invalid_parquet_files(self):
        tables = [
            pa.table({"other": [1]}),
            pa.table({"items": ["1,2"]}),
            pa.table({"items": pa.array([None], type=pa.list_(pa.uint32()))}),
            pa.table({"items": pa.array([[None]], type=pa.list_(pa.uint32()))}),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.parquet"
            for table in tables:
                with self.subTest(schema=table.schema):
                    pq.write_table(table, path)
                    with self.assertRaises(ValueError):
                        cuApriori(path, 1)


if __name__ == "__main__":
    unittest.main()
