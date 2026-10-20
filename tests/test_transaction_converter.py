import tempfile
import unittest
from pathlib import Path

from cupami import convertFile, convertTransactions, cuApriori


class TransactionConverterTest(unittest.TestCase):
    def test_shared_converter_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "transactions.csv"
            convertFile([[1, 2], [3]], path, dataType="transactional")
            self.assertEqual(path.read_text(), "1,2\n3\n")
            with self.assertRaisesRegex(ValueError, "Unsupported data type"):
                convertFile([[1, 2]], path, dataType="temporal")
            self.assertEqual(path.read_text(), "1,2\n3\n")

    def test_tab_csv_round_trip_all_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "tabbed.csv"
            source.write_text("25\t52\t164\r\n39\t120\r\n", encoding="utf-8")
            expected = [("25", "52", "164"), ("39", "120")]
            for suffix in ("csv", "bin", "parquet"):
                path = root / f"converted.{suffix}"
                summary = convertTransactions(source, path)
                self.assertEqual(summary["transactions"], 2)
                self.assertEqual(summary["item_occurrences"], 5)
                self.assertEqual(summary["max_item_id"], 164)
                reader = cuApriori(path, 1, use_gpu_io=False, use_gds=False)
                self.assertEqual(reader._Database, expected)
            self.assertEqual((root / "converted.csv").read_text(), "25,52,164\n39,120\n")

    def test_normalization_and_empty_database(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "transactions.bin"
            summary = convertTransactions([["01", "1", "2"], [], ["0", "4294967295"]], path)
            self.assertEqual(summary["item_occurrences"], 4)
            self.assertEqual(cuApriori(path, 1, use_gds=False, use_gpu_io=False)._Database,
                             [("1", "2"), ("0", "4294967295")])
            for suffix in ("csv", "bin", "parquet"):
                path = Path(folder) / f"empty.{suffix}"
                summary = convertTransactions([], path)
                self.assertEqual(summary["transactions"], 0)
                self.assertEqual(cuApriori(path, 1, use_gds=False, use_gpu_io=False)._Database, [])

    def test_invalid_ids_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "transactions.bin"
            for item in ("apple", "1.5", "-1", "4294967296"):
                with self.subTest(item=item), self.assertRaises(ValueError):
                    convertTransactions([[item]], path)
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
