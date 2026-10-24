import tempfile
import unittest
from pathlib import Path

from cupami import convertFile, convertTransactions, cuApriori, cuPFApriori


class TransactionConverterTest(unittest.TestCase):
    def test_shared_converter_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "transactions.csv"
            convertFile([[1, 2], [3]], path, dataType="transactional")
            self.assertEqual(path.read_text(), "1,2\n3\n")
            with self.assertRaisesRegex(ValueError, "Unsupported data type"):
                convertFile([[1, 2]], path, dataType="spatial")
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

    def test_temporal_round_trip_preserves_repeats_and_empty_baskets(self):
        rows = [[3, 1, 2], [1, 0], [1, 2], [8]]
        with tempfile.TemporaryDirectory() as directory:
            source = rows
            for suffix in ('csv', 'bin', 'parquet', 'csv'):
                path = Path(directory) / ('temporal.' + suffix)
                summary = convertFile(source, path, dataType='temporal')
                self.assertEqual(summary['transactions'], 4)
                reader = cuPFApriori(path, 1, 8)
                self.assertEqual(reader._timestamps, [1, 1, 3, 8])
                self.assertEqual(reader._Database, [('0',), ('2',), ('1', '2'), ()])
                source = path

    def test_invalid_ids_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "transactions.bin"
            for item in ("apple", "1.5", "-1", "4294967296"):
                with self.subTest(item=item), self.assertRaises(ValueError):
                    convertTransactions([[item]], path)
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
