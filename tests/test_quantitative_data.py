import itertools
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cupami import (BinaryTransactionalDatabase, QuantitativeTransactionalDatabase,
                    TransactionalDatabase, convertFile, cuApriori)
from cupami.extras import _quantitative_io as qio


class QuantitativeDataTest(unittest.TestCase):
    def test_separate_generators_and_reproducibility(self):
        args = dict(transactions=100, items=10, avg_length=4.5, seed=42)
        self.assertEqual(BinaryTransactionalDatabase(**args).create(), TransactionalDatabase(**args).create())
        first = QuantitativeTransactionalDatabase(**args, min_quantity=2, max_quantity=7)
        rows = first.create()
        self.assertEqual(rows, QuantitativeTransactionalDatabase(**args, min_quantity=2, max_quantity=7).create())
        self.assertTrue(all(list(row) == sorted(row) and all(2 <= q <= 7 for q in row.values()) for row in rows))
        rows[0].clear()
        self.assertTrue(first.getTransactions()[0])
        with self.assertRaises(ValueError):
            QuantitativeTransactionalDatabase(**args, min_quantity=5, max_quantity=2)
        with self.assertRaises(TypeError):
            QuantitativeTransactionalDatabase(**args, min_quantity=True)

    def test_quantitative_all_format_pairs(self):
        rows = [{0: 2.5, 4: 7}, {}, {4294967295: .125}]
        with tempfile.TemporaryDirectory() as d:
            for left, right in itertools.product(('csv', 'tsv', 'bin', 'parquet'), repeat=2):
                source, output = Path(d) / ('source.' + left), Path(d) / ('output.' + right)
                convertFile(rows, source, dataType='quantitative')
                summary = convertFile(source, output, dataType='quantitative')
                self.assertEqual(qio.read(output), rows)
                self.assertEqual(summary['transactions'], 3)
                self.assertEqual(summary['item_occurrences'], 3)
            for suffix in ('csv', 'bin', 'parquet'):
                path = Path(d) / ('empty.' + suffix)
                convertFile([], path, dataType='quantitative')
                self.assertEqual(qio.read(path), [])

    def test_generated_saved_and_streamed_data(self):
        with tempfile.TemporaryDirectory() as d:
            for extension in ('csv', 'tsv', 'bin', 'parquet'):
                saved, direct = (Path(d) / (name + '.' + extension) for name in ('saved', 'direct'))
                generator = QuantitativeTransactionalDatabase(50, 8, 3.5, seed=10)
                rows = generator.create()
                generator.save(saved)
                QuantitativeTransactionalDatabase(50, 8, 3.5, seed=10).createTransactionalDatabase(direct)
                self.assertEqual(qio.read(saved), rows)
                self.assertEqual(qio.read(direct), rows)

    def test_binary_and_temporal_all_format_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            for kind, rows in (('binary', [[0, 2], [3]]), ('temporal', [[1, 0, 2], [1], [4, 3]])):
                for left, right in itertools.product(('csv', 'tsv', 'bin', 'parquet'), repeat=2):
                    source, output, result = (Path(d) / name for name in ('source.' + left, 'output.' + right, 'result.csv'))
                    convertFile(rows, source, dataType=kind)
                    convertFile(source, output, dataType=kind)
                    convertFile(output, result, dataType=kind)
                    self.assertEqual(result.read_text(), '0,2\n3\n' if kind == 'binary' else '1,0,2\n1\n4,3\n')

    def test_atomic_failure_and_in_place_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'data.csv'
            path.write_text('keep this')
            for rows in ([{1: -1}], [{1: float('nan')}], [{'01': 1, '1': 2}], [{1: True}], [{1: 2**53 + 1}]):
                with self.assertRaises(ValueError):
                    convertFile(rows, path, dataType='quantitative')
                self.assertEqual(path.read_text(), 'keep this')
            path.write_text('1,2\n3\n')
            convertFile(path, path)
            self.assertEqual(path.read_text(), '1,2\n3\n')
            self.assertEqual(list(Path(d).iterdir()), [path])

    def test_bad_quantitative_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.csv'
            for text in ('1,2:3\n', '01,1:2,3\n', '1:nan\n', '1,1:2,3\n', '1:9007199254740993\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    qio.read(path)
            binary = Path(d) / 'bad.bin'
            qio.write(binary, [{1: 2}])
            original = binary.read_bytes()
            for payload in (original[:-1], b'CUPAMI01' + original[8:], original[:32] + struct.pack('<QQ', 1, 1) + original[48:]):
                binary.write_bytes(payload)
                with self.assertRaises(ValueError):
                    qio.read(binary)
            with self.assertRaises(ValueError):
                cuApriori(binary, 1, use_gpu_io=False, use_gds=False)

    def test_cli_and_delimiters(self):
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d) / 'input.csv', Path(d) / 'output.tsv'
            source.write_text('1,3:2,4\n')
            result = subprocess.run([sys.executable, '-m', 'cupami.extras.datasetConverter', str(source), str(output),
                                     '--data-type', 'quantitative', '--output-sep', r'\t'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text(), '1\t3:2\t4\n')
            self.assertEqual(qio.read(output), [{1: 2, 3: 4}])
            convertFile([[1, 2]], output, outputSep=';')
            self.assertEqual(output.read_text(), '1;2\n')


if __name__ == '__main__':
    unittest.main()
