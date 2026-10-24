import csv
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from cupami import TransactionalDatabase, TemporalDatabase, cuApriori, cuPFApriori


class GeneratorTest(unittest.TestCase):
    def test_average_length_and_reproducibility(self):
        for cls in (TransactionalDatabase, TemporalDatabase):
            with self.subTest(generator=cls.__name__):
                rows = cls(20000, 20, avg_length=6.5, seed=7).create()
                self.assertEqual(rows, cls(20000, 20, avg_length=6.5, seed=7).create())
                baskets = [row[1:] for row in rows] if cls is TemporalDatabase else rows
                self.assertAlmostEqual(statistics.mean(map(len, baskets)), 6.5, delta=.1)
                self.assertGreater(len(set(map(len, baskets))), 1)
                for basket in baskets:
                    self.assertEqual(basket, sorted(set(basket)))
                    self.assertTrue(1 <= len(basket) <= 20)
                    self.assertTrue(all(1 <= item <= 20 for item in basket))

    def test_boundaries_and_fractional_lengths(self):
        for cls in (TransactionalDatabase, TemporalDatabase):
            for average in (1, 1.2, 4.8, 5):
                rows = cls(10000, 5, avg_length=average, seed=4).create()
                lengths = [len(row) - (cls is TemporalDatabase) for row in rows]
                self.assertAlmostEqual(statistics.mean(lengths), average, delta=.05)
            self.assertEqual(cls(0, 1, avg_length=1).create(), [])
            self.assertEqual(len(cls(1, 1).create()[0]), 1 + (cls is TemporalDatabase))

    def test_invalid_average(self):
        for cls in (TransactionalDatabase, TemporalDatabase):
            for value in (0, 5.1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    cls(10, 5, avg_length=value)
            for value in (True, '3'):
                with self.assertRaises(TypeError):
                    cls(10, 5, avg_length=value)

    def test_measurements_for_memory_and_direct_file_generation(self):
        process = Mock()
        process.memory_info.return_value.rss = 1234
        process.memory_full_info.return_value.uss = 567
        psutil = Mock()
        psutil.Process.return_value = process
        psutil.AccessDenied = PermissionError
        for cls in (TransactionalDatabase, TemporalDatabase):
            generator = cls(10, 5, avg_length=3, seed=2)
            self.assertEqual(generator.getRuntime(), 0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'data.csv'
                for direct in (False, True):
                    with patch.dict('sys.modules', {'psutil': psutil}), patch(
                        'cupami.extras.syntheticDataGenerator.TransactionalDatabase.time.perf_counter',
                        side_effect=[10, 12.5],
                    ):
                        if direct:
                            method = (generator.createTemporalDatabase if cls is TemporalDatabase
                                      else generator.createTransactionalDatabase)
                            method(path)
                        else:
                            generator.create()
                    self.assertEqual(generator.getRuntime(), 2.5)
                    self.assertEqual(generator.getMemoryRSS(), 1234)
                    self.assertEqual(generator.getMemoryUSS(), 567)
                with path.open() as source:
                    self.assertEqual(len(list(csv.reader(source))), 10)
                generator.save(path)
                self.assertEqual(generator.getRuntime(), 2.5)

    def test_saved_and_direct_generation_round_trip(self):
        for cls, reader in ((TransactionalDatabase, cuApriori),
                            (TemporalDatabase, cuPFApriori)):
            for count in (0, 1, 50):
                for extension in ('csv', 'bin', 'parquet'):
                    with self.subTest(generator=cls.__name__, count=count,
                                      extension=extension), tempfile.TemporaryDirectory() as directory:
                        saved = Path(directory) / ('saved.' + extension)
                        direct = Path(directory) / ('direct.' + extension)
                        generator = cls(count, 8, avg_length=3.5, seed=10)
                        rows = generator.create()
                        generator.save(saved)
                        streaming = cls(count, 8, avg_length=3.5, seed=10)
                        method = (streaming.createTemporalDatabase if cls is TemporalDatabase
                                  else streaming.createTransactionalDatabase)
                        method(direct)
                        kwargs = {} if cls is TemporalDatabase else dict(
                            use_gds=False, use_gpu_io=False)
                        args = (1, 1000) if cls is TemporalDatabase else (1,)
                        left, right = (reader(path, *args, **kwargs) for path in (saved, direct))
                        expected = [tuple(map(str, row[1:] if cls is TemporalDatabase else row))
                                    for row in rows]
                        self.assertEqual(left._Database, expected)
                        self.assertEqual(right._Database, expected)
                        if cls is TemporalDatabase:
                            self.assertEqual(left._timestamps, [row[0] for row in rows])
                            self.assertEqual(right._timestamps, left._timestamps)
                        self.assertGreaterEqual(streaming.getRuntime(), 0)

    def test_missing_psutil_and_write_errors(self):
        generator = TransactionalDatabase(2, 3, avg_length=2)
        with patch.dict('sys.modules', {'psutil': None}):
            generator.create()
            self.assertEqual(generator.getMemoryRSS(), 0)
            self.assertEqual(generator.getMemoryUSS(), 0)
            with self.assertRaises(ValueError):
                generator.createTransactionalDatabase('invalid.extension')

    def test_temporal_repeat_and_missing_flags(self):
        for repeat in (False, True):
            for missing in (False, True):
                generator = TemporalDatabase(200, 10, avg_length=3, seed=11,
                                             repeatAllowed=repeat, missingAllowed=missing)
                rows = generator.create()
                steps = [right[0] - left[0] for left, right in zip(rows, rows[1:])]
                allowed = {1} | ({0} if repeat else set()) | ({2} if missing else set())
                self.assertEqual(set(steps), allowed)
                if repeat:
                    merged = TemporalDatabase(200, 10, avg_length=3, seed=11,
                                               repeatAllowed=True, missingAllowed=missing,
                                               mergeTimestamps=True).create()
                    expected = {}
                    for timestamp, *items in rows:
                        expected.setdefault(timestamp, set()).update(items)
                    self.assertEqual(merged, [[ts, *sorted(items)] for ts, items in expected.items()])
        with self.assertRaises(ValueError):
            TemporalDatabase(10, 5, mergeTimestamps=True)
        for key in ('repeatAllowed', 'missingAllowed', 'mergeTimestamps'):
            with self.assertRaises(TypeError):
                TemporalDatabase(10, 5, **{key: 1})

    def test_temporal_merging_keeps_unique_items(self):
        generator = TemporalDatabase(20, 10, avg_length=3, seed=3,
                                     repeatAllowed=True, mergeTimestamps=True)
        with patch.object(generator._random, 'randint', return_value=0):
            rows = generator.create()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 1)
        self.assertEqual(rows[0][1:], sorted(set(rows[0][1:])))


if __name__ == '__main__':
    unittest.main()
