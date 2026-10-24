"""Small, explicitly counted examples shared across miners and formats."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from cupami import (TransactionalDatabase, TemporalDatabase, cuApriori, cuEclat,
                    cuFPGrowth, cuPFApriori, cuPFEclat, cuPFGrowth)


class MinerExamplesTest(unittest.TestCase):
    def check_timings(self, miner):
        self.assertGreaterEqual(miner.getLoadingTime(), 0)
        self.assertGreaterEqual(miner.getMiningTime(), 0)
        self.assertEqual(miner.getRuntime(), miner.getMiningTime())
        self.assertAlmostEqual(miner.getTotalRuntime(), miner.getLoadingTime() + miner.getMiningTime())

    def test_timing_labels_exclude_idle_time(self):
        with patch.object(cuApriori, '_readTransactions', return_value=[('a',)]), patch(
            'cupami.frequentPattern.cuApriori.time.perf_counter', side_effect=[10, 12.5]
        ):
            miner = cuApriori([], 1, use_gpu_io=False, use_gds=False)
        self.assertEqual(miner.getLoadingTime(), 2.5)
        self.assertEqual(miner.getMiningTime(), 0)
        self.assertEqual(miner.getTotalRuntime(), 2.5)
        miner._startTime, miner._endTime = 100, 104
        self.assertEqual(miner.getMiningTime(), 4)
        self.assertEqual(miner.getTotalRuntime(), 6.5)

    def test_transactional_manual_counts_all_formats(self):
        rows = [[1, 2, 2, 3], [], [1, 2], [1, 3], [2, 3]]
        expected = {'1': 3, '2': 3, '3': 3, '1\t2': 2, '1\t3': 2, '2\t3': 2}
        with tempfile.TemporaryDirectory() as directory:
            generator = TransactionalDatabase(len(rows), 3, avg_length=2)
            generator._database = rows
            files = []
            for extension in ('csv', 'bin', 'parquet'):
                file = Path(directory) / ('manual.' + extension)
                generator.save(file)
                files.append(file)
            for cls in (cuApriori, cuEclat, cuFPGrowth):
                for support in (2, .5):
                    for source in [rows, *files]:
                        for gpu_io in (False, True):
                            with self.subTest(miner=cls.__name__, source=str(source), gpu=gpu_io):
                                miner = cls(source, support, use_gds=False, use_gpu_io=gpu_io)
                                self.assertEqual(miner.getMiningTime(), 0)
                                miner.mine()
                                self.assertEqual(miner.getPatterns(), expected)
                                self.check_timings(miner)

    def test_temporal_manual_supports_and_gaps_all_formats(self):
        rows = [[1, 1, 2], [1, 2, 3], [3, 1, 3], [5, 1, 2, 3], [8]]
        expected = {'1': (3, 3), '2': (3, 4), '3': (3, 3),
                    '1\t2': (2, 4), '1\t3': (2, 3), '2\t3': (2, 4)}
        with tempfile.TemporaryDirectory() as directory:
            generator = TemporalDatabase(len(rows), 3, avg_length=2)
            generator._database = rows
            files = []
            for extension in ('csv', 'bin', 'parquet'):
                file = Path(directory) / ('manual.' + extension)
                generator.save(file)
                files.append(file)
            for cls in (cuPFApriori, cuPFEclat, cuPFGrowth):
                for support in (2, .4):
                    for source in [rows, *files]:
                        with self.subTest(miner=cls.__name__, source=str(source), minSup=support):
                            miner = cls(source, support, maxPer=4)
                            miner.mine()
                            self.assertEqual(miner.getPatterns(), expected)
                            self.check_timings(miner)


if __name__ == '__main__':
    unittest.main()
