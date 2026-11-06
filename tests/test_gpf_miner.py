"""Independent correctness checks for the public gPF-Miner bitset adaptation."""
import itertools
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from cupami import gPFMiner, TemporalDatabase


def oracle(rows, min_support, max_period, end=None):
    labels = sorted({str(item) for row in rows for item in row[1:]})
    timeline = {}
    for row in rows:
        timeline.setdefault(row[0], set()).update(map(str, row[1:]))
    end = max(timeline, default=0) if end is None else end
    result = {}
    for size in range(1, len(labels) + 1):
        for pattern in itertools.combinations(labels, size):
            points = sorted(t for t, basket in timeline.items() if set(pattern) <= basket)
            if not points:
                continue
            period = max(b - a for a, b in zip([0, *points], [*points, end]))
            if len(points) >= min_support and period <= max_period:
                result['\t'.join(pattern)] = (len(points), period)
    return result


class GPFMinerTest(unittest.TestCase):
    def check(self, rows, support, period, end=None):
        miner = gPFMiner(rows, support, period, endTime=end)
        expected = oracle(rows, support, period, end)
        miner.mine()
        self.assertEqual(miner.getPatterns(), expected)
        self.assertIsNone(miner._timelineBits)
        miner.mine()
        self.assertEqual(miner.getPatterns(), expected)
        return miner

    def test_random_against_exhaustive_enumeration(self):
        for seed in range(8):
            rng = random.Random(seed)
            rows = [[i * 3, *rng.sample(range(6), rng.randrange(7))] for i in range(30)]
            rng.shuffle(rows)
            for support, period in [(1, 90), (4, 30), (8, 15)]:
                with self.subTest(seed=seed, support=support, period=period):
                    self.check(rows, support, period)

    def test_word_boundaries_padding_start_and_end_gaps(self):
        for end in [0, 1, 63, 64, 65, 127, 128, 129]:
            rows = [[t, 'a', *(['b'] if t % 3 == 0 else [])] for t in range(end + 1)]
            self.check(rows, 1, 3)
        self.check([[0, 'a', 'b'], [63, 'a', 'b'], [64, 'a', 'b'], [65, 'a', 'b'], [129]], 2, 64)
        # Largest interior gap must not be overwritten by a short final gap.
        self.check([[1, 'a', 'b'], [10, 'a', 'b'], [11, 'a', 'b']], 1, 9)
        self.check([[5, 'a', 'b'], [6, 'a', 'b']], 1, 5, end=20)
        self.check([[0, 'a', 'b'], [0, 'a', 'c']], 1, 0)
        self.check([], 1, 0)
        self.check([[10]], 1, 10)

    def test_same_timestamp_unions_and_fractional_thresholds(self):
        rows = [[1, 'a', 'a'], [1, 'b'], [3, 'a', 'b'], [5]]
        miner = self.check(rows, 2, 2)
        self.assertEqual(miner.getPatterns()['a\tb'], (2, 2))
        fractional = gPFMiner(rows, .5, .4)
        fractional.mine()
        self.assertEqual(fractional.getPatterns(), oracle(rows, 2, 2))

    def test_batched_deep_patterns(self):
        rows = [[t, 'a', 'b', 'c', 'd', 'e', 'f'] for t in range(1, 12)]
        miner = gPFMiner(rows, 2, 1)
        miner._BATCH_SIZE = 2
        miner.mine()
        self.assertEqual(miner.getPatterns(), oracle(rows, 2, 1))
        self.assertEqual(len(miner.getPatterns()), 63)

    def test_generator_formats_outputs_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = TemporalDatabase(40, 5, avg_length=2, seed=42)
            rows = generator.create()
            for suffix in ('csv', 'bin', 'parquet'):
                file = Path(directory) / ('temporal.' + suffix)
                generator.save(file)
                miner = gPFMiner(file, 3, 20)
                miner.mine()
                self.assertEqual(miner.getPatterns(), oracle(rows, 3, 20))
                self.assertEqual(list(miner.getPatternsAsDataFrame().columns), ['Patterns', 'Support', 'Periodicity'])
                miner.save(Path(directory) / 'patterns.txt')
                self.assertGreaterEqual(miner.getTotalRuntime(), miner.getRuntime())
            file = Path(directory) / 'cli.csv'
            file.write_text('1,a,b\n3,a,b\n5,a,b\n')
            result = subprocess.run([sys.executable, 'cupami/periodicFrequentPattern/gPFMiner.py',
                                     str(file), str(Path(directory)/'cli.txt'), '2', '2'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('a\tb', (Path(directory)/'cli.txt').read_text())

    def test_allocation_limit_and_invalid_parameters(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                gPFMiner([], 1, 2, maxTimelineBytes=value)
        for rows, support, period in [([[-1,'a']],1,2), ([[1,'a']],0,2), ([[1,'a']],1,-1)]:
            with self.assertRaises(ValueError):
                gPFMiner(rows, support, period)
        miner = gPFMiner([[2**53+1,'a']],1,2**53+1)
        with self.assertRaisesRegex(ValueError, 'Timestamp bitsets need'):
            miner.mine()
        self.assertIsNone(miner._timelineBits)


if __name__ == '__main__':
    unittest.main()
