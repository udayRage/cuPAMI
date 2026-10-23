import itertools
import random
import tempfile
import unittest
from pathlib import Path

from cupami import cuPFApriori, cuPFEclat, cuPFGrowth, TemporalDatabase

MINERS = (cuPFApriori, cuPFEclat, cuPFGrowth)


def brute(rows, support, period, end=None):
    labels = sorted({str(item) for row in rows for item in row[1:]})
    end = max((row[0] for row in rows), default=0) if end is None else end
    result = {}
    for width in range(1, len(labels) + 1):
        for pattern in itertools.combinations(labels, width):
            timestamps = sorted(row[0] for row in rows if set(pattern) <= set(map(str, row[1:])))
            if not timestamps:
                continue
            points = [0, *timestamps, end]
            maximum = max(b - a for a, b in zip(points, points[1:]))
            if len(timestamps) >= support and maximum <= period:
                result['\t'.join(pattern)] = (len(timestamps), maximum)
    return result


class PeriodicMinersTest(unittest.TestCase):
    def check(self, rows, support, period, end=None):
        expected = brute(rows, support, period, end)
        for cls in MINERS:
            with self.subTest(miner=cls.__name__):
                miner = cls(rows, support, period, endTime=end)
                miner.mine()
                self.assertEqual(miner.getPatterns(), expected)
                miner.mine()
                self.assertEqual(miner.getPatterns(), expected)

    def test_paper_example(self):
        baskets = ['ab', 'acdi', 'cefj', 'abfgh', 'bcd', 'def', 'abi', 'cde', 'abef', 'acg']
        rows = [[i + 1, *basket] for i, basket in enumerate(baskets)]
        self.check(rows, 3, 3)
        miner = cuPFApriori(rows, 3, 3)
        miner.mine()
        self.assertEqual(len(miner.getPatterns()), 9)
        self.assertEqual(miner.getPatterns()['a\tb'], (4, 3))

    def test_random_bruteforce(self):
        for seed in range(4):
            rng = random.Random(seed)
            rows = [[i * 3, *rng.sample(range(6), rng.randint(0, 6))] for i in range(1, 19)]
            rng.shuffle(rows)
            for support, period in [(1, 54), (3, 15), (5, 9)]:
                self.check(rows, support, period)

    def test_boundaries_duplicates_empty_and_zero(self):
        self.check([[9, 'a'], [0, 'a', 'a', '0'], [4], [9, 'a', 'b'], [15]], 1, 10)
        self.check([[2, 'x'], [3, 'x']], 1, 5, end=20)
        self.check([], 1, 0)
        self.check([[0, 'x'], [0, 'y']], 1, 0)
        self.check([[2**53 + 1, 'x'], [2**53 + 3, 'x']], 1, 2**53 + 1)

    def test_single_path_and_word_boundaries(self):
        self.check([[i, 'a', 'b', *(['c'] if i % 2 else [])] for i in range(1, 66)], 1, 4)

    def test_fractional_thresholds(self):
        rows = [[2, 'a'], [4, 'a'], [6, 'b'], [8, 'a'], [10, 'a']]
        for cls in MINERS:
            miner = cls(rows, .4, .4)
            miner.mine()
            self.assertEqual(miner.getPatterns(), {'a': (4, 4)})

    def test_generator_csv_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = TemporalDatabase(50, 5, avg_length=2.5, seed=42, timestampStep=2,
                                               repeatChance=.3)
            generator.create()
            path = Path(directory) / 'temporal.csv'
            generator.save(path)
            expected = brute(generator.getTransactions(), 3, 20)
            for cls in MINERS:
                miner = cls(path, 3, 20)
                miner.mine()
                self.assertEqual(miner.getPatterns(), expected)
                self.assertEqual(list(miner.getPatternsAsDataFrame().columns), ['Patterns', 'Support', 'Periodicity'])
                miner.save(Path(directory) / 'patterns.txt')
                self.assertGreaterEqual(miner.getRuntime(), 0)
                self.assertEqual(miner.getInputBackend()['backend'], 'cpu')
            tab = Path(directory) / 'tab.txt'
            tab.write_text('1\ta\tb\n3\ta\n5\tb\n')
            self.assertEqual(cuPFApriori(tab, 1, 5)._timestamps, [1, 3, 5])

    def test_invalid_inputs(self):
        for cls in MINERS:
            for rows in ([[-1, 'a']], [[1.2, 'a']], [['bad', 'a']]):
                with self.assertRaises(ValueError):
                    cls(rows, 1, 2)
            for support, period in [(0, 1), (1, -1), (1, float('nan')), (float('inf'), 1)]:
                with self.assertRaises((ValueError, OverflowError)):
                    cls([[1, 'a']], support, period)
            with self.assertRaises(ValueError):
                cls([[5, 'a']], 1, 2, endTime=3)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'transactional.bin'
                path.write_bytes(b'CUPAMI01' + bytes(24))
                with self.assertRaises(ValueError):
                    cls(path, 1, 2)


if __name__ == '__main__':
    unittest.main()
