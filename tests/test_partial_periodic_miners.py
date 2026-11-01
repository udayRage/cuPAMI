import itertools
import random
import tempfile
import unittest
from pathlib import Path

from cupami import cuPPPApriori, cuPPPEclat

MINERS = (cuPPPApriori, cuPPPEclat)


def brute(rows, minimum, period):
    merged = {}
    for ts, *items in rows:
        merged.setdefault(ts, set()).update(map(str, items))
    labels = sorted(set().union(*merged.values())) if merged else []
    result = {}
    for size in range(1, len(labels) + 1):
        for pattern in itertools.combinations(labels, size):
            times = sorted(t for t, items in merged.items() if set(pattern) <= items)
            support = sum(b - a <= period for a, b in zip(times, times[1:]))
            if support >= minimum:
                result['\t'.join(pattern)] = support
    return result


class PartialPeriodicTest(unittest.TestCase):
    def check(self, rows, minimum, period):
        expected = brute(rows, minimum, period)
        for cls in MINERS:
            miner = cls(rows, minimum, period)
            miner.mine()
            self.assertEqual(miner.getPatterns(), expected)
            miner.mine()
            self.assertEqual(miner.getPatterns(), expected)

    def test_random_against_bruteforce(self):
        for seed in range(5):
            rng = random.Random(seed)
            rows = [[rng.randrange(40), *rng.sample('abcde', rng.randrange(6))] for _ in range(30)]
            for minimum, period in ((1, 2), (3, 8), (0, 0)):
                self.check(rows, minimum, period)

    def test_boundaries_duplicates_and_word_boundaries(self):
        self.check([[100, 'a'], [102, 'a'], [200, 'a']], 1, 2)
        self.check([[1, 'a'], [1, 'b'], [2, 'a', 'b']], 1, 1)
        self.check([[i, 'a', *(['b'] if i % 2 else [])] for i in range(70)], 10, 2)
        self.check([], 1, 2)
        self.check([[2**53 + 1, 'a'], [2**53 + 3, 'a']], 1, 2)

    def test_thresholds_and_exports(self):
        rows = [[100, 'a'], [102, 'a'], [104, 'a'], [106, 'b']]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.tsv'
            path.write_text('100\ta\n102\ta\n104\ta\n106\tb\n')
            for cls in MINERS:
                for minimum, period in ((.5, .5), ('0.5', '0.5'), ('2', '2')):
                    miner = cls(path, minimum, period, '\t')
                    miner.mine()
                    self.assertEqual(miner.getPatterns(), {'a': 2})
                    self.assertEqual(list(miner.getPatternsAsDataFrame().columns), ['Patterns', 'periodicSupport'])
                    miner.save(Path(directory) / 'out.txt')
                    self.assertIn('a : 2', (Path(directory) / 'out.txt').read_text())
                for minimum, period in ((True, 1), (-1, 1), (1, -1), ('bad', 1), (1, float('inf'))):
                    with self.assertRaises(ValueError):
                        cls(rows, minimum, period)


if __name__ == '__main__':
    unittest.main()
