"""Independent support checks for SBT and flat multiple fuzzy miners."""
import itertools
from pathlib import Path
import random
import tempfile
import unittest

from cupami import cuMFFIMiner, cuMultiFuzzyBaseline


class MultipleFuzzyTest(unittest.TestCase):
    def test_both_searches_match_exhaustive_supports(self):
        rng = random.Random(42)
        labels = ['1.L', '1.H', '2.L', '2.H', '3.L']
        rows = [{label: rng.randrange(11) / 10 for label in labels} for _ in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'fuzzy.csv'
            # Include a duplicate label with lower membership to check max-dedup.
            file.write_text('\n'.join('\t'.join(labels + ['1.L']) + ':' +
                            '\t'.join(str(row[label]) for label in labels) + '\t0'
                            for row in rows) + '\n')
            for threshold in (1, 0.2):
                cutoff = threshold * len(rows) if isinstance(threshold, float) else threshold
                expected = {}
                for size in range(1, 4):
                    for pattern in itertools.combinations(labels, size):
                        if len({label.split('.')[0] for label in pattern}) != size:
                            continue
                        support = sum(min(row[label] for label in pattern) for row in rows)
                        if support >= cutoff - 1e-10:
                            expected[tuple(sorted(pattern))] = support
                for miner_class in (cuMFFIMiner, cuMultiFuzzyBaseline):
                    for mode in ('gpu', 'python'):
                        with self.subTest(miner=miner_class.__name__, mode=mode, threshold=threshold):
                            miner = miner_class(file, threshold, inputMode=mode)
                            miner.mine()
                            actual = {tuple(sorted(pattern.split('\t') if isinstance(pattern, str) else pattern)): support
                                      for pattern, support in miner.getPatterns().items()}
                            self.assertEqual(actual.keys(), expected.keys())
                            for pattern in expected:
                                self.assertAlmostEqual(actual[pattern], expected[pattern], places=8)
                            miner.save(Path(directory) / 'patterns.txt')
                            self.assertTrue((Path(directory) / 'patterns.txt').is_file())
                            self.assertEqual(miner._grouped, miner_class is cuMFFIMiner)


if __name__ == '__main__':
    unittest.main()
