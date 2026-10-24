import importlib
import io
from itertools import combinations
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from cupami import cuFFIMiner


def reference(rows, threshold):
    labels = sorted({label for row in rows for label in row})
    totals = {label: math.fsum(row.get(label, 0) for row in rows) for label in labels}
    best = {}
    for label in sorted(labels, key=lambda label: (-totals[label], label)):
        best.setdefault(label.rsplit('.', 1)[0], label)
    selected = sorted(best.values())
    result = {}
    for size in range(1, len(selected) + 1):
        for pattern in combinations(selected, size):
            support = math.fsum(min(row.get(label, 0) for label in pattern) for row in rows)
            if support > 0 and support >= threshold:
                result['\t'.join(pattern)] = support
    return result


def csv_text(rows, sep=','):
    return '\n'.join(sep.join(row) + ':' + sep.join(str(value) for value in row.values()) for row in rows)


class FuzzyTest(unittest.TestCase):
    def compare(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for pattern in actual:
            self.assertAlmostEqual(actual[pattern], expected[pattern], places=10)

    def test_manual_fuzzy_supports(self):
        text = 'A.L,A.H,B.H,C.M:0.4,0.6,1,0.5\nA.L,A.H,B.H,C.M:0.7,0.3,0.5,0.8\nA.L,A.H,C.M:0.8,0.2,0.2\n:\n'
        expected = {'A.L': 1.9, 'B.H': 1.5, 'C.M': 1.5, 'A.L\tB.H': .9,
                    'A.L\tC.M': 1.3, 'B.H\tC.M': 1.0, 'A.L\tB.H\tC.M': .9}
        with tempfile.TemporaryDirectory() as directory:
            for separator in (',', '\t', ' '):
                file = Path(directory) / 'fuzzy.csv'
                file.write_text(text.replace(',', separator))
                miner = cuFFIMiner(file, .2)
                self.assertEqual(miner._transactionCount, 4)
                self.assertEqual(miner.getMiningTime(), 0)
                miner.mine()
                self.compare(miner.getPatterns(), expected)
                miner.mine()
                self.compare(miner.getPatterns(), expected)
                self.assertAlmostEqual(miner.getTotalRuntime(), miner.getLoadingTime() + miner.getRuntime())
                self.assertEqual(list(miner.getPatternsAsDataFrame().columns), ['Patterns', 'Support'])
                miner.save(Path(directory) / 'patterns.txt')
            miner = cuFFIMiner(file, 1)
            miner.mine()
            self.compare(miner.getPatterns(), {key: value for key, value in expected.items() if value >= 1})

    def test_randomized_against_cpu_enumeration(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'fuzzy.csv'
            for seed in range(4):
                rng = random.Random(seed)
                rows = []
                for _ in range(12):
                    row = {}
                    for item in range(5):
                        if rng.random() < .7:
                            value = rng.choice([0, .25, .5, .75, 1])
                            row[f'{item}.L'], row[f'{item}.H'] = value, 1 - value
                    rows.append(row)
                file.write_text(csv_text(rows))
                for threshold in (1, 2, 3):
                    with self.subTest(seed=seed, threshold=threshold):
                        miner = cuFFIMiner(file, threshold)
                        miner._INTERSECTION_BATCH_BYTES = 8 * len(rows)
                        miner.mine()
                        self.compare(miner.getPatterns(), reference(rows, threshold))

    def test_url_ties_zeros_and_empty_data(self):
        module = importlib.import_module('cupami.fuzzyFrequentPattern.cuFFIMiner')
        for text, expected in [('A.L,A.H,B.H:0.5,0.5,1\n', {'A.H': .5, 'B.H': 1, 'A.H\tB.H': .5}),
                               ('A.L:0\n:\n', {}), ('', {})]:
            with patch.object(module, 'urlopen', return_value=io.BytesIO(text.encode())):
                miner = cuFFIMiner('https://example.test/fuzzy.csv', .1)
            miner.mine()
            self.compare(miner.getPatterns(), expected)

    def test_decimal_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'fuzzy.csv'
            file.write_text('A.H:0.7\n' * 10000)
            miner = cuFFIMiner(file, 7000)
            miner.mine()
            self.compare(miner.getPatterns(), {'A.H': 7000})

    def test_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'fuzzy.csv'
            for text in ['a,b:0.5', 'a,a:0.5,0.5', 'a:-0.1', 'a:1.1', 'a:nan',
                         'a:inf', 'a:bad', '.L:1', 'a.:1', 'a,,:1,1,1', 'a,b', 'a:1:2']:
                file.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    cuFFIMiner(file, 1)
            file.write_text('A:1')
            for support in (0, -1, True, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    cuFFIMiner(file, support)
            for separator in ('', '::', ':', '\n'):
                with self.assertRaises(ValueError):
                    cuFFIMiner(file, 1, sep=separator)
            with self.assertRaises(ValueError):
                cuFFIMiner('fuzzy.bin', 1)
            with self.assertRaises(TypeError):
                cuFFIMiner([], 1)


if __name__ == '__main__':
    unittest.main()
