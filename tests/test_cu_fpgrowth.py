import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

from cupami import convertFile, cuApriori, cuEclat, cuFPGrowth


def brute_force(rows, threshold):
    rows = [set(str(value).strip() for value in row if str(value).strip()) for row in rows]
    rows = [row for row in rows if row]
    items = sorted(set().union(*rows)) if rows else []
    patterns = {}
    for length in range(1, len(items) + 1):
        for pattern in combinations(items, length):
            support = sum(set(pattern).issubset(row) for row in rows)
            if support >= threshold:
                patterns["\t".join(pattern)] = support
    return patterns


class FPGrowthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cuFPGrowth._getCudaModule()
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(str(error))

    def test_original_paper_example(self):
        rows = [list("facdgimp"), list("abcf lmo".replace(" ", "")),
                list("bfhjo"), list("bcksp"), list("afcelpmn")]
        expected = {"a": 3, "c": 4, "f": 4, "m": 3, "b": 3, "p": 3, "c\tp": 3}
        for length in range(2, 5):
            for pattern in combinations("acfm", length):
                expected["\t".join(pattern)] = 3
        miner = cuFPGrowth(rows, 3)
        miner.mine()
        self.assertEqual(miner.getPatterns(), expected)

    def test_randomized_patterns_and_supports(self):
        for seed in range(4):
            rng = random.Random(seed)
            rows = [rng.sample(range(7), rng.randrange(8)) for _ in range(33 + seed)]
            for support in (1, 7, 0.25, 1.0, 100):
                with self.subTest(seed=seed, support=support):
                    miner = cuFPGrowth(rows, support)
                    miner.mine()
                    expected = brute_force(rows, miner._minSup)
                    self.assertEqual(miner.getPatterns(), expected)
                    for algorithm in (cuApriori, cuEclat):
                        reference = algorithm(rows, support)
                        reference.mine()
                        self.assertEqual(miner.getPatterns(), reference.getPatterns())

    def test_single_path_with_different_supports(self):
        rows = [[0, 1, 2]] * 33 + [[0, 1]] * 32 + [[0]] * 2
        miner = cuFPGrowth(rows, 1)
        miner.mine()
        self.assertEqual(miner.getPatterns(), brute_force(rows, 1))
        dense = cuFPGrowth([list(range(8))] * 65, 1)
        dense.mine()
        self.assertEqual(len(dense.getPatterns()), 255)
        self.assertEqual(set(dense.getPatterns().values()), {65})

    def test_empty_duplicate_and_arbitrary_labels(self):
        for rows in ([], [[], []], [[0, 0, 1], [], [1, 4294967295]],
                     [["apple,pear", "banana"], ["banana", "cherry"], [" "]]):
            with self.subTest(rows=rows):
                miner = cuFPGrowth(rows, 1)
                miner.mine()
                self.assertEqual(miner.getPatterns(), brute_force(rows, 1))
        with self.assertRaisesRegex(ValueError, "positive minimum support"):
            cuFPGrowth([[1]], 0).mine()

    def test_weighted_conditional_counts_are_exact_uint64(self):
        import cupy as cp
        from cupami.frequentPattern._fptree import build_tree, conditional_tree

        big = 2**53 + 7
        tree = build_tree(cp, cp.array([0, 0, 1], dtype=cp.uint64),
                          cp.array([0, 1, 0], dtype=cp.uint32),
                          cp.array([big, big + 1], dtype=cp.uint64), 2, 1)
        self.assertEqual(dict(tree.header), {0: 2 * big + 1, 1: big})
        conditional = conditional_tree(cp, tree, 1, 2, 1)
        self.assertEqual(dict(conditional.header), {0: big})

    def test_all_formats_cpu_and_gpu_readers(self):
        rows = [[0, 1, 2], [0, 1], [1, 2], [0, 1, 2], [2, 3]]
        expected = brute_force(rows, 2)
        with tempfile.TemporaryDirectory() as folder:
            for extension in ("csv", "bin", "parquet"):
                path = Path(folder) / f"transactions.{extension}"
                convertFile(rows, path)
                for gpu in (False, True):
                    with self.subTest(extension=extension, gpu=gpu):
                        miner = cuFPGrowth(path, 2, use_gds=False, use_gpu_io=gpu)
                        miner.mine()
                        self.assertEqual(miner.getPatterns(), expected)

    def test_exports_and_repeated_mining(self):
        miner = cuFPGrowth([[1, 2], [2, 3], [1, 2, 3]], 1)
        miner.mine()
        expected = miner.getPatterns()
        miner.mine()
        self.assertEqual(miner.getPatterns(), expected)
        frame = miner.getPatternsAsDataFrame()
        self.assertEqual(dict(zip(frame.Patterns, frame.Support)), expected)
        self.assertGreaterEqual(miner.getRuntime(), 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "patterns.txt"
            miner.save(path)
            saved = dict((key, int(count)) for key, count in (
                line.rsplit(" : ", 1) for line in path.read_text().splitlines()
            ))
            self.assertEqual(saved, expected)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "transactions.csv"
            output = Path(folder) / "patterns.txt"
            source.write_text("1,2\n1\n2\n")
            result = subprocess.run(
                [sys.executable, "cupami/frequentPattern/cuFPGrowth.py", str(source), str(output), "2"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text(), "1 : 2\n2 : 2\n")

    def test_strict_gds_input(self):
        try:
            import kvikio.cufile_driver as driver
            if not driver.get("is_gds_available"):
                self.skipTest("GDS unavailable")
        except (ImportError, RuntimeError) as error:
            self.skipTest(str(error))
        rows = [[0, 1, 2], [0, 1], [1, 2]]
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            root = Path(folder)
            path = root / "transactions.bin"
            convertFile(rows, path)
            config = root / "cufile.json"
            config.write_text(json.dumps({"properties": {"allow_compat_mode": False},
                                          "logging": {"level": "ERROR", "dir": folder}}))
            code = '''
import json,sys
from cupami import cuFPGrowth
try:
    miner=cuFPGrowth(sys.argv[1],1,use_gds=True)
except RuntimeError as error:
    print(str(error));sys.exit(77)
assert miner.getInputBackend()["cufile_allows_compatibility"] is False
miner.mine()
print(json.dumps(miner.getPatterns()))
'''
            result = subprocess.run([sys.executable, "-c", code, str(path)],
                                    env=dict(os.environ, CUFILE_ENV_PATH_JSON=str(config)),
                                    capture_output=True, text=True, timeout=30)
            if result.returncode == 77:
                self.skipTest(result.stdout.strip())
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), brute_force(rows, 1))


if __name__ == "__main__":
    unittest.main()
