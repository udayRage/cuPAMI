import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

from cupami import convertFile, cuApriori, cuEclat


def brute_force(rows, threshold):
    baskets = [set(str(item).strip() for item in row if str(item).strip()) for row in rows]
    baskets = [row for row in baskets if row]
    items = sorted(set().union(*baskets)) if baskets else []
    patterns = {}
    for length in range(1, len(items) + 1):
        for pattern in combinations(items, length):
            count = sum(set(pattern).issubset(row) for row in baskets)
            if count >= threshold:
                patterns["\t".join(pattern)] = count
    return patterns


class EclatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cuEclat._getCudaModule()
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(str(error))

    def test_randomized_supports_against_independent_reference(self):
        for seed in range(5):
            rng = random.Random(seed)
            rows = [rng.sample(range(8), rng.randrange(9)) for _ in range(33 + seed)]
            for support in (1, 7, 0.25, 1.0, 100):
                with self.subTest(seed=seed, support=support):
                    miner = cuEclat(rows, support)
                    miner.mine()
                    self.assertEqual(miner.getPatterns(), brute_force(rows, miner._minSup))
                    apriori = cuApriori(rows, support)
                    apriori.mine()
                    self.assertEqual(miner.getPatterns(), apriori.getPatterns())

    def test_dense_patterns_and_batched_intersections(self):
        rows = [list(range(8))] * 65
        miner = cuEclat(rows, 1)
        miner._PAIR_BATCH_SIZE = 3
        miner._INTERSECTION_BATCH_BYTES = 12  # one row per temporary batch
        miner.mine()
        self.assertEqual(len(miner.getPatterns()), 255)
        self.assertEqual(miner.getPatterns(), brute_force(rows, 1))
        self.assertIn("0\t1\t2\t3\t4\t5\t6\t7", miner.getPatterns())

    def test_empty_and_duplicate_baskets_and_large_ids(self):
        for rows in ([], [[], []], [[0, 0, 1], [], [1, 4294967295], [0, 1, 4294967295]]):
            with self.subTest(rows=rows):
                miner = cuEclat(rows, 1)
                miner.mine()
                self.assertEqual(miner.getPatterns(), brute_force(rows, 1))

    def test_csv_bin_and_parquet_with_cpu_and_gpu_loading(self):
        rows = [[0, 1, 2], [0, 1], [1, 2], [0, 1, 2], [2, 3]]
        expected = brute_force(rows, 2)
        with tempfile.TemporaryDirectory() as folder:
            for extension in ("csv", "bin", "parquet"):
                path = Path(folder) / f"transactions.{extension}"
                convertFile(rows, path)
                for gpu in (False, True):
                    with self.subTest(extension=extension, gpu=gpu):
                        miner = cuEclat(path, 2, use_gpu_io=gpu, use_gds=False)
                        miner.mine()
                        self.assertEqual(miner.getPatterns(), expected)

    def test_tabbed_url_and_quoted_item_labels(self):
        data = b'"apple,pear"\tbanana\r\nbanana\tcherry\r\n'
        with patch("cupami.frequentPattern.cuApriori.urlopen", return_value=io.BytesIO(data)):
            miner = cuEclat("https://example.test/transactions.csv", 1, sep="\t")
        miner.mine()
        self.assertEqual(miner.getPatterns(), brute_force(
            [["apple,pear", "banana"], ["banana", "cherry"]], 1
        ))

    def test_output_apis_and_repeated_mining(self):
        miner = cuEclat([[1, 2], [2, 3], [1, 2, 3]], 1)
        miner.mine()
        expected = miner.getPatterns()
        miner.mine()
        self.assertEqual(miner.getPatterns(), expected)
        self.assertGreaterEqual(miner.getRuntime(), 0)
        self.assertGreaterEqual(miner.getMemoryRSS(), 0)
        self.assertGreaterEqual(miner.getMemoryUSS(), 0)
        frame = miner.getPatternsAsDataFrame()
        self.assertEqual(dict(zip(frame.Patterns, frame.Support)), expected)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "patterns.txt"
            miner.save(path)
            saved = dict((key, int(count)) for key, count in (
                line.rsplit(" : ", 1) for line in path.read_text().splitlines()
            ))
            self.assertEqual(saved, expected)

    def test_required_gds_in_separate_strict_process(self):
        try:
            import kvikio.cufile_driver as driver
            if not driver.get("is_gds_available"):
                self.skipTest("GDS unavailable")
        except (ImportError, RuntimeError) as error:
            self.skipTest(str(error))
        rows = [[0, 1, 2], [0, 1], [1, 2], [0, 1, 2]]
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            root = Path(folder)
            path = root / "transactions.bin"
            convertFile(rows, path)
            config = root / "cufile.json"
            config.write_text(json.dumps({
                "properties": {"allow_compat_mode": False},
                "logging": {"level": "ERROR", "dir": str(root)},
            }))
            code = '''
import json,sys
from cupami import cuEclat
try:
    miner=cuEclat(sys.argv[1],1,use_gds=True)
except RuntimeError as error:
    print(str(error)); sys.exit(77)
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

    def test_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "transactions.csv"
            output = Path(folder) / "patterns.txt"
            source.write_text("1,2\n1\n2\n")
            result = subprocess.run(
                [sys.executable, "cupami/frequentPattern/cuEclat.py", str(source), str(output), "2"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text(), "1 : 2\n2 : 2\n")


if __name__ == "__main__":
    unittest.main()
