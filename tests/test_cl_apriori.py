"""OpenCL GPU integration checks against an independent exhaustive oracle."""
import importlib
import itertools
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cupami import BinaryTransactionalDatabase, clApriori, cuApriori


def oracle(rows, threshold):
    baskets = [set(str(v).strip() for v in row if str(v).strip()) for row in rows]
    baskets = [row for row in baskets if row]
    labels = sorted(set().union(*baskets)) if baskets else []
    result = {}
    for width in range(1, len(labels) + 1):
        for pattern in itertools.combinations(labels, width):
            support = sum(set(pattern) <= row for row in baskets)
            if support > 0 and support >= threshold:
                result['\t'.join(pattern)] = support
    return result


class OpenCLInputValidationTest(unittest.TestCase):
    def test_invalid_thresholds_and_device_options(self):
        for support in [0, -1, True, float('inf'), float('nan'), '2']:
            with self.assertRaises(ValueError): clApriori([['a']], support)
        for kwargs in [dict(device=-1), dict(device=True), dict(device=1.5),
                       dict(vendor=''), dict(vendor=2), dict(workGroupSize=3),
                       dict(workGroupSize=True), dict(workGroupSize=512)]:
            with self.assertRaises(ValueError): clApriori([['a']], 1, **kwargs)

    def test_package_import_does_not_require_cupy_or_pyopencl(self):
        code = 'import sys; sys.modules["cupy"]=None; sys.modules["pyopencl"]=None; from cupami import clApriori; print(clApriori.__name__)'
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('clApriori', result.stdout)

    def test_gpu_only_enumeration_prefers_amd(self):
        from types import SimpleNamespace
        from cupami.frequentPattern._opencl_bitsets import _devices
        amd = SimpleNamespace(vendor='Advanced Micro Devices, Inc.')
        nvidia = SimpleNamespace(vendor='NVIDIA Corporation')
        class Platform:
            def get_devices(self, device_type):
                self.device_type = device_type
                return [nvidia, amd]
        platform = Platform()
        cl = SimpleNamespace(get_platforms=lambda: [platform], device_type=SimpleNamespace(GPU=4))
        self.assertEqual(_devices(cl), [amd, nvidia])
        self.assertEqual(platform.device_type, 4)
        self.assertEqual(_devices(cl, 'AMD'), [amd])
        self.assertEqual(_devices(cl, 'NVIDIA'), [nvidia])


class OpenCLGPUAprioriTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            if not clApriori.listDevices(): raise RuntimeError('No OpenCL GPUs')
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(str(error))

    def check(self, rows, threshold, **kwargs):
        # A regression into CUDA readers/compute must fail even on this NVIDIA host.
        with patch.object(cuApriori, '_getCudaModule', side_effect=AssertionError('CUDA called by OpenCL miner')):
            miner = clApriori(rows, threshold, **kwargs)
            miner.mine()
            normalized = [set(str(v).strip() for v in row if str(v).strip()) for row in rows]
            count = sum(bool(row) for row in normalized)
            cutoff = threshold * count if isinstance(threshold, float) and threshold <= 1 else threshold
            self.assertEqual(miner.getPatterns(), oracle(rows, cutoff))
            miner.mine()
            self.assertEqual(miner.getPatterns(), oracle(rows, cutoff))
        self.assertEqual(miner.getComputeBackend()['backend'], 'opencl')
        self.assertIsNone(miner._backend.bits)
        return miner

    def test_randomized_supports(self):
        for seed in range(4):
            rng = random.Random(seed)
            rows = [rng.sample(range(6), rng.randrange(7)) for _ in range(35)]
            for threshold in [1, 7, .25]:
                with self.subTest(seed=seed, threshold=threshold):
                    self.check(rows, threshold)

    def test_padding_duplicates_empty_and_group_sizes(self):
        for count in [0, 1, 31, 32, 33, 63, 64, 65, 129]:
            rows = [['a','a','b',*(['c'] if i%3==0 else [])] for i in range(count)]
            self.check(rows + [[]], 1)
        for group in [1, 32, 64, 128]:
            self.check([['a','b','c']] * 70, 2, workGroupSize=group)
        self.check([['arbitrary label','a,b'],['arbitrary label','0'],['0','a,b']], 2)
        # Absolute non-integral thresholds round up, not down.
        self.check([['a'],['a'],['b']], 2.1)

    def test_formats_outputs_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = BinaryTransactionalDatabase(60, 6, avg_length=3, seed=42)
            rows = generator.create()
            for extension in ['csv','bin','parquet']:
                file = Path(directory)/('data.'+extension);generator.save(file)
                with patch.object(cuApriori, '_getCudaModule', side_effect=AssertionError('CUDA called')):
                    miner = clApriori(file, 8);miner.mine()
                self.assertEqual(miner.getPatterns(), oracle(rows,8))
                self.assertEqual(list(miner.getPatternsAsDataFrame().columns), ['Patterns','Support'])
                miner.save(Path(directory)/'patterns.txt')
                self.assertTrue((Path(directory)/'patterns.txt').stat().st_size)
                self.assertGreaterEqual(miner.getTotalRuntime(), miner.getMiningTime())
                self.assertEqual(miner.getInputBackend()['backend'], 'cpu')
            file=Path(directory)/'data.csv'
            run=subprocess.run([sys.executable,'cupami/frequentPattern/clApriori.py',str(file),
                                str(Path(directory)/'cli.txt'),'8'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn('OpenCL GPU:',run.stdout)

    def test_cuda_and_opencl_agree_on_generated_data(self):
        rows=BinaryTransactionalDatabase(200,8,avg_length=3,seed=42).create()
        cuda=cuApriori(rows,10,use_gpu_io=False);cuda.mine()
        opencl=clApriori(rows,10);opencl.mine()
        self.assertEqual(opencl.getPatterns(),cuda.getPatterns())
        self.assertEqual(opencl.getPatterns(),oracle(rows,10))

    def test_deep_patterns_with_small_batches(self):
        rows = [['a','b','c','d','e','f']] * 8
        miner = clApriori(rows, 2)
        miner._BATCH_SIZE = 2
        miner.mine()
        self.assertEqual(miner.getPatterns(), oracle(rows, 2))
        self.assertEqual(len(miner.getPatterns()), 63)

    def test_url_loading_and_vendor_errors(self):
        import io
        module=importlib.import_module('cupami.frequentPattern.cuApriori')
        with patch.object(module,'urlopen',return_value=io.BytesIO(b'a\tb\na\tc\n')):
            miner=clApriori('https://example.test/data.csv',1);miner.mine()
        self.assertEqual(miner.getPatterns(),oracle([['a','b'],['a','c']],1))
        with self.assertRaisesRegex(RuntimeError,'No OpenCL GPU'):
            clApriori([['a']],1,vendor='nonexistent-vendor').mine()
        with self.assertRaisesRegex(ValueError,'out of range'):
            clApriori([['a']],1,device=100000).mine()
        with patch('cupami.frequentPattern._opencl_bitsets._devices',return_value=[]):
            with self.assertRaisesRegex(RuntimeError,'CPU fallback is not enabled'):
                clApriori([['a']],1).mine()


if __name__ == '__main__':
    unittest.main()
