gPF-Miner

`gPFMiner` is a CuPy adaptation of Tarun Sreepada's public PAMI timestamp-bitset implementation. It discovers patterns meeting minimum support and maximum periodicity, including gaps from timestamp zero and to the observation end.

Paper: “A Novel GPU-Accelerated Algorithm to Discover Periodic-Frequent Patterns in Temporal Databases” by Sreepada Tarun, Rage Uday Kiran, Yutaka Watanobe, and Kazuo Goda. IEEE Big Data 2022, pp. 121–126. [Paper](https://doi.org/10.1109/BigData55660.2022.10020216).

The public bitset variant stores singleton occurrences as bits indexed by timestamp. Depth-first prefix search schedules batches of complete candidate keys. A GPU thread intersects the singleton bitsets and calculates support and maximum gap for each candidate. This is distinct from cuPAMI's row-based temporal bitsets.

The paper's flattened array is a contiguous collection of packed item bitsets. This port uses that core design with local changes: lexical item ordering instead of support-descending ranks, set-bit scanning and corrected boundary handling. It adapts the public [cuGPFMiner](https://github.com/UdayLab/PAMI/blob/2d4e7bcc2879bec8c12473e8fb82b0ac22aa3918/PAMI/periodicFrequentPattern/cuda/cuGPFMiner.py) and [gPFMinerBit](https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/periodicFrequentPattern/cuda/gPFMinerBit.py) sources. The original programs were written by Tarun Sreepada under Professor Rage Uday Kiran. cuPAMI integration and corrections are by Mithun Thangaraj under Professor Rage Uday Kiran. Original GPL notices remain applicable.

Input rows contain a timestamp followed by items. Input may be an iterable of rows, CSV, temporal BIN/Parquet, or a CSV HTTP(S) URL. Repeated timestamps are unioned, and support counts distinct occurrence timestamps. A float `minSup` in (0, 1] is still relative to the original input row count. Integer `minSup` values are absolute.

`maxPer` is the allowed maximum gap in timestamp units; a float in (0, 1] is a fraction of the timeline. `endTime` optionally extends the observation end beyond the last input timestamp. Empty rows with timestamps also extend the observation end.

```python
from cupami import TemporalDatabase, gPFMiner

generator = TemporalDatabase(100, 8, avg_length=3, seed=42)
data = generator.create()

miner = gPFMiner(data, minSup=0.1, maxPer=30)
miner.mine()
print(miner.getPatterns())
miner.save("patterns.txt")
```

`getPatterns()` maps tab-separated item strings to `(support, periodicity)` values. `getPatternsAsDataFrame()` requires Pandas. Timing methods return seconds; memory methods report process RSS/USS in bytes when psutil is installed.

Mining requires CuPy and an NVIDIA CUDA GPU. Parquet input requires PyArrow. Dense timestamp bitsets can be large on sparse timelines. `maxTimelineBytes` limits their size to 512 MiB by default, checked before allocation. Use `cuPFEclat` for large sparse timelines instead.

From a terminal:

```bash
python -m cupami.periodicFrequentPattern.gPFMiner temporal.csv patterns.txt 10 30
```
