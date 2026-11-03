# Algorithm and CUDA implementation references

Each miner lists its references in its module docstring. This page maps the
CUDA techniques to their implementations and records differences from the sources.
PAMI comparison sources are pinned to
[`48d0c25`](https://github.com/UdayLab/PAMI/tree/48d0c25fb5f1e4ff32e797fa00873033921f1aa7),
with the separate `cuMFFIMiner` revision below. Existing source credits and
GPL notices remain applicable. The two multi miners and their support modules
are outside this reference update.

## Frequent patterns

### cuApriori

The GPU design follows Fan Zhang, Yan Zhang and Jason Bakos,
**GPApriori: GPU-Accelerated Frequent Itemset Mining**, CLUSTER 2011,
pp. 590–594, DOI: 10.1109/CLUSTER.2011.61.
[Author-hosted paper](https://cse.sc.edu/~jbakos/assets/papers/cluster11_paper_2.pdf),
Section IV, especially support counting and CUDA execution.

`cuApriori.mine()` retains singleton bitsets. `_bitsets.count_supports()`
intersects the singleton vectors for each complete candidate, with a CUDA block
per candidate, consecutive words assigned to adjacent threads, popcount, and
shared-memory summation. This follows the paper's complete-intersection design.
Local differences include uint64 counts, CuPy RawKernels, tuple-prefix candidate
joins with subset pruning rather than a candidate trie, and optional GPU/GDS input.

PAMI's CUDA implementations provide additional API and implementation references:
[`cuApriori`](https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/frequentPattern/cuda/cuApriori.py)
uses tid-list intersections;
[`cuAprioriBit`](https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/frequentPattern/cuda/cuAprioriBit.py)
uses packed bitsets. These are distinct CUDA representations. The original
[Apriori paper](https://www.vldb.org/conf/1994/P487.PDF) supplies the search and
subset-pruning foundation, rather than the GPU execution design.

### cuEclat

CUDA implementation reference:
[PAMI cuEclatBit](https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/frequentPattern/cuda/cuEclatBit.py).
`mine()` forms frequent pair classes; `_searchClass()` performs depth-first
vertical intersections through `_bitsets.intersect_extensions()`. Local kernels
retain each child bitset and reduce support within a block, with bounded batches.

The search follows Zaki's [Eclat treatment](https://www.cs.rpi.edu/~zaki/PaperDir/TKDE00.pdf),
IEEE TKDE 2000, DOI: 10.1109/69.846291. For GPU word intersection and reduction,
see Claudio Silvestri and Salvatore Orlando, **gpuDCI: Exploiting GPUs in Frequent
Itemset Mining**, PDP 2012, DOI: 10.1109/PDP.2012.94,
[author-hosted paper](https://www.dsi.unive.it/~orlando/PUB/gpudci2012.pdf), Section 4.
cuPAMI uses Eclat prefix classes; it does not implement DCI's search strategy.

### cuFPGrowth

GPU reference: Khedija Arour and Amani Belkahla,
**Frequent Pattern-growth Algorithm on Multi-core CPU and GPU Processors**,
CIT 22(3), 2014, pp. 159–169, DOI: 10.2498/cit.1002361.
[Journal record and paper](https://hrcak.srce.hr/en/clanak/190853), Section 5.3.

The reference motivates GPU array representations and projected pattern mining.
cuPAMI's `_fptree.py` uses its own parent/item/count arrays, level-wise GPU tree
construction, weighted ancestor counting, and conditional projection. It does
not reproduce the paper's OpenCL FP-array/list representation. CPU frames
schedule GPU conditional trees. Search and the single-path case follow
[Han, Pei and Yin](https://doi.org/10.1145/342009.335372), SIGMOD 2000.

## Periodic-frequent patterns

CUDA implementation reference for all three temporal miners and `_temporal.py`:
[PAMI gPFMinerBit](https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/periodicFrequentPattern/cuda/gPFMinerBit.py),
particularly `supportAndPeriod`. It intersects occurrence bitsets and computes
support and temporal gaps per candidate.

- `cuPFApriori` uses level-wise Apriori candidate generation.
- `cuPFEclat` uses depth-first prefix classes.
- `cuPFGrowth` combines the FP-tree GPU representation above with temporal
  acceptance checks before expanding conditional trees.

The local kernel stores bits by transaction row and reads explicit int64
timestamps, rather than allocating a bit for each elapsed time unit. This
supports sparse timelines and keeps equal-time rows as separate transactions.
It scans set bits with `__ffs`, computes exact integer gaps, and includes gaps
from timestamp zero and to the observation end. FP-growth plus this kernel is
a local hybrid, not a separately published CUDA PF-growth algorithm.

Related GPU research: Sreepada Tarun, Rage Uday Kiran, Yutaka Watanobe and
Kazuo Goda, **A Novel GPU-Accelerated Algorithm to Discover Periodic-Frequent
Patterns in Temporal Databases**, IEEE Big Data 2022, pp. 121–126,
DOI: [10.1109/BigData55660.2022.10020216](https://doi.org/10.1109/BigData55660.2022.10020216).
[PAMI cuGPFMiner](https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/periodicFrequentPattern/cuda/cuGPFMiner.py)
references this work. That paper's flattened temporal-array representation
is distinct from cuPAMI's row bitsets; these miners do not claim to reproduce it.
The [JSS 2016 paper](https://www.tkl.iis.u-tokyo.ac.jp/new/uploads/publication_file/file/744/5-8-2015_final.pdf)
defines the periodic-frequent model and boundary gaps.

## Partial-periodic patterns

CUDA implementation reference:
[PAMI gPPMiner](https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/partialPeriodicPattern/cuda/gPPMiner.py).
Both `cuPPPApriori` and `cuPPPEclat` follow its GPU bitset intersection and
temporal scanning approach in `_partial.py`. The former schedules breadth-first
levels; the latter schedules depth-first prefix classes. Local storage uses
transaction-row bits and explicit timestamps.

The public arguments and periodic-support semantics follow
[PAMI PPP_ECLAT](https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/partialPeriodicPattern/basic/PPP_ECLAT.py):
count consecutive distinct occurrence gaps at most `period`, accept
`periodicSupport >= minPS`, and exclude start/end gaps. This differs from
`gPPMiner`'s start-gap counting and strict candidate acceptance. Equal timestamps
are unioned before mining, matching timestamp-set intersections. Floating-point
thresholds and decimal strings are multiplied by the original number of rows.
See the model discussed in the
[JSS partial-periodic paper](https://www.tkl.iis.u-tokyo.ac.jp/new/uploads/publication_file/file/774/JSS_2017.pdf).

## Fuzzy patterns

- `cuFFIMiner` and `_fuzzy.py`: GPU minimum-membership intersections and
  reductions follow
  [PAMI cuFFIMiner](https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/fuzzyFrequentPattern/cuda/cuFFIMiner.py).
  Strongest-region selection and the support model come from Lin et al.,
  JIFS 2015, DOI: [10.3233/IFS-151936](https://doi.org/10.3233/IFS-151936).
- `cuMFFIMiner` uses shared base-item traversal; `cuMultiFuzzyBaseline` uses
  flat traversal. These accompany “cuMFFP-Miner: GPU-Accelerated Multiple Fuzzy
  Frequent Pattern Mining via Shared Base-Item Traversal” by Mithun Thangaraj
  and Rage Uday Kiran, The University of Aizu, Japan. Both retain frequent
  regions and exclude combinations of regions belonging to the same base item.
  The multiple fuzzy frequent itemset model is described by Lin et al.,
  IJFS 2017 (online 2016), DOI:
  [10.1007/s40815-016-0246-1](https://doi.org/10.1007/s40815-016-0246-1).

Raw quantitative amounts require a membership transformation before fuzzy mining.

## Spatial patterns

`cuSpatialEclat` follows neighbour pruning in
[PAMI SpatialECLAT](https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/georeferencedFrequentPattern/basic/SpatialECLAT.py)
and GPU intersections in PAMI cuEclatBit. It checks neighbours on the CPU and
counts support using the shared CUDA Eclat kernels. This is a local CPU/GPU
combination, not a claim that the spatial paper provides CUDA source.
The spatial model is Kiran et al., SIGSPATIAL 2020,
DOI: [10.1145/3397536.3422206](https://doi.org/10.1145/3397536.3422206).

## Shared code, generation, and formats

Existing `_bitsets.py` and `_gpu_io.py` provenance remains
[udayRage/cuda_pami, revision 2ed352a](https://github.com/udayRage/cuda_pami/blob/2ed352a92190bf45455f61925f76b2363b7a7590/src/algorithms/frequent/fpminer-bitset.py).
Research references describe the GPU design; they do not replace this source credit.

The binary and quantitative generators use cuPAMI's local uniform basket-length
sampler. Quantitative amounts are independent uniform integer draws. The temporal
generator adds local timestamp-step sampling. They are not replicas of IBM's
synthetic workload generator. No mining-paper attribution is assigned to these
local generation routines.

`fileConverter.py`, `datasetConverter.py`, `_temporal_io.py`, and
`_quantitative_io.py` implement cuPAMI's own versioned BIN schemas. Parquet
uses [Apache Arrow's nested list support](https://arrow.apache.org/docs/python/parquet.html).
GDS uses [NVIDIA cuFile](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)
through [RAPIDS KvikIO](https://docs.rapids.ai/api/kvikio/stable/api/).
Package `__init__.py` files only export these implementations.
