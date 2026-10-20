# Algorithm and implementation references

## Source mapping

| Component | Source | Relationship to cuPAMI |
|---|---|---|
| Apriori candidate generation and subset pruning | Agrawal and Srikant, VLDB 1994, Section 2.1 [1] | Algorithmic basis for `cuApriori._generateCandidates` and the level-wise mining loop. The Python implementation is local code, not the authors' published source code. |
| Eclat vertical intersections and prefix equivalence classes | Zaki et al., KDD 1997 [7]; Zaki, IEEE TKDE 2000 [8] | Algorithmic basis for `cuEclat.py`. This local CUDA adaptation uses batched pair preprocessing and depth-first scheduling of prefix classes, retaining intermediate bitsets. No C++ code was copied from the author's repository [9]; diffsets and maximal-itemset variants are not implemented. |
| FP-growth | Han, Pei and Yin, SIGMOD 2000 [10], Algorithms 1–2 and Lemma 3.2 | `cuFPGrowth.py` and `_fptree.py` implement weighted FP-trees, conditional pattern bases, and single-path enumeration. Tree arrays and weighted operations use CuPy; CPU code controls the search. Parallel grouping of equal prefixes replaces sequential node insertion. This is local implementation work, not copied author code or a reproduction of a published GPU FP-growth implementation. |
| CUDA bitset construction and support counting | `udayRage/cuda_pami`, `fpminer-bitset.py`, revision `2ed352a92190bf45455f61925f76b2363b7a7590` [2] | `_bitsets.py` adapts its atomic bitset construction and AND/popcount support counting. cuPAMI changes indexing, kernel interfaces, and support/transaction widths. |
| cuDF load/split/explode processing | Same repository and revision [2] | `_gpu_io.py` adapts the processing approach, while retaining cuPAMI's list-column Parquet schema, duplicate handling, and text fallback. |
| Direct storage-to-GPU input | NVIDIA GPUDirect Storage and cuFile documentation [3, 4] | GDS is an NVIDIA I/O technology. It is not an Apriori mining algorithm or a new algorithm proposed by cuPAMI. |
| Python cuFile interface and fallback controls | RAPIDS KvikIO documentation and source [5, 6] | `cuApriori._tryBinaryDeviceRead` uses `kvikio.CuFile.pread` and runtime settings. This integration was written locally using those APIs; it is not copied from a published mining paper. |
| Non-GDS BIN loading and conversion | Local cuPAMI implementation | Bulk NumPy reads, CUDA transfers, the cuPAMI BIN layout, and `fileConverter.py` are local implementation work. They are not attributed to the Apriori paper. |

The bitset support counter is a GPU implementation choice within an Apriori
candidate-search procedure. The 1994 paper should not be cited as the source
of CUDA kernels, bitsets, cuDF, or GDS. No specific published paper has been
identified as the source of the predecessor repository's bitset code; its
repository is the direct implementation reference.

## References

1. Rakesh Agrawal and Ramakrishnan Srikant. **Fast Algorithms for Mining
   Association Rules.** Proceedings of the 20th International Conference on
   Very Large Data Bases (VLDB), 1994, pp. 487–499.
   [Publisher-hosted paper](https://www.vldb.org/conf/1994/P487.PDF).
   cuPAMI implements frequent itemset mining, not the subsequent generation
   of confidence-based association rules.

2. `udayRage/cuda_pami`. **CUDA bitset-based frequent pattern miner.**
   `src/algorithms/frequent/fpminer-bitset.py`, revision
   `2ed352a92190bf45455f61925f76b2363b7a7590`, GPL-3.0.
   [Pinned source](https://github.com/udayRage/cuda_pami/blob/2ed352a92190bf45455f61925f76b2363b7a7590/src/algorithms/frequent/fpminer-bitset.py).
   This is a software reference, not a publication citation.

3. NVIDIA. **GPUDirect Storage Overview Guide.**
   [Official documentation](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html).
   Defines the storage/GPU data path and CPU compatibility fallback.

4. NVIDIA. **GDS cuFile API Reference.**
   [Official API documentation](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html).
   Reference for cuFile operations and configuration. cuPAMI accesses cuFile
   through KvikIO rather than directly calling its C API.

5. RAPIDS. **KvikIO: High Performance File I/O.**
   [Project and Python examples](https://github.com/rapidsai/kvikio),
   [API documentation](https://docs.rapids.ai/api/kvikio/stable/api/).
   Reference for `CuFile.pread` into a device buffer.

6. RAPIDS. **KvikIO Runtime Settings.**
   [Compatibility mode and other runtime controls](https://docs.rapids.ai/api/kvikio/stable/runtime_settings/).
   KvikIO's compatibility mode and cuFile's compatibility mode are separate.
   Disabling only KvikIO's fallback does not prove a direct storage-to-GPU read.

7. Mohammed J. Zaki, Srinivasan Parthasarathy, Mitsunori Ogihara, and Wei Li.
   **New Algorithms for Fast Discovery of Association Rules.** KDD 1997,
   pp. 283–286. [Author-hosted paper](https://www.cs.rpi.edu/~zaki/PaperDir/KDD97.pdf).
   Reference for prefix equivalence classes, vertical tidset intersections,
   and pair-support preprocessing. cuPAMI includes preprocessing in its runtime.

8. Mohammed J. Zaki. **Scalable Algorithms for Association Mining.**
   IEEE Transactions on Knowledge and Data Engineering, 12(3), 2000,
   pp. 372–390. DOI: 10.1109/69.846291.
   [Author-hosted paper](https://www.cs.rpi.edu/~zaki/PaperDir/TKDE00.pdf).
   Expanded treatment of vertical mining and equivalence-class decomposition.

9. Mohammed J. Zaki. **Original ECLAT implementation.**
   [Author's source repository](https://github.com/zakimjz/ECLAT).
   A related implementation reference; cuPAMI does not copy its C++ source
   and does not claim to reproduce all options in that implementation.

10. Jiawei Han, Jian Pei, and Yiwen Yin. **Mining Frequent Patterns without
    Candidate Generation.** Proceedings of ACM SIGMOD, 2000, pp. 1–12.
    DOI: 10.1145/342009.335372.
    [Paper](https://cse.hkust.edu.hk/~raywong/comp5331/References/MiningFrequentPatternsWithoutCandidateGeneration.pdf).
    Algorithm 1 defines FP-tree construction; Algorithm 2 defines FP-growth;
    Lemma 3.2 gives single-path enumeration and minimum-count support.

## GDS verification and experimental scope

The inspected environment on 2026-10-20 uses KvikIO 25.08.00, GDS 1.15.1.6,
libcufile 2.12, and nvidia-fs 2.29. Online documentation may describe newer
versions; record installed versions with each experiment.

Local verification used files on `/tmp`'s NVMe/ext4 filesystem, with
`data=ordered`, KvikIO compatibility mode OFF, and cuFile
`properties.allow_compat_mode=false` in a process-specific configuration.
A 4 MiB read was checked byte-for-byte. BIN mining was subsequently checked
against non-GDS loading for identical bitsets, item labels, patterns, and
support counts. `/home`'s RAID10 path failed strict direct-I/O registration.

The BIN header is still inspected on the CPU. GDS concerns the bulk data path;
it does not mean every operation executes on the GPU. Benchmark observations
are local experimental results, not performance guarantees from the cited
paper or NVIDIA documentation.

References accessed 2026-10-20.

## Temporal periodic-frequent miners

The support/maximum-period definition and the boundary-gap convention follow:

- R. Uday Kiran, Masaru Kitsuregawa, P. Krishna Reddy,
  **Efficient Discovery of Periodic-Frequent Patterns in Very Large Databases**,
  Journal of Systems and Software 112 (2016), 110–121.
  [Author manuscript](https://www.tkl.iis.u-tokyo.ac.jp/new/uploads/publication_file/file/744/5-8-2015_final.pdf),
  Section 1, Example 1 and Property 1. Periodicity includes the initial and
  final gaps; failed periodicity permits pruning all supersets.

Implementation mapping:

- `cuPFApriori`: the Apriori search cited above, augmented with the periodic
  constraint. Local CUDA adaptation, not a separately published algorithm.
- `cuPFEclat`: Eclat-style depth-first prefix classes over vertical occurrence
  bitsets. Candidate intersections are recomputed from singleton bitsets;
  this version does not retain each child's intersections. Local adaptation
  using the Eclat references above and the periodicity definition above.
- `cuPFGrowth`: based on Han et al.'s FP-growth, with GPU timestamp bitsets
  and periodicity pruning using the definition above.
- `_temporal.py`: new local CUDA kernel for exact support and maximum gaps,
  including initial/final gaps. It scans set bits in timestamp order and uses
  int64 timestamps and uint64 supports. No code is copied from the paper.

For irregular timestamps this library ends the observation window at the
largest timestamp, optionally extended by `endTime`. Fractional `maxPer`
uses that window length, while fractional support uses the number of records.
Repeated timestamps count as separate transactions. These explicit API
choices extend the paper's illustrated unit-spaced transaction convention.
Temporal readers currently parse CSV/text on the CPU; GDS and the existing
transactional BIN/Parquet schema are not used for temporal inputs.
