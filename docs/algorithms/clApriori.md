# clApriori

Experimental Apriori for OpenCL GPUs, including AMD GPUs with a compatible vendor OpenCL runtime. It uses the same level-wise candidate generation and subset pruning as cuPAMI's CUDA Apriori.

Transaction–item positions are transferred in bulk. OpenCL kernels build packed 32-bit bitsets on the GPU, intersect them for each candidate and sum popcounts using local-memory reductions. Work-group sizes adapt to device limits, the reported preferred execution multiple and the bitset length. No fixed NVIDIA warp size or 64-bit atomics are required.

This backend has been tested through NVIDIA's OpenCL driver. AMD hardware compatibility and performance have not been measured; the implementation is intended for AMD-compatible OpenCL runtimes, not claimed to be tuned for a particular Radeon or Instinct GPU.

## Setup

Install a vendor OpenCL GPU runtime supporting OpenCL C 1.2 for your GPU and operating system, then install PyOpenCL:

```bash
python -m pip install pyopencl
```

The cuPAMI package also exposes an `opencl` installation extra. Installing PyOpenCL alone does not install an AMD GPU driver. Consult the [AMD programming documentation](https://rocm.docs.amd.com/en/latest/reference/hip-programming.html) and [PyOpenCL installation guide](https://documen.tician.de/pyopencl/misc.html) for runtime setup. This backend does not use CuPy, CUDA, NVIDIA GDS or automatic CPU fallback.

## Example

```python
from cupami import BinaryTransactionalDatabase, clApriori

print(clApriori.listDevices())

data = BinaryTransactionalDatabase(1000, 20, avg_length=5, seed=42).create()
miner = clApriori(data, minSup=0.1, vendor="AMD")
miner.mine()
print(miner.getPatterns())
print(miner.getComputeBackend())
miner.save("patterns.txt")
```

Omit `vendor="AMD"` to use the first available OpenCL GPU, preferring AMD when multiple vendors are available. `device=0` selects an index from `clApriori.listDevices(vendor="AMD")` when that filter is used. Vendor filters do not silently fall back to another vendor or CPU device.

Input may be transaction rows, CSV, BIN, Parquet, or a CSV HTTP(S) URL. Reading is done on the CPU and repeated items in a transaction count once. Empty baskets are omitted, matching cuApriori's transactional reader. Parquet input requires PyArrow.

`minSup` is a positive absolute count or a float fraction in (0, 1] of the retained transaction count. Thresholds round upward. `sep` selects the CSV delimiter; the default detects tabs or commas.

`workGroupSize` optionally sets a power of two from 1 to the selected kernel/device limit, up to 256. Leave it unset to use device-aware sizing. Candidate levels are evaluated in bounded batches; singleton bitsets stay on the GPU during a mining call and are released afterward.

`getPatterns()` returns tab-separated item strings mapped to integer supports. `getPatternsAsDataFrame()` requires Pandas. Timing methods return seconds. Loading time covers CPU input parsing; mining time includes first-call OpenCL context creation, compilation, transfers, bitset construction, counting and result downloads. Repeated `mine()` calls reuse the context and program, so first-call and warmed timings should be distinguished.

## Terminal

```bash
python -m cupami.frequentPattern.clApriori transactions.csv patterns.txt 10 --vendor AMD
```

## References and credits

Apriori: Rakesh Agrawal and Ramakrishnan Srikant, “Fast Algorithms for Mining Association Rules”, VLDB 1994, pp. 487–499. [Paper](https://www.vldb.org/conf/1994/P487.PDF).

GPU counting design: Fan Zhang, Yan Zhang and Jason Bakos, “GPApriori: GPU-Accelerated Frequent Itemset Mining”, CLUSTER 2011, pp. 590–594. [Paper](https://doi.org/10.1109/CLUSTER.2011.61).

The OpenCL kernels adapt cuPAMI's CUDA bitset kernels, with original source from [udayRage/cuda_pami](https://github.com/udayRage/cuda_pami/tree/2ed352a92190bf45455f61925f76b2363b7a7590), GPL-3.0-or-later. OpenCL port by Mithun Thangaraj under the supervision of Professor Rage Uday Kiran. This is a local OpenCL implementation, not an AMD implementation supplied by the referenced papers.
