"""Bitset kernels adapted from udayRage/cuda_pami (GPL-3.0).

Source: src/algorithms/frequent/fpminer-bitset.py, commit 2ed352a.
https://github.com/udayRage/cuda_pami
Uses zero based indices and 64 bit transaction/support counts in cuPAMI.
"""

from functools import lru_cache

import numpy as np


_SOURCE = r'''
extern "C" __global__ void build_bitsets(
    unsigned long long count, const unsigned long long* rows,
    const unsigned int* columns, unsigned long long words,
    unsigned int* bitsets) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    atomicOr(&bitsets[(unsigned long long)columns[i] * words + rows[i] / 32],
             1u << (rows[i] % 32));
}

extern "C" __global__ void count_supports(
    const unsigned int* bitsets, unsigned long long words,
    const unsigned int* candidates, unsigned int width,
    unsigned long long* supports) {
    unsigned int candidate = blockIdx.x;
    unsigned int tid = threadIdx.x;
    __shared__ unsigned long long partial[256];
    unsigned long long support = 0;
    for (unsigned long long word = tid; word < words; word += blockDim.x) {
        unsigned int intersection = 0xffffffffu;
        for (unsigned int j = 0; j < width; ++j) {
            unsigned int item = candidates[(unsigned long long)candidate * width + j];
            intersection &= bitsets[(unsigned long long)item * words + word];
        }
        support += __popc(intersection);
    }
    partial[tid] = support;
    __syncthreads();
    for (unsigned int step = blockDim.x / 2; step; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        __syncthreads();
    }
    if (tid == 0) supports[candidate] = partial[0];
}
'''


@lru_cache(maxsize=None)
def _kernels(device):
    import cupy as cp

    module = cp.RawModule(code=_SOURCE, options=("--std=c++11",))
    return module.get_function("build_bitsets"), module.get_function("count_supports")


def build_bitsets(cp, rows, columns, transactions, items):
    words = (transactions + 31) // 32
    bitsets = cp.zeros((items, words), dtype=cp.uint32)
    if rows.size:
        build, _ = _kernels(cp.cuda.runtime.getDevice())
        build(((rows.size + 255) // 256,), (256,), (
            np.uint64(rows.size), rows.astype(cp.uint64, copy=False),
            columns.astype(cp.uint32, copy=False), np.uint64(words), bitsets,
        ))
    return bitsets


def count_supports(cp, bitsets, candidates):
    if not len(candidates):
        return []
    indexes = cp.asarray(candidates, dtype=cp.uint32)
    supports = cp.zeros(len(candidates), dtype=cp.uint64)
    _, count = _kernels(cp.cuda.runtime.getDevice())
    count((len(candidates),), (256,), (
        bitsets, np.uint64(bitsets.shape[1]), indexes,
        np.uint32(indexes.shape[1]), supports,
    ))
    return cp.asnumpy(supports).tolist()
