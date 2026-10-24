"""Float64 fuzzy support kernels adapted from PAMI cuFFIMiner (GPL-3.0).

Copyright (C) 2021 Rage Uday Kiran. PAMI credits its CUDA implementation to
Mithun Thangaraj. Source revision and paper credits: README.md#references.
"""
from functools import lru_cache
import numpy as np

_SOURCE = r'''
extern "C" __global__ void singleton_supports(
    const double* values, const long long* offsets, double* supports) {
    unsigned int item = blockIdx.x, tid = threadIdx.x;
    __shared__ double partial[256];
    double total = 0.0;
    for (long long i = offsets[item] + tid; i < offsets[item + 1]; i += blockDim.x)
        total += values[i];
    partial[tid] = total;
    __syncthreads();
    for (unsigned int step = blockDim.x / 2; step; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        __syncthreads();
    }
    if (tid == 0) supports[item] = partial[0];
}
extern "C" __global__ void intersect_memberships(
    const double* prefix, const double* extensions, unsigned long long n,
    double* output, double* supports) {
    unsigned int item = blockIdx.x, tid = threadIdx.x;
    unsigned long long offset = (unsigned long long)item * n;
    __shared__ double partial[256];
    double total = 0.0;
    for (unsigned long long t = tid; t < n; t += blockDim.x) {
        double value = fmin(prefix[t], extensions[offset + t]);
        output[offset + t] = value;
        total += value;
    }
    partial[tid] = total;
    __syncthreads();
    for (unsigned int step = blockDim.x / 2; step; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        __syncthreads();
    }
    if (tid == 0) supports[item] = partial[0];
}
'''


@lru_cache(maxsize=None)
def _kernels(device):
    import cupy as cp
    module = cp.RawModule(code=_SOURCE, options=('--std=c++11',))
    return tuple(module.get_function(name) for name in ('singleton_supports', 'intersect_memberships'))


def singleton_supports(cp, columns, values, items):
    counts = cp.bincount(columns.astype(cp.int64), minlength=items)
    offsets = cp.concatenate((cp.zeros(1, dtype=cp.int64), cp.cumsum(counts, dtype=cp.int64)))
    grouped = values[cp.argsort(columns)]
    supports = cp.empty(items, dtype=cp.float64)
    if items:
        _kernels(cp.cuda.runtime.getDevice())[0]((items,), (256,), (grouped, offsets, supports))
    return cp.asnumpy(supports).tolist()


def intersect_memberships(cp, prefix, extensions):
    output = cp.empty_like(extensions)
    supports = cp.empty(len(extensions), dtype=cp.float64)
    if len(extensions):
        _kernels(cp.cuda.runtime.getDevice())[1]((len(extensions),), (256,), (
            prefix, extensions, np.uint64(extensions.shape[1]), output, supports,
        ))
    return output, cp.asnumpy(supports).tolist()
