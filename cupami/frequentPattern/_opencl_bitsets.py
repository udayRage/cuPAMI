"""OpenCL translation of cuPAMI's packed-bitset Apriori kernels.

Original CUDA bitsets: udayRage/cuda_pami, GPL-3.0-or-later, revision 2ed352a.
https://github.com/udayRage/cuda_pami
GPU counting design: GPApriori, CLUSTER 2011, doi:10.1109/CLUSTER.2011.61.
Local OpenCL port: Mithun Thangaraj, supervised by Rage Uday Kiran.

No CUDA, warp-size assumptions, or 64-bit atomic extensions are required.
"""
import numpy as np

_SOURCE = r'''
__kernel void build_bitsets(ulong count, __global const ulong* rows,
    __global const uint* columns, ulong words, __global uint* bits) {
    size_t i = get_global_id(0);
    if (i < count)
        atomic_or((volatile __global uint*)&bits[(ulong)columns[i] * words + rows[i] / 32],
                  1u << (uint)(rows[i] % 32));
}

__kernel void count_supports(__global const uint* bits, ulong words,
    __global const uint* candidates, uint width, __global ulong* supports,
    __local ulong* partial) {
    size_t candidate = get_group_id(0);
    size_t lane = get_local_id(0);
    size_t group_size = get_local_size(0);
    ulong support = 0;
    // Adjacent workers read adjacent words, for coalesced memory access.
    for (ulong word = lane; word < words; word += group_size) {
        uint intersection = 0xffffffffu;
        for (uint j = 0; j < width; ++j)
            intersection &= bits[(ulong)candidates[candidate * width + j] * words + word];
        support += (ulong)popcount(intersection);
    }
    partial[lane] = support;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t step = group_size / 2; step; step >>= 1) {
        if (lane < step) partial[lane] += partial[lane + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) supports[candidate] = partial[0];
}
'''


def _runtime():
    try:
        import pyopencl as cl
    except ImportError as error:
        raise ImportError('clApriori requires PyOpenCL: pip install pyopencl, plus a GPU OpenCL driver') from error
    return cl


def _is_amd(vendor):
    vendor = vendor.casefold()
    return 'amd' in vendor or 'advanced micro devices' in vendor


def _devices(cl, vendor=None):
    try:
        platforms = cl.get_platforms()
    except cl.Error as error:
        raise RuntimeError('No usable OpenCL platform; install the GPU vendor OpenCL runtime') from error
    devices = []
    for platform in platforms:
        try:
            devices.extend(platform.get_devices(device_type=cl.device_type.GPU))
        except cl.Error as error:
            if error.code != cl.status_code.DEVICE_NOT_FOUND:
                raise
    # Prefer AMD when several vendors are installed; never select a CPU device.
    devices.sort(key=lambda device: not _is_amd(device.vendor))
    if vendor is not None:
        devices = [d for d in devices if (_is_amd(d.vendor) if vendor.casefold() == 'amd'
                                          else vendor.casefold() in d.vendor.casefold())]
    return devices


def list_devices(vendor=None):
    cl = _runtime()
    return [dict(index=i, name=d.name.strip(), vendor=d.vendor.strip(),
                 platform=d.platform.name.strip(), version=d.version)
            for i, d in enumerate(_devices(cl, vendor))]


class OpenCLBitsets:
    """Persistent GPU context/program; per-mine data buffers are released."""
    def __init__(self, device_index=0, vendor=None, work_group_size=None):
        self.cl = cl = _runtime()
        devices = _devices(cl, vendor)
        if not devices:
            raise RuntimeError(f'No OpenCL GPU found for vendor={vendor!r}; CPU fallback is not enabled')
        if device_index >= len(devices):
            raise ValueError(f'device={device_index} is out of range; found {len(devices)} matching OpenCL GPUs')
        self.device = devices[device_index]
        self.context = cl.Context(devices=[self.device])
        self.queue = cl.CommandQueue(self.context, device=self.device)
        self.program = cl.Program(self.context, _SOURCE).build(options=['-cl-std=CL1.2'])
        self.build_kernel = cl.Kernel(self.program, 'build_bitsets')
        self.count_kernel = cl.Kernel(self.program, 'count_supports')
        info = cl.kernel_work_group_info
        limit = min(256, self.device.max_work_group_size, self.device.max_work_item_sizes[0],
                    self.device.local_mem_size // 8,
                    self.build_kernel.get_work_group_info(info.WORK_GROUP_SIZE, self.device),
                    self.count_kernel.get_work_group_info(info.WORK_GROUP_SIZE, self.device))
        self.max_group_size = 1 << (int(limit).bit_length() - 1)
        if work_group_size is not None and work_group_size > self.max_group_size:
            raise ValueError(f'workGroupSize exceeds supported limit {self.max_group_size}')
        self.requested_group_size = work_group_size
        self.preferred_multiple = int(self.count_kernel.get_work_group_info(info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, self.device))
        self.work_group_size = None
        self.words = 0
        self.bits = None

    def metadata(self):
        return dict(backend='opencl', device=self.device.name.strip(), vendor=self.device.vendor.strip(),
                    platform=self.device.platform.name.strip(), work_group_size=self.work_group_size)

    def _choose_group_size(self, words):
        if self.requested_group_size is not None:
            return self.requested_group_size
        # Adapt to reported kernel/device limits and SIMD width, including AMD's
        # wave32/wave64 implementations; reduce idle workers on short bitsets.
        target = max(self.preferred_multiple, 1 << (max(1, words) - 1).bit_length())
        return 1 << (min(self.max_group_size, target).bit_length() - 1)

    def build(self, rows, columns, transactions, items):
        cl = self.cl
        self.words = (transactions + 31) // 32
        self.work_group_size = self._choose_group_size(self.words)
        if rows.size == 0:
            return
        size = items * self.words * np.dtype(np.uint32).itemsize
        if size > self.device.max_mem_alloc_size:
            raise MemoryError(f'Bitsets require {size} bytes, exceeding the OpenCL device allocation limit')
        flags = cl.mem_flags
        self.bits = cl.Buffer(self.context, flags.READ_WRITE, size=size)
        cl.enqueue_fill_buffer(self.queue, self.bits, np.uint32(0), 0, size)
        row_buffer = column_buffer = None
        try:
            row_buffer = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=rows)
            column_buffer = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=columns)
            group = self.work_group_size
            self.build_kernel(self.queue, (((rows.size + group - 1) // group) * group,), (group,),
                              np.uint64(rows.size), row_buffer, column_buffer, np.uint64(self.words), self.bits)
            self.queue.finish()
        finally:
            if row_buffer is not None: row_buffer.release()
            if column_buffer is not None: column_buffer.release()

    def count(self, candidates):
        if not candidates:
            return []
        cl = self.cl
        indexes = np.ascontiguousarray(candidates, dtype=np.uint32)
        supports = np.empty(len(candidates), dtype=np.uint64)
        flags = cl.mem_flags
        candidate_buffer = support_buffer = None
        try:
            candidate_buffer = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=indexes)
            support_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, size=supports.nbytes)
            group = self.work_group_size
            self.count_kernel(self.queue, (len(candidates) * group,), (group,), self.bits,
                              np.uint64(self.words), candidate_buffer, np.uint32(indexes.shape[1]),
                              support_buffer, cl.LocalMemory(group * 8))
            cl.enqueue_copy(self.queue, supports, support_buffer, is_blocking=True)
            return supports.tolist()
        finally:
            if candidate_buffer is not None: candidate_buffer.release()
            if support_buffer is not None: support_buffer.release()

    def clear(self):
        self.queue.finish()
        if self.bits is not None:
            self.bits.release()
            self.bits = None
