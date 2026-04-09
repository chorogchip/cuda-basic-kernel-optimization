#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../../../common/culib.h"
#include <cuda_runtime.h>

namespace {

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

constexpr int WARP_P_BLOCK = MY_BLOCKDIM / 32;

struct Data {
    size_t n;
    float* h_s;
    float* h_d;
    float* d_s;
    float* d_d;
    float* d_b;
} data_;

__global__ void kernel_prefix(
    float* dest_blocksums,
    float* dest_partial,
    const float* src,
    size_t n
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float val = idx < n ? src[idx] : 0.0f;

    for (int offset = 1; offset <= 16; offset <<= 1) {
        float tmp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += tmp;
        }
    }

    __shared__ float mem_block[WARP_P_BLOCK];
    if (lane == 31) {
        mem_block[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        float warp_sum = lane < WARP_P_BLOCK ? mem_block[lane] : 0.0f;
        for (int offset = 1; offset <= 16; offset <<= 1) {
            float tmp = __shfl_up_sync(0xffffffff, warp_sum, offset);
            if (lane >= offset) {
                warp_sum += tmp;
            }
        }

        if (lane == 31) {
            dest_blocksums[blockIdx.x] = warp_sum;
        }

        warp_sum = __shfl_up_sync(0xffffffff, warp_sum, 1);
        if (lane == 0) {
            warp_sum = 0.0f;
        }
        if (lane < WARP_P_BLOCK) {
            mem_block[lane] = warp_sum;
        }
    }
    __syncthreads();

    val += mem_block[warp];
    if (idx < n) {
        dest_partial[idx] = val;
    }
}

__global__ void kernel_sum(float* buf, const float* src_block, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n && blockIdx.x > 0) {
        buf[idx] += src_block[blockIdx.x - 1];
    }
}

void exec_problem_inner(float* dest_block, float* dest, float* src, size_t n) {
    size_t grid_dim = (n + static_cast<size_t>(MY_BLOCKDIM) - 1) / static_cast<size_t>(MY_BLOCKDIM);
    kernel_prefix<<<grid_dim, MY_BLOCKDIM>>>(dest_block, dest, src, n);
    CHECK_CUDA(cudaGetLastError());

    if (grid_dim <= 1) {
        return;
    }

    exec_problem_inner(src + n, dest + n, dest_block, grid_dim);

    kernel_sum<<<grid_dim, MY_BLOCKDIM>>>(dest, dest + n, n);
    CHECK_CUDA(cudaGetLastError());
}

}  // namespace

size_t get_work_items() {
    return data_.n;
}

void init_problem(size_t n) {
    data_.n = n;
    data_.h_s = my_cuda_opt::gen_sparse_buf(n);
    data_.h_d = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.d_s = nullptr;
    data_.d_d = nullptr;
    data_.d_b = nullptr;
    if (data_.h_s == nullptr || data_.h_d == nullptr) {
        if (data_.h_d == nullptr) {
            std::fprintf(stderr, "Host allocation failed for prefix_scan.output (count=%zu)\n", n);
        }
        std::free(data_.h_s);
        std::free(data_.h_d);
        std::exit(1);
    }

    const size_t bytes = n * sizeof(float);
    CHECK_CUDA(cudaMalloc(&data_.d_s, bytes * 2));
    CHECK_CUDA(cudaMalloc(&data_.d_d, bytes * 2));
    CHECK_CUDA(cudaMalloc(&data_.d_b, bytes * 2));
    CHECK_CUDA(cudaMemcpy(data_.d_s, data_.h_s, bytes, cudaMemcpyHostToDevice));
}

void exec_problem() {
    exec_problem_inner(data_.d_b, data_.d_d, data_.d_s, data_.n);
}

bool validate_problem() {
    CHECK_CUDA(cudaMemcpy(data_.h_d, data_.d_d, data_.n * sizeof(float), cudaMemcpyDeviceToHost));
    if (data_.n == 0) {
        return true;
    }

    float sum = data_.h_s[0];
    if (std::abs(sum - data_.h_d[0]) > 0.0001f * static_cast<float>(data_.n)) {
        return false;
    }

    for (size_t i = 1; i < data_.n; ++i) {
        sum += data_.h_s[i];
        if (std::abs(sum - data_.h_d[i]) > 0.0001f * static_cast<float>(data_.n)) {
            return false;
        }
    }
    return true;
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_s));
    CHECK_CUDA(cudaFree(data_.d_d));
    CHECK_CUDA(cudaFree(data_.d_b));
    std::free(data_.h_s);
    std::free(data_.h_d);
}
