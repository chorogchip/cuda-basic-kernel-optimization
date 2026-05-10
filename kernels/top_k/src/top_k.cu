#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "../../../common/culib.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

namespace {

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

#ifndef MY_TOPK
#define MY_TOPK 16
#endif

#ifndef TOP_K_VARIANT
#define TOP_K_VARIANT 1
#endif

struct Data {
    size_t n;
    float* h_in;
    float* h_topk;
    float* d_in;
    float* d_work;
    float* d_topk;
} data_;

__global__ void top_k_rank_naive_kernel(const float* input, float* topk, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const float value = input[idx];
    size_t rank = 0;
    for (size_t i = 0; i < n; ++i) {
        const float other = input[i];
        if (other > value || (other == value && i < idx)) {
            ++rank;
        }
    }
    if (rank < static_cast<size_t>(MY_TOPK)) {
        topk[rank] = value;
    }
}

}  // namespace

size_t get_work_items() {
    return data_.n;
}

void init_problem(size_t n) {
    data_.n = n;
    data_.h_in = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.h_topk = static_cast<float*>(std::malloc(MY_TOPK * sizeof(float)));
    data_.d_in = nullptr;
    data_.d_work = nullptr;
    data_.d_topk = nullptr;
    if (data_.h_in == nullptr || data_.h_topk == nullptr) {
        std::fprintf(stderr, "Host allocation failed for top_k buffers (count=%zu)\n", n);
        std::exit(1);
    }
    for (size_t i = 0; i < n; ++i) {
        data_.h_in[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
    CHECK_CUDA(cudaMalloc(&data_.d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&data_.d_work, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&data_.d_topk, MY_TOPK * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(data_.d_in, data_.h_in, n * sizeof(float), cudaMemcpyHostToDevice));
}

void exec_problem() {
    CHECK_CUDA(cudaMemset(data_.d_topk, 0, MY_TOPK * sizeof(float)));
    const int grid = static_cast<int>((data_.n + MY_BLOCKDIM - 1) / MY_BLOCKDIM);
    top_k_rank_naive_kernel<<<grid, MY_BLOCKDIM>>>(data_.d_in, data_.d_topk, data_.n);
    CHECK_CUDA(cudaGetLastError());
}

bool validate_problem() {
    return validate_baseline();
}

void exec_baseline() {
    CHECK_CUDA(cudaMemcpy(data_.d_work, data_.d_in, data_.n * sizeof(float), cudaMemcpyDeviceToDevice));
    thrust::device_ptr<float> work(data_.d_work);
    thrust::sort(work, work + data_.n, thrust::greater<float>());
    const size_t k = std::min(static_cast<size_t>(MY_TOPK), data_.n);
    CHECK_CUDA(cudaMemcpy(data_.d_topk, data_.d_work, k * sizeof(float), cudaMemcpyDeviceToDevice));
}

bool validate_baseline() {
    const size_t k = std::min(static_cast<size_t>(MY_TOPK), data_.n);
    CHECK_CUDA(cudaMemcpy(data_.h_topk, data_.d_topk, k * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> expected(data_.h_in, data_.h_in + data_.n);
    std::sort(expected.begin(), expected.end(), std::greater<float>());
    for (size_t i = 0; i < k; ++i) {
        if (std::abs(data_.h_topk[i] - expected[i]) > 1.0e-6f) {
            std::fprintf(stderr, "Mismatch at top-k index %zu: got %.8f expected %.8f\n", i, data_.h_topk[i], expected[i]);
            return false;
        }
    }
    return true;
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_in));
    CHECK_CUDA(cudaFree(data_.d_work));
    CHECK_CUDA(cudaFree(data_.d_topk));
    std::free(data_.h_in);
    std::free(data_.h_topk);
}
