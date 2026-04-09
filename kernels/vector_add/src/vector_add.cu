#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../../../common/culib.h"
#include <cuda_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 256;

struct Data {
    size_t n;
    float* h_a;
    float* h_b;
    float* h_c;
    float* d_a;
    float* d_b;
    float* d_c;
} data_;

__global__ void vec_add(const float* a, const float* b, float* c, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

}  // namespace

size_t get_work_items() {
    return data_.n;
}

void init_problem(size_t n) {
    data_.n = n;
    data_.h_a = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.h_b = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.h_c = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.d_a = nullptr;
    data_.d_b = nullptr;
    data_.d_c = nullptr;

    if (data_.h_a == nullptr || data_.h_b == nullptr || data_.h_c == nullptr) {
        std::fprintf(stderr, "Host allocation failed for vector_add buffers (count=%zu)\n", n);
        std::free(data_.h_a);
        std::free(data_.h_b);
        std::free(data_.h_c);
        std::exit(1);
    }

    for (size_t i = 0; i < n; ++i) {
        data_.h_a[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        data_.h_b[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    const size_t bytes = n * sizeof(float);
    CHECK_CUDA(cudaMalloc(&data_.d_a, bytes));
    CHECK_CUDA(cudaMalloc(&data_.d_b, bytes));
    CHECK_CUDA(cudaMalloc(&data_.d_c, bytes));
    CHECK_CUDA(cudaMemcpy(data_.d_a, data_.h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data_.d_b, data_.h_b, bytes, cudaMemcpyHostToDevice));
}

void exec_problem() {
    const int blocks_per_grid =
        static_cast<int>((data_.n + static_cast<size_t>(kThreadsPerBlock) - 1) / static_cast<size_t>(kThreadsPerBlock));
    vec_add<<<blocks_per_grid, kThreadsPerBlock>>>(data_.d_a, data_.d_b, data_.d_c, data_.n);
    CHECK_CUDA(cudaGetLastError());
}

bool validate_problem() {
    CHECK_CUDA(cudaMemcpy(data_.h_c, data_.d_c, data_.n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < data_.n; ++i) {
        if (std::abs(data_.h_c[i] - data_.h_a[i] - data_.h_b[i]) > 0.001f) {
            return false;
        }
    }
    return true;
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_a));
    CHECK_CUDA(cudaFree(data_.d_b));
    CHECK_CUDA(cudaFree(data_.d_c));
    std::free(data_.h_a);
    std::free(data_.h_b);
    std::free(data_.h_c);
}
