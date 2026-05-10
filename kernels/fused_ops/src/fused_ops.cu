#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../../../common/culib.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <tuple>

namespace {

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

#ifndef FUSED_OPS_VARIANT
#define FUSED_OPS_VARIANT 1
#endif

struct Data {
    size_t n;
    float* h_x;
    float* h_bias;
    float* h_y;
    float* d_x;
    float* d_bias;
    float* d_y;
} data_;

struct BiasRelu {
    __host__ __device__ float operator()(const thrust::tuple<float, float>& item) const {
        const float x = thrust::get<0>(item);
        const float bias = thrust::get<1>(item);
        const float y = x + bias;
        return y > 0.0f ? y : 0.0f;
    }
};

__global__ void fused_ops_naive_kernel(const float* x, const float* bias, float* y, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float sum = x[idx] + bias[idx];
        y[idx] = sum > 0.0f ? sum : 0.0f;
    }
}

}  // namespace

size_t get_work_items() {
    return data_.n;
}

void init_problem(size_t n) {
    data_.n = n;
    data_.h_x = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.h_bias = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.h_y = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.d_x = nullptr;
    data_.d_bias = nullptr;
    data_.d_y = nullptr;
    if (data_.h_x == nullptr || data_.h_bias == nullptr || data_.h_y == nullptr) {
        std::fprintf(stderr, "Host allocation failed for fused_ops buffers (count=%zu)\n", n);
        std::exit(1);
    }
    for (size_t i = 0; i < n; ++i) {
        data_.h_x[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        data_.h_bias[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
    CHECK_CUDA(cudaMalloc(&data_.d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&data_.d_bias, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&data_.d_y, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(data_.d_x, data_.h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data_.d_bias, data_.h_bias, n * sizeof(float), cudaMemcpyHostToDevice));
}

void exec_problem() {
    const int grid = static_cast<int>((data_.n + MY_BLOCKDIM - 1) / MY_BLOCKDIM);
    fused_ops_naive_kernel<<<grid, MY_BLOCKDIM>>>(data_.d_x, data_.d_bias, data_.d_y, data_.n);
    CHECK_CUDA(cudaGetLastError());
}

bool validate_problem() {
    return validate_baseline();
}

void exec_baseline() {
    thrust::device_ptr<float> x(data_.d_x);
    thrust::device_ptr<float> bias(data_.d_bias);
    thrust::device_ptr<float> y(data_.d_y);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(x, bias)),
        thrust::make_zip_iterator(thrust::make_tuple(x + data_.n, bias + data_.n)),
        y,
        BiasRelu()
    );
}

bool validate_baseline() {
    CHECK_CUDA(cudaMemcpy(data_.h_y, data_.d_y, data_.n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < data_.n; ++i) {
        const float sum = data_.h_x[i] + data_.h_bias[i];
        const float expected = sum > 0.0f ? sum : 0.0f;
        if (std::abs(data_.h_y[i] - expected) > 1.0e-6f) {
            std::fprintf(stderr, "Mismatch at %zu: got %.8f expected %.8f\n", i, data_.h_y[i], expected);
            return false;
        }
    }
    return true;
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_x));
    CHECK_CUDA(cudaFree(data_.d_bias));
    CHECK_CUDA(cudaFree(data_.d_y));
    std::free(data_.h_x);
    std::free(data_.h_bias);
    std::free(data_.h_y);
}
