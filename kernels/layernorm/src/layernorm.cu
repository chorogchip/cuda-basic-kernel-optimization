#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../../../common/culib.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace {

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

#ifndef LAYERNORM_VARIANT
#define LAYERNORM_VARIANT 1
#endif

struct Data {
    size_t n;
    float* h_in;
    float* h_out;
    float* d_in;
    float* d_out;
} data_;

struct SquareDiff {
    float mean;
    __host__ __device__ float operator()(float x) const {
        const float diff = x - mean;
        return diff * diff;
    }
};

struct LayerNormOp {
    float mean;
    float inv_std;
    __host__ __device__ float operator()(float x) const {
        return (x - mean) * inv_std;
    }
};

__global__ void layernorm_naive_kernel(const float* input, float* output, size_t n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += input[i];
    }
    const float mean = sum / static_cast<float>(n);

    float variance_sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float diff = input[i] - mean;
        variance_sum += diff * diff;
    }
    const float inv_std = rsqrtf(variance_sum / static_cast<float>(n) + 1.0e-5f);

    for (size_t i = 0; i < n; ++i) {
        output[i] = (input[i] - mean) * inv_std;
    }
}

}  // namespace

size_t get_work_items() {
    return data_.n;
}

void init_problem(size_t n) {
    data_.n = n;
    data_.h_in = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.h_out = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.d_in = nullptr;
    data_.d_out = nullptr;
    if (data_.h_in == nullptr || data_.h_out == nullptr) {
        std::fprintf(stderr, "Host allocation failed for layernorm buffers (count=%zu)\n", n);
        std::exit(1);
    }
    for (size_t i = 0; i < n; ++i) {
        data_.h_in[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
    CHECK_CUDA(cudaMalloc(&data_.d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&data_.d_out, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(data_.d_in, data_.h_in, n * sizeof(float), cudaMemcpyHostToDevice));
}

void exec_problem() {
    layernorm_naive_kernel<<<1, 1>>>(data_.d_in, data_.d_out, data_.n);
    CHECK_CUDA(cudaGetLastError());
}

bool validate_problem() {
    return validate_baseline();
}

void exec_baseline() {
    thrust::device_ptr<float> input(data_.d_in);
    thrust::device_ptr<float> output(data_.d_out);
    const float sum = thrust::reduce(input, input + data_.n, 0.0f, thrust::plus<float>());
    const float mean = sum / static_cast<float>(data_.n);
    const float variance_sum =
        thrust::transform_reduce(input, input + data_.n, SquareDiff{mean}, 0.0f, thrust::plus<float>());
    const float inv_std = rsqrtf(variance_sum / static_cast<float>(data_.n) + 1.0e-5f);
    thrust::transform(input, input + data_.n, output, LayerNormOp{mean, inv_std});
}

bool validate_baseline() {
    CHECK_CUDA(cudaMemcpy(data_.h_out, data_.d_out, data_.n * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (size_t i = 0; i < data_.n; ++i) {
        sum += data_.h_in[i];
    }
    const float mean = sum / static_cast<float>(data_.n);
    float variance_sum = 0.0f;
    for (size_t i = 0; i < data_.n; ++i) {
        const float diff = data_.h_in[i] - mean;
        variance_sum += diff * diff;
    }
    const float inv_std = 1.0f / std::sqrt(variance_sum / static_cast<float>(data_.n) + 1.0e-5f);
    for (size_t i = 0; i < data_.n; ++i) {
        const float expected = (data_.h_in[i] - mean) * inv_std;
        if (std::abs(data_.h_out[i] - expected) > 1.0e-4f) {
            std::fprintf(stderr, "Mismatch at %zu: got %.8f expected %.8f\n", i, data_.h_out[i], expected);
            return false;
        }
    }
    return true;
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_in));
    CHECK_CUDA(cudaFree(data_.d_out));
    std::free(data_.h_in);
    std::free(data_.h_out);
}
