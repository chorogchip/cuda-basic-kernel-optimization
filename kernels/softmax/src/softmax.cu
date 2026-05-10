#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "../../../common/culib.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace {

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

#ifndef SOFTMAX_VARIANT
#define SOFTMAX_VARIANT 1
#endif

struct Data {
    size_t n;
    float* h_in;
    float* h_out;
    float* d_in;
    float* d_out;
} data_;

struct ExpShift {
    float max_value;
    __host__ __device__ float operator()(float x) const {
        return expf(x - max_value);
    }
};

struct Normalize {
    float inv_sum;
    __host__ __device__ float operator()(float x) const {
        return x * inv_sum;
    }
};

__global__ void softmax_naive_kernel(const float* input, float* output, size_t n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    float max_value = input[0];
    for (size_t i = 1; i < n; ++i) {
        max_value = fmaxf(max_value, input[i]);
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float value = expf(input[i] - max_value);
        output[i] = value;
        sum += value;
    }

    const float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; ++i) {
        output[i] *= inv_sum;
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
        std::fprintf(stderr, "Host allocation failed for softmax buffers (count=%zu)\n", n);
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
    softmax_naive_kernel<<<1, 1>>>(data_.d_in, data_.d_out, data_.n);
    CHECK_CUDA(cudaGetLastError());
}

bool validate_problem() {
    return validate_baseline();
}

void exec_baseline() {
    thrust::device_ptr<float> input(data_.d_in);
    thrust::device_ptr<float> output(data_.d_out);
    const float max_value = *thrust::max_element(input, input + data_.n);
    thrust::transform(input, input + data_.n, output, ExpShift{max_value});
    const float sum = thrust::reduce(output, output + data_.n, 0.0f, thrust::plus<float>());
    thrust::transform(output, output + data_.n, output, Normalize{1.0f / sum});
}

bool validate_baseline() {
    CHECK_CUDA(cudaMemcpy(data_.h_out, data_.d_out, data_.n * sizeof(float), cudaMemcpyDeviceToHost));
    const float max_value = *std::max_element(data_.h_in, data_.h_in + data_.n);
    std::vector<float> expected(data_.n);
    float sum = 0.0f;
    for (size_t i = 0; i < data_.n; ++i) {
        expected[i] = std::exp(data_.h_in[i] - max_value);
        sum += expected[i];
    }
    for (size_t i = 0; i < data_.n; ++i) {
        expected[i] /= sum;
        if (std::abs(data_.h_out[i] - expected[i]) > 1.0e-5f) {
            std::fprintf(stderr, "Mismatch at %zu: got %.8f expected %.8f\n", i, data_.h_out[i], expected[i]);
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
