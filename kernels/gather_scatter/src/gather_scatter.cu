#include <cstdio>
#include <cstdlib>

#include "../../../common/culib.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>

namespace {

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

#ifndef GATHER_SCATTER_VARIANT
#define GATHER_SCATTER_VARIANT 1
#endif

struct Data {
    size_t n;
    float* h_in;
    float* h_out;
    unsigned int* h_index;
    float* d_in;
    float* d_out;
    unsigned int* d_index;
} data_;

__global__ void gather_naive_kernel(const float* input, const unsigned int* index, float* output, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[index[idx]];
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
    data_.h_index = static_cast<unsigned int*>(std::malloc(n * sizeof(unsigned int)));
    data_.d_in = nullptr;
    data_.d_out = nullptr;
    data_.d_index = nullptr;
    if (data_.h_in == nullptr || data_.h_out == nullptr || data_.h_index == nullptr) {
        std::fprintf(stderr, "Host allocation failed for gather_scatter buffers (count=%zu)\n", n);
        std::exit(1);
    }
    for (size_t i = 0; i < n; ++i) {
        data_.h_in[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        data_.h_index[i] = static_cast<unsigned int>((i * 1315423911ULL + 2654435761ULL) % n);
    }
    CHECK_CUDA(cudaMalloc(&data_.d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&data_.d_out, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&data_.d_index, n * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(data_.d_in, data_.h_in, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data_.d_index, data_.h_index, n * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void exec_problem() {
    const int grid = static_cast<int>((data_.n + MY_BLOCKDIM - 1) / MY_BLOCKDIM);
    gather_naive_kernel<<<grid, MY_BLOCKDIM>>>(data_.d_in, data_.d_index, data_.d_out, data_.n);
    CHECK_CUDA(cudaGetLastError());
}

bool validate_problem() {
    return validate_baseline();
}

void exec_baseline() {
    CHECK_CUDA(cudaMemset(data_.d_out, 0, data_.n * sizeof(float)));
    thrust::device_ptr<unsigned int> index(data_.d_index);
    thrust::device_ptr<float> input(data_.d_in);
    thrust::device_ptr<float> output(data_.d_out);
    thrust::gather(index, index + data_.n, input, output);
}

bool validate_baseline() {
    CHECK_CUDA(cudaMemcpy(data_.h_out, data_.d_out, data_.n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < data_.n; ++i) {
        const float expected = data_.h_in[data_.h_index[i]];
        if (data_.h_out[i] != expected) {
            std::fprintf(stderr, "Mismatch at %zu: got %.8f expected %.8f\n", i, data_.h_out[i], expected);
            return false;
        }
    }
    return true;
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_in));
    CHECK_CUDA(cudaFree(data_.d_out));
    CHECK_CUDA(cudaFree(data_.d_index));
    std::free(data_.h_in);
    std::free(data_.h_out);
    std::free(data_.h_index);
}
