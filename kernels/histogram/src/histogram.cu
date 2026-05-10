#include <cstdio>
#include <cstdlib>

#include "../../../common/culib.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace {

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

#ifndef MY_BIN_COUNT
#define MY_BIN_COUNT 256
#endif

#ifndef HISTOGRAM_VARIANT
#define HISTOGRAM_VARIANT 1
#endif

struct Data {
    size_t n;
    unsigned int* h_in;
    unsigned int* h_bins;
    unsigned int* d_in;
    unsigned int* d_bins;
    void* d_temp_storage;
    size_t temp_storage_bytes;
} data_;

__global__ void histogram_naive_kernel(const unsigned int* input, unsigned int* bins, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&bins[input[idx]], 1U);
    }
}

}  // namespace

size_t get_work_items() {
    return data_.n;
}

void init_problem(size_t n) {
    data_.n = n;
    data_.h_in = static_cast<unsigned int*>(std::malloc(n * sizeof(unsigned int)));
    data_.h_bins = static_cast<unsigned int*>(std::malloc(MY_BIN_COUNT * sizeof(unsigned int)));
    data_.d_in = nullptr;
    data_.d_bins = nullptr;
    data_.d_temp_storage = nullptr;
    data_.temp_storage_bytes = 0;
    if (data_.h_in == nullptr || data_.h_bins == nullptr) {
        std::fprintf(stderr, "Host allocation failed for histogram buffers (count=%zu)\n", n);
        std::exit(1);
    }
    for (size_t i = 0; i < n; ++i) {
        data_.h_in[i] = static_cast<unsigned int>(std::rand()) % MY_BIN_COUNT;
    }
    CHECK_CUDA(cudaMalloc(&data_.d_in, n * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&data_.d_bins, MY_BIN_COUNT * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(data_.d_in, data_.h_in, n * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cub::DeviceHistogram::HistogramEven(
        nullptr,
        data_.temp_storage_bytes,
        data_.d_in,
        data_.d_bins,
        MY_BIN_COUNT + 1,
        0,
        MY_BIN_COUNT,
        data_.n
    ));
    CHECK_CUDA(cudaMalloc(&data_.d_temp_storage, data_.temp_storage_bytes));
}

void exec_problem() {
    CHECK_CUDA(cudaMemset(data_.d_bins, 0, MY_BIN_COUNT * sizeof(unsigned int)));
    const int grid = static_cast<int>((data_.n + MY_BLOCKDIM - 1) / MY_BLOCKDIM);
    histogram_naive_kernel<<<grid, MY_BLOCKDIM>>>(data_.d_in, data_.d_bins, data_.n);
    CHECK_CUDA(cudaGetLastError());
}

bool validate_problem() {
    CHECK_CUDA(cudaMemcpy(data_.h_bins, data_.d_bins, MY_BIN_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int expected[MY_BIN_COUNT];
    for (int bin = 0; bin < MY_BIN_COUNT; ++bin) {
        expected[bin] = 0;
    }
    for (size_t i = 0; i < data_.n; ++i) {
        ++expected[data_.h_in[i]];
    }
    for (int bin = 0; bin < MY_BIN_COUNT; ++bin) {
        if (data_.h_bins[bin] != expected[bin]) {
            std::fprintf(stderr, "Mismatch at bin %d: got %u expected %u\n", bin, data_.h_bins[bin], expected[bin]);
            return false;
        }
    }
    return true;
}

void exec_baseline() {
    CHECK_CUDA(cudaMemset(data_.d_bins, 0, MY_BIN_COUNT * sizeof(unsigned int)));
    CHECK_CUDA(cub::DeviceHistogram::HistogramEven(
        data_.d_temp_storage,
        data_.temp_storage_bytes,
        data_.d_in,
        data_.d_bins,
        MY_BIN_COUNT + 1,
        0,
        MY_BIN_COUNT,
        data_.n
    ));
}

bool validate_baseline() {
    CHECK_CUDA(cudaMemcpy(data_.h_bins, data_.d_bins, MY_BIN_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int expected[MY_BIN_COUNT];
    for (int bin = 0; bin < MY_BIN_COUNT; ++bin) {
        expected[bin] = 0;
    }
    for (size_t i = 0; i < data_.n; ++i) {
        ++expected[data_.h_in[i]];
    }
    for (int bin = 0; bin < MY_BIN_COUNT; ++bin) {
        if (data_.h_bins[bin] != expected[bin]) {
            std::fprintf(stderr, "Mismatch at bin %d: got %u expected %u\n", bin, data_.h_bins[bin], expected[bin]);
            return false;
        }
    }
    return true;
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_in));
    CHECK_CUDA(cudaFree(data_.d_bins));
    CHECK_CUDA(cudaFree(data_.d_temp_storage));
    std::free(data_.h_in);
    std::free(data_.h_bins);
}
