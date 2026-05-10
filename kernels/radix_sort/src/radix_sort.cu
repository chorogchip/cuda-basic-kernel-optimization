#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "../../../common/culib.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace {

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

#ifndef MY_BITS_PER_PASS
#define MY_BITS_PER_PASS 4
#endif

#ifndef RADIX_SORT_VARIANT
#define RADIX_SORT_VARIANT 1
#endif

struct Data {
    size_t n;
    unsigned int* h_keys;
    unsigned int* h_out;
    unsigned int* d_keys_orig;
    unsigned int* d_keys;
    unsigned int* d_tmp;
    void* d_temp_storage;
    size_t temp_storage_bytes;
} data_;

__global__ void rank_sort_naive_kernel(const unsigned int* input, unsigned int* output, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const unsigned int key = input[idx];
    size_t rank = 0;
    for (size_t i = 0; i < n; ++i) {
        const unsigned int other = input[i];
        if (other < key || (other == key && i < idx)) {
            ++rank;
        }
    }
    output[rank] = key;
}

}  // namespace

size_t get_work_items() {
    return data_.n;
}

void init_problem(size_t n) {
    data_.n = n;
    data_.h_keys = static_cast<unsigned int*>(std::malloc(n * sizeof(unsigned int)));
    data_.h_out = static_cast<unsigned int*>(std::malloc(n * sizeof(unsigned int)));
    data_.d_keys_orig = nullptr;
    data_.d_keys = nullptr;
    data_.d_tmp = nullptr;
    data_.d_temp_storage = nullptr;
    data_.temp_storage_bytes = 0;
    if (data_.h_keys == nullptr || data_.h_out == nullptr) {
        std::fprintf(stderr, "Host allocation failed for radix_sort buffers (count=%zu)\n", n);
        std::exit(1);
    }
    for (size_t i = 0; i < n; ++i) {
        data_.h_keys[i] = static_cast<unsigned int>(my_cuda_opt::next_rand64());
    }
    CHECK_CUDA(cudaMalloc(&data_.d_keys_orig, n * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&data_.d_keys, n * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&data_.d_tmp, n * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(data_.d_keys_orig, data_.h_keys, n * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cub::DeviceRadixSort::SortKeys(
        nullptr,
        data_.temp_storage_bytes,
        data_.d_keys,
        data_.d_tmp,
        data_.n
    ));
    CHECK_CUDA(cudaMalloc(&data_.d_temp_storage, data_.temp_storage_bytes));
}

void exec_problem() {
    CHECK_CUDA(cudaMemset(data_.d_tmp, 0, data_.n * sizeof(unsigned int)));
    const int grid = static_cast<int>((data_.n + MY_BLOCKDIM - 1) / MY_BLOCKDIM);
    rank_sort_naive_kernel<<<grid, MY_BLOCKDIM>>>(data_.d_keys_orig, data_.d_tmp, data_.n);
    CHECK_CUDA(cudaGetLastError());
}

bool validate_problem() {
    return validate_baseline();
}

void exec_baseline() {
    CHECK_CUDA(cudaMemcpy(data_.d_keys, data_.d_keys_orig, data_.n * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cub::DeviceRadixSort::SortKeys(
        data_.d_temp_storage,
        data_.temp_storage_bytes,
        data_.d_keys,
        data_.d_tmp,
        data_.n
    ));
}

bool validate_baseline() {
    CHECK_CUDA(cudaMemcpy(data_.h_out, data_.d_tmp, data_.n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    std::sort(data_.h_keys, data_.h_keys + data_.n);
    for (size_t i = 0; i < data_.n; ++i) {
        if (data_.h_out[i] != data_.h_keys[i]) {
            std::fprintf(stderr, "Mismatch at %zu: got %u expected %u\n", i, data_.h_out[i], data_.h_keys[i]);
            return false;
        }
    }
    return true;
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_keys_orig));
    CHECK_CUDA(cudaFree(data_.d_keys));
    CHECK_CUDA(cudaFree(data_.d_tmp));
    CHECK_CUDA(cudaFree(data_.d_temp_storage));
    std::free(data_.h_keys);
    std::free(data_.h_out);
}
