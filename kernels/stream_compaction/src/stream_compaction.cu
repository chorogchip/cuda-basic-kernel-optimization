#include <cstdio>
#include <cstdlib>

#include "../../../common/culib.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace {

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

#ifndef STREAM_COMPACTION_VARIANT
#define STREAM_COMPACTION_VARIANT 1
#endif

struct Data {
    size_t n;
    int* h_in;
    int* h_out;
    int* d_in;
    int* d_out;
    int* d_count;
    void* d_temp_storage;
    size_t temp_storage_bytes;
} data_;

struct IsPositive {
    __host__ __device__ bool operator()(int x) const {
        return x > 0;
    }
};

__global__ void stream_compaction_naive_kernel(const int* input, int* output, int* count, size_t n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int out_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (input[i] > 0) {
            output[out_idx] = input[i];
            ++out_idx;
        }
    }
    *count = out_idx;
}

}  // namespace

size_t get_work_items() {
    return data_.n;
}

void init_problem(size_t n) {
    data_.n = n;
    data_.h_in = static_cast<int*>(std::malloc(n * sizeof(int)));
    data_.h_out = static_cast<int*>(std::malloc(n * sizeof(int)));
    data_.d_in = nullptr;
    data_.d_out = nullptr;
    data_.d_count = nullptr;
    data_.d_temp_storage = nullptr;
    data_.temp_storage_bytes = 0;
    if (data_.h_in == nullptr || data_.h_out == nullptr) {
        std::fprintf(stderr, "Host allocation failed for stream_compaction buffers (count=%zu)\n", n);
        std::exit(1);
    }
    for (size_t i = 0; i < n; ++i) {
        data_.h_in[i] = static_cast<int>(std::rand() % 1024) - 512;
    }
    CHECK_CUDA(cudaMalloc(&data_.d_in, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&data_.d_out, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&data_.d_count, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(data_.d_in, data_.h_in, n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cub::DeviceSelect::If(
        nullptr,
        data_.temp_storage_bytes,
        data_.d_in,
        data_.d_out,
        data_.d_count,
        data_.n,
        IsPositive()
    ));
    CHECK_CUDA(cudaMalloc(&data_.d_temp_storage, data_.temp_storage_bytes));
}

void exec_problem() {
    CHECK_CUDA(cudaMemset(data_.d_out, 0, data_.n * sizeof(int)));
    CHECK_CUDA(cudaMemset(data_.d_count, 0, sizeof(int)));
    stream_compaction_naive_kernel<<<1, 1>>>(data_.d_in, data_.d_out, data_.d_count, data_.n);
    CHECK_CUDA(cudaGetLastError());
}

bool validate_problem() {
    return validate_baseline();
}

void exec_baseline() {
    CHECK_CUDA(cudaMemset(data_.d_count, 0, sizeof(int)));
    CHECK_CUDA(cub::DeviceSelect::If(
        data_.d_temp_storage,
        data_.temp_storage_bytes,
        data_.d_in,
        data_.d_out,
        data_.d_count,
        data_.n,
        IsPositive()
    ));
}

bool validate_baseline() {
    int count = 0;
    CHECK_CUDA(cudaMemcpy(&count, data_.d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(data_.h_out, data_.d_out, static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost));
    int expected_count = 0;
    for (size_t i = 0; i < data_.n; ++i) {
        if (data_.h_in[i] > 0) {
            if (data_.h_out[expected_count] != data_.h_in[i]) {
                std::fprintf(stderr, "Mismatch at compacted index %d\n", expected_count);
                return false;
            }
            ++expected_count;
        }
    }
    return count == expected_count;
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_in));
    CHECK_CUDA(cudaFree(data_.d_out));
    CHECK_CUDA(cudaFree(data_.d_count));
    CHECK_CUDA(cudaFree(data_.d_temp_storage));
    std::free(data_.h_in);
    std::free(data_.h_out);
}
