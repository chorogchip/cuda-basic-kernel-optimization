#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../../../common/culib.h"
#include <cuda_runtime.h>

namespace {

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

#ifndef MY_READPERTHREAD
#define MY_READPERTHREAD 4
#endif

#ifndef REDUCTION_VERSION
#define REDUCTION_VERSION 1
#endif

struct Data {
    size_t n;
    float* h_a;
    float* h_b;
    float* d_a;
    float* d_b;
    float* d_result;
} data_;

#if REDUCTION_VERSION == 2
unsigned long long quick_rand64(unsigned long long* state) {
    return my_cuda_opt::xorshift64(state);
}
#endif

__global__ void kernel_reduce_sum_partial(float* dest, const float* src, size_t n) {
    float local_sum = 0.0f;

    for (int i = 0; i < MY_READPERTHREAD; ++i) {
        size_t target = static_cast<size_t>(i) * gridDim.x * MY_BLOCKDIM
            + static_cast<size_t>(blockIdx.x) * MY_BLOCKDIM
            + threadIdx.x;
        if (target < n) {
            local_sum += src[target];
        }
    }

    __shared__ float buf[MY_BLOCKDIM];
    buf[threadIdx.x] = local_sum;

    for (int offset = MY_BLOCKDIM / 2; offset > 0; offset >>= 1) {
        __syncthreads();
        if (threadIdx.x < offset) {
            buf[threadIdx.x] += buf[threadIdx.x + offset];
        }
    }

    if (threadIdx.x == 0) {
        dest[blockIdx.x] = buf[0];
    }
}

#if REDUCTION_VERSION == 1
double cpu_sum_double(const float* data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}
#endif

#if REDUCTION_VERSION == 2
size_t reduce_to_one(float*& d_in, float*& d_out, size_t n) {
    const size_t read_per_block = static_cast<size_t>(MY_BLOCKDIM) * MY_READPERTHREAD;
    size_t rem = n;
    while (rem > 1) {
        size_t blk_cnt = (rem + read_per_block - 1) / read_per_block;
        kernel_reduce_sum_partial<<<blk_cnt, MY_BLOCKDIM>>>(d_out, d_in, rem);
        CHECK_CUDA(cudaGetLastError());
        std::swap(d_in, d_out);
        rem = blk_cnt;
    }
    return rem;
}
#endif

}  // namespace

size_t get_work_items() {
    return data_.n;
}

void init_problem(size_t n) {
    data_.n = n;
    data_.h_a = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.h_b = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.d_a = nullptr;
    data_.d_b = nullptr;
    data_.d_result = nullptr;
    if (data_.h_a == nullptr || data_.h_b == nullptr) {
        std::fprintf(stderr, "Host allocation failed for reduction buffers (count=%zu)\n", n);
        std::free(data_.h_a);
        std::free(data_.h_b);
        std::exit(1);
    }
    const size_t bytes = n * sizeof(float);

#if REDUCTION_VERSION == 1
    for (size_t i = 0; i < n; ++i) {
        data_.h_a[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
#elif REDUCTION_VERSION == 2
    static unsigned long long seed = 88172645463325252ULL;
    for (size_t i = 0; i < n; ++i) {
        unsigned long long r = quick_rand64(&seed);
        if ((r % n) < 10000) {
            data_.h_a[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        } else {
            data_.h_a[i] = 0.0f;
        }
    }
#else
#error "Unsupported REDUCTION_VERSION"
#endif

    CHECK_CUDA(cudaMalloc(&data_.d_a, bytes));
    CHECK_CUDA(cudaMalloc(&data_.d_b, bytes));
    CHECK_CUDA(cudaMemcpy(data_.d_a, data_.h_a, bytes, cudaMemcpyHostToDevice));
}

void exec_problem() {
#if REDUCTION_VERSION == 1
    const size_t read_per_block = static_cast<size_t>(MY_BLOCKDIM) * MY_READPERTHREAD;
    const size_t blk_cnt = (data_.n + read_per_block - 1) / read_per_block;
    kernel_reduce_sum_partial<<<blk_cnt, MY_BLOCKDIM>>>(data_.d_b, data_.d_a, data_.n);
    CHECK_CUDA(cudaGetLastError());
    data_.d_result = data_.d_b;
#elif REDUCTION_VERSION == 2
    float* d_reduce_in = data_.d_a;
    float* d_reduce_out = data_.d_b;
    reduce_to_one(d_reduce_in, d_reduce_out, data_.n);
    data_.d_result = d_reduce_in;
#endif
}

bool validate_problem() {
#if REDUCTION_VERSION == 1
    const size_t read_per_block = static_cast<size_t>(MY_BLOCKDIM) * MY_READPERTHREAD;
    const size_t blk_cnt = (data_.n + read_per_block - 1) / read_per_block;
    CHECK_CUDA(cudaMemcpy(data_.h_b, data_.d_b, blk_cnt * sizeof(float), cudaMemcpyDeviceToHost));

    double sum_ans = cpu_sum_double(data_.h_a, data_.n);
    double sum_res = 0.0;
    for (size_t i = 0; i < blk_cnt; ++i) {
        sum_res += data_.h_b[i];
    }
    return std::abs(sum_ans - sum_res) <= 0.0001 * static_cast<double>(data_.n);
#elif REDUCTION_VERSION == 2
    float device_result = 0.0f;
    CHECK_CUDA(cudaMemcpy(&device_result, data_.d_result, sizeof(float), cudaMemcpyDeviceToHost));

    float sum_ans = 0.0f;
    for (size_t i = 0; i < data_.n; ++i) {
        sum_ans += data_.h_a[i];
    }
    return std::abs(static_cast<double>(sum_ans) - static_cast<double>(device_result)) <=
        0.000001 * static_cast<double>(data_.n);
#endif
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_a));
    CHECK_CUDA(cudaFree(data_.d_b));
    std::free(data_.h_a);
    std::free(data_.h_b);
}
