#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

unsigned long long my_rand64();
float* gen_sparse_buf(size_t n);

__global__ void kernel_prefix(float* dest, const float* src, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_in_warp = threadIdx.x % 32;
    float val = src[idx];
    for (int i = 1; i <= 16; i *= 2) {
        float temp = __shfl_up_sync(0xffffffff, val, i);
        if (idx_in_warp >= i) val += temp;
    }

    __shared__ float mem_block[32];
    int idx_of_warp = threadIdx.x / 32;
    if (idx_in_warp == 31) mem_block[(idx_of_warp + 1) % 32] = val;
    __syncthreads();

    float val2 = idx_in_warp == 0 ? 0.0f : mem_block[idx_in_warp];
    for (int i = 1; i <= 16; i *= 2) {
        float temp = __shfl_up_sync(0xffffffff, val2, i);
        if (idx_in_warp >= i) val2 += temp;
    }
    if (idx_of_warp == 0)
        mem_block[idx_in_warp] = val2;
    __syncthreads();
    
    val += mem_block[idx_of_warp];

    dest[idx] = val;
}

static struct Data {
    size_t n;
    float *h_s, *h_d, *d_s, *d_d;
} data_;

size_t get_flop() {
    return data_.n;
}

void init_problem(size_t n) {
    size_t bytes = n * sizeof(float);

    data_.n = n;
    data_.h_s = gen_sparse_buf(n);
    data_.h_d = (float*)malloc(bytes);
    cudaMalloc(&data_.d_s, bytes);
    cudaMalloc(&data_.d_d, bytes);

    cudaMemcpy(data_.d_s, data_.h_s, bytes, cudaMemcpyHostToDevice);
}

void clear_problem() {
    cudaFree(data_.d_s);
    free(data_.h_s);
    free(data_.h_d);
    cudaFree(data_.d_d);
}

bool validate_problem() {
    cudaMemcpy(data_.h_d, data_.d_d, data_.n * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = data_.h_s[0];
    for (size_t i = 1; i < data_.n; ++i) {
        sum += data_.h_s[i];
        if (std::abs(sum - data_.h_d[i]) > 0.0001f * (float)data_.n)
            return false;
    }
    return true;
}

void exec_problem() {
    kernel_prefix<<<(data_.n + 1024 - 1) / 1024, 1024>>>(data_.d_d, data_.d_s, data_.n);
}

