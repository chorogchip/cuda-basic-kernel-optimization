#include <iostream>
#include <cuda_runtime.h>

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

#ifndef MY_READPERTHREAD
#define MY_READPERTHREAD 4
#endif

__global__ void kernel_reduce_sum_partial(
        float* dest, const float* src, int n) {

    float local_sum = 0.0f;

    for (int i = 0; i < MY_READPERTHREAD; ++i) {
        size_t target = (size_t)i * gridDim.x * MY_BLOCKDIM
            + blockIdx.x * MY_BLOCKDIM
            + threadIdx.x;
        if (target < n)
            local_sum += src[target];
    }

    __shared__ float buf[MY_BLOCKDIM];
    buf[threadIdx.x] = local_sum;

    for (int i = MY_BLOCKDIM/2; i > 0; i >>= 1) {
        __syncthreads();
        if (threadIdx.x < i)
            buf[threadIdx.x] += buf[threadIdx.x + i];
    }

    if (threadIdx.x == 0)
        dest[blockIdx.x] = buf[0];
}

__global__ void kernel_reduce_sum_final(
        float* dest, const float* src, int n) {
    
}

int main(int argc, const char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: ./prog {n}\n");
        exit(1);
    }

    int n;
    if (sscanf(argv[1], "%d", &n) != 1) {
        fprintf(stderr, "Error: invalid number format [%s]\n", argv[1]);
        exit(1);
    }

    if (n <= 0) {
        fprintf(stderr, "Error: invalid n: [%d]", n);
        exit(1);
    }

    size_t bytes = (size_t)n * sizeof(float);

    float *h_a, *h_b;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);

    for (int i = 0; i < n; ++i)
        h_a[i] = (float)rand() / (float)RAND_MAX;

    float *d_a, *d_b;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    const int blk_cnt = (n + MY_BLOCKDIM - 1) / MY_BLOCKDIM;

    kernel_reduce_sum_partial<<<blk_cnt, MY_BLOCKDIM>>>(d_b, d_a, n);
    
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

    float sum_ans = 0.0f;
    float sum_res = 0.0f;

    for (int i = 0; i < n; ++i) sum_ans += h_a[i];
    for (int i = 0; i < blk_cnt; ++i) sum_res += h_b[i];
    
    if (std::abs(sum_ans - sum_res) > 0.0001f * (float)n) {
        fprintf(stderr, "Validation Failed, ans:[%f] res:[%f]\n", sum_ans, sum_res);
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double mili = 0.0;

    const int iter_cnt = 10;
    for (int iter = 0; iter < iter_cnt; ++iter) {

        cudaEventRecord(start);
        
        kernel_reduce_sum_partial<<<blk_cnt, MY_BLOCKDIM>>>(d_b, d_a, n);

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float mili_sec = 0.0f;
        cudaEventElapsedTime(&mili_sec, start, stop);
        mili += mili_sec;
    }

    size_t flop = (size_t)n * iter_cnt;
    double flops = (double)flop * 1000.0 / mili;
    printf("%.6f %.6f\n", flops, mili);

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
            
    return 0;
}

