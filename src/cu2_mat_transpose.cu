#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel_mat_transpose(float* res, const float* src, int n) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x < n && idx_y < n) {
        res[idx_y * n + idx_x] = src[idx_x * n + idx_y];
    }
}

int main(int argc, const char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: ./prog {n}\n");
        exit(1);
    }

    int n;
    if (sscanf(argv[1], "%d", &n) != 1) {
        fprintf(stderr, "Error: invalin number format [%s]\n", argv[1]);
        exit(1);
    }

    if (n <= 0) {
        fprintf(stderr, "Error: invalid n: [%d]", n);
        exit(1);
    }

    size_t bytes = (size_t)n * n * sizeof(float);

    float *h_d, *h_s;
    h_d = (float*)malloc(bytes);
    h_s = (float*)malloc(bytes);

    for (int yy = 0; yy < n; ++yy) {
        for (int xx = 0; xx < n; ++xx) {
            h_s[yy * n + xx] = (float)rand() / (float)RAND_MAX;
        }
    }

    float *d_d, *d_s;
    cudaMalloc(&d_d, bytes);
    cudaMalloc(&d_s, bytes);

    cudaMemcpy(d_s, h_s, bytes, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((n + 16 - 1) / 16, (n + 16 - 1) / 16);

    kernel_mat_transpose<<<grid_size, block_size>>>(d_d, d_s, n);

    cudaMemcpy(h_d, d_d, bytes, cudaMemcpyDeviceToHost);

    for (int yy = 0; yy < n; ++yy) {
        for (int xx = 0; xx < n; ++xx) {
            if (h_d[yy * n + xx] != h_s[xx * n + yy]) {
                fprintf(stderr, "Validation Failed\n");
                exit(1);
            }
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double mili = 0.0;

    const int iter_cnt = 10;
    for (int iter = 0; iter < iter_cnt; ++iter) {

        cudaEventRecord(start);
        
        kernel_mat_transpose<<<grid_size, block_size>>>(d_d, d_s, n);

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);


        float mili_sec = 0.0f;
        cudaEventElapsedTime(&mili_sec, start, stop);
        mili += mili_sec;
    }

    size_t flop = (size_t)n * n * iter_cnt;
    double flops = (double)flop * 1000.0 / mili;
    printf("%.6f %.6f\n", flops, mili);

    cudaFree(d_s);
    cudaFree(d_d);
    free(h_s);
    free(h_d);
            
    return 0;
}
