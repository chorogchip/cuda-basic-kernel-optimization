#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
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

    size_t bytes = n * sizeof(float);

    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)rand() / (float)RAND_MAX;
        h_b[i] = (float)rand() / (float)RAND_MAX;
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    vec_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    bool succeed = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(h_c[i] - h_a[i] - h_b[i]) > 0.001f) {
            succeed = false;
            break;
        }
    }

    if (!succeed) {
        fprintf(stderr, "Validation Failed\n");
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double mili = 0.0;

    const int iter_count = 10;
    for (int iter = 0; iter < iter_count; ++iter) {
        cudaEventRecord(start);
        vec_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float mili_sec = 0.0f;
        cudaEventElapsedTime(&mili_sec, start, stop);
        mili += mili_sec;

    }

    int flop = n * iter_count;
    double flops = (double)flop * 1000.0 / mili;
    printf("%.6f %.6f\n", flops, mili);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

     


	
