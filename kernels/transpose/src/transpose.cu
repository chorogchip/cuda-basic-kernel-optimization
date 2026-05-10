#include <cstdio>
#include <cstdlib>

#include "../../../common/culib.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace {

#ifndef TRANSPOSE_VARIANT
#define TRANSPOSE_VARIANT 1
#endif

#if TRANSPOSE_VARIANT == 6
constexpr int kTileX = 32;
constexpr int kTileY = 32;
#else
constexpr int kTileX = 16;
constexpr int kTileY = 16;
#endif

struct Data {
    int n;
    int stride;
    size_t elem_count;
    float* h_s;
    float* h_d;
    float* d_s;
    float* d_d;
} data_;

struct TransposeWrite {
    const float* src;
    float* dest;
    int n;
    int stride;

    __host__ __device__ void operator()(size_t idx) const {
        const int y = static_cast<int>(idx / static_cast<size_t>(n));
        const int x = static_cast<int>(idx - static_cast<size_t>(y) * static_cast<size_t>(n));
        dest[y * stride + x] = src[x * stride + y];
    }
};

__global__ void kernel_mat_transpose(float* res, const float* src, int n) {
#if TRANSPOSE_VARIANT == 1
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n) {
        res[y * n + x] = src[x * n + y];
    }
#elif TRANSPOSE_VARIANT == 2
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride = n + 1;
    if (x < n && y < n) {
        res[y * stride + x] = src[x * stride + y];
    }
#elif TRANSPOSE_VARIANT == 3
    __shared__ float tile[16 * 16];

    int base_x = blockIdx.x * blockDim.x;
    int base_y = blockIdx.y * blockDim.y;
    bool valid = base_x + threadIdx.x < n && base_y + threadIdx.y < n;

    if (valid) {
        tile[threadIdx.y * 16 + threadIdx.x] =
            src[(base_x + threadIdx.y) * n + (base_y + threadIdx.x)];
    }

    __syncthreads();

    if (valid) {
        res[(base_y + threadIdx.y) * n + (base_x + threadIdx.x)] =
            tile[threadIdx.x * 16 + threadIdx.y];
    }
#elif TRANSPOSE_VARIANT == 4
    __shared__ float tile[16 * 16];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool valid = x < n && y < n;

    if (valid) {
        tile[threadIdx.y * 16 + threadIdx.x] = src[y * n + x];
    }

    __syncthreads();

    int out_x = blockIdx.y * blockDim.y + threadIdx.x;
    int out_y = blockIdx.x * blockDim.x + threadIdx.y;
    if (valid) {
        res[out_y * n + out_x] = tile[threadIdx.x * 16 + threadIdx.y];
    }
#elif TRANSPOSE_VARIANT == 5
    __shared__ float tile[16 * 17];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool valid = x < n && y < n;

    if (valid) {
        tile[threadIdx.y * 17 + threadIdx.x] = src[y * n + x];
    }

    __syncthreads();

    int out_x = blockIdx.y * blockDim.y + threadIdx.x;
    int out_y = blockIdx.x * blockDim.x + threadIdx.y;
    if (valid) {
        res[out_y * n + out_x] = tile[threadIdx.x * 17 + threadIdx.y];
    }
#elif TRANSPOSE_VARIANT == 6
    __shared__ float tile[32 * 32];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool valid = x < n && y < n;

    if (valid) {
        tile[threadIdx.y * 32 + threadIdx.x] = src[y * n + x];
    }

    __syncthreads();

    int out_x = blockIdx.y * blockDim.y + threadIdx.x;
    int out_y = blockIdx.x * blockDim.x + threadIdx.y;
    if (valid) {
        res[out_y * n + out_x] = tile[threadIdx.x * 32 + threadIdx.y];
    }
#else
#error "Unsupported TRANSPOSE_VARIANT"
#endif
}

}  // namespace

size_t get_work_items() {
    return static_cast<size_t>(data_.n) * data_.n;
}

void init_problem(size_t n_input) {
    int n = static_cast<int>(n_input);
#if TRANSPOSE_VARIANT == 2
    const int stride = n + 1;
#else
    const int stride = n;
#endif

    data_.n = n;
    data_.stride = stride;
    data_.elem_count = static_cast<size_t>(n) * stride;
    data_.h_s = static_cast<float*>(std::malloc(data_.elem_count * sizeof(float)));
    data_.h_d = static_cast<float*>(std::malloc(data_.elem_count * sizeof(float)));
    data_.d_s = nullptr;
    data_.d_d = nullptr;

    if (data_.h_s == nullptr || data_.h_d == nullptr) {
        std::fprintf(stderr, "Host allocation failed for transpose buffers (count=%zu)\n", data_.elem_count);
        std::free(data_.h_s);
        std::free(data_.h_d);
        std::exit(1);
    }

    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            data_.h_s[y * stride + x] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        }
    }

    const size_t bytes = data_.elem_count * sizeof(float);
    CHECK_CUDA(cudaMalloc(&data_.d_s, bytes));
    CHECK_CUDA(cudaMalloc(&data_.d_d, bytes));
    CHECK_CUDA(cudaMemcpy(data_.d_s, data_.h_s, bytes, cudaMemcpyHostToDevice));
}

void exec_problem() {
    dim3 block(kTileX, kTileY);
    dim3 grid(
        static_cast<unsigned int>(
            (static_cast<size_t>(data_.n) + static_cast<size_t>(block.x) - 1) / static_cast<size_t>(block.x)
        ),
        static_cast<unsigned int>(
            (static_cast<size_t>(data_.n) + static_cast<size_t>(block.y) - 1) / static_cast<size_t>(block.y)
        )
    );
    kernel_mat_transpose<<<grid, block>>>(data_.d_d, data_.d_s, data_.n);
    CHECK_CUDA(cudaGetLastError());
}

void exec_baseline() {
    thrust::counting_iterator<size_t> first(0);
    thrust::for_each(
        first,
        first + static_cast<size_t>(data_.n) * static_cast<size_t>(data_.n),
        TransposeWrite{data_.d_s, data_.d_d, data_.n, data_.stride}
    );
}

bool validate_problem() {
    CHECK_CUDA(cudaMemcpy(data_.h_d, data_.d_d, data_.elem_count * sizeof(float), cudaMemcpyDeviceToHost));
    for (int y = 0; y < data_.n; ++y) {
        for (int x = 0; x < data_.n; ++x) {
            if (data_.h_d[y * data_.stride + x] != data_.h_s[x * data_.stride + y]) {
                return false;
            }
        }
    }
    return true;
}

bool validate_baseline() {
    return validate_problem();
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_s));
    CHECK_CUDA(cudaFree(data_.d_d));
    std::free(data_.h_s);
    std::free(data_.h_d);
}
