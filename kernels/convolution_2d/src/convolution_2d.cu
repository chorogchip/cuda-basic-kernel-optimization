#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../../../common/culib.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace {

#ifndef MY_TILE_DIM
#define MY_TILE_DIM 16
#endif

#ifndef MY_FILTER_RADIUS
#define MY_FILTER_RADIUS 1
#endif

#ifndef CONVOLUTION_2D_VARIANT
#define CONVOLUTION_2D_VARIANT 1
#endif

struct Data {
    size_t width;
    float* h_in;
    float* h_out;
    float* h_filter;
    float* d_in;
    float* d_out;
    float* d_filter;
} data_;

struct Conv2DOp {
    const float* input;
    const float* filter;
    int width;

    __host__ __device__ float operator()(int linear_idx) const {
        const int row = linear_idx / width;
        const int col = linear_idx % width;
        float sum = 0.0f;
        for (int fr = -MY_FILTER_RADIUS; fr <= MY_FILTER_RADIUS; ++fr) {
            for (int fc = -MY_FILTER_RADIUS; fc <= MY_FILTER_RADIUS; ++fc) {
                const int in_row = row + fr;
                const int in_col = col + fc;
                if (in_row >= 0 && in_row < width && in_col >= 0 && in_col < width) {
                    const int filter_width = 2 * MY_FILTER_RADIUS + 1;
                    const int filter_idx = (fr + MY_FILTER_RADIUS) * filter_width + (fc + MY_FILTER_RADIUS);
                    sum += input[in_row * width + in_col] * filter[filter_idx];
                }
            }
        }
        return sum;
    }
};

__global__ void convolution_2d_naive_kernel(
    const float* input,
    const float* filter,
    float* output,
    int width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int elems = width * width;
    if (idx >= elems) {
        return;
    }

    const int row = idx / width;
    const int col = idx % width;
    const int filter_width = 2 * MY_FILTER_RADIUS + 1;
    float sum = 0.0f;
    for (int fr = -MY_FILTER_RADIUS; fr <= MY_FILTER_RADIUS; ++fr) {
        for (int fc = -MY_FILTER_RADIUS; fc <= MY_FILTER_RADIUS; ++fc) {
            const int in_row = row + fr;
            const int in_col = col + fc;
            if (in_row >= 0 && in_row < width && in_col >= 0 && in_col < width) {
                const int filter_idx = (fr + MY_FILTER_RADIUS) * filter_width + (fc + MY_FILTER_RADIUS);
                sum += input[in_row * width + in_col] * filter[filter_idx];
            }
        }
    }
    output[idx] = sum;
}

}  // namespace

size_t get_work_items() {
    return data_.width * data_.width;
}

void init_problem(size_t n) {
    data_.width = n;
    const size_t elems = n * n;
    const int filter_width = 2 * MY_FILTER_RADIUS + 1;
    const size_t filter_elems = static_cast<size_t>(filter_width * filter_width);
    data_.h_in = static_cast<float*>(std::malloc(elems * sizeof(float)));
    data_.h_out = static_cast<float*>(std::malloc(elems * sizeof(float)));
    data_.h_filter = static_cast<float*>(std::malloc(filter_elems * sizeof(float)));
    data_.d_in = nullptr;
    data_.d_out = nullptr;
    data_.d_filter = nullptr;
    if (data_.h_in == nullptr || data_.h_out == nullptr || data_.h_filter == nullptr) {
        std::fprintf(stderr, "Host allocation failed for convolution_2d buffers (width=%zu)\n", n);
        std::exit(1);
    }
    for (size_t i = 0; i < elems; ++i) {
        data_.h_in[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
    for (size_t i = 0; i < filter_elems; ++i) {
        data_.h_filter[i] = 1.0f / static_cast<float>(filter_elems);
    }
    CHECK_CUDA(cudaMalloc(&data_.d_in, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&data_.d_out, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&data_.d_filter, filter_elems * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(data_.d_in, data_.h_in, elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data_.d_filter, data_.h_filter, filter_elems * sizeof(float), cudaMemcpyHostToDevice));
}

void exec_problem() {
    const size_t elems = data_.width * data_.width;
    const int grid = static_cast<int>((elems + MY_TILE_DIM * MY_TILE_DIM - 1) / (MY_TILE_DIM * MY_TILE_DIM));
    convolution_2d_naive_kernel<<<grid, MY_TILE_DIM * MY_TILE_DIM>>>(
        data_.d_in,
        data_.d_filter,
        data_.d_out,
        static_cast<int>(data_.width)
    );
    CHECK_CUDA(cudaGetLastError());
}

bool validate_problem() {
    return validate_baseline();
}

void exec_baseline() {
    thrust::device_ptr<float> output(data_.d_out);
    const int elems = static_cast<int>(data_.width * data_.width);
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(elems),
        output,
        Conv2DOp{data_.d_in, data_.d_filter, static_cast<int>(data_.width)}
    );
}

bool validate_baseline() {
    const size_t elems = data_.width * data_.width;
    CHECK_CUDA(cudaMemcpy(data_.h_out, data_.d_out, elems * sizeof(float), cudaMemcpyDeviceToHost));
    const int width = static_cast<int>(data_.width);
    const int filter_width = 2 * MY_FILTER_RADIUS + 1;
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float expected = 0.0f;
            for (int fr = -MY_FILTER_RADIUS; fr <= MY_FILTER_RADIUS; ++fr) {
                for (int fc = -MY_FILTER_RADIUS; fc <= MY_FILTER_RADIUS; ++fc) {
                    const int in_row = row + fr;
                    const int in_col = col + fc;
                    if (in_row >= 0 && in_row < width && in_col >= 0 && in_col < width) {
                        const int filter_idx = (fr + MY_FILTER_RADIUS) * filter_width + (fc + MY_FILTER_RADIUS);
                        expected += data_.h_in[in_row * width + in_col] * data_.h_filter[filter_idx];
                    }
                }
            }
            const size_t idx = static_cast<size_t>(row) * data_.width + static_cast<size_t>(col);
            if (std::abs(data_.h_out[idx] - expected) > 1.0e-5f) {
                std::fprintf(stderr, "Mismatch at (%d,%d): got %.8f expected %.8f\n", row, col, data_.h_out[idx], expected);
                return false;
            }
        }
    }
    return true;
}

void clear_problem() {
    CHECK_CUDA(cudaFree(data_.d_in));
    CHECK_CUDA(cudaFree(data_.d_out));
    CHECK_CUDA(cudaFree(data_.d_filter));
    std::free(data_.h_in);
    std::free(data_.h_out);
    std::free(data_.h_filter);
}
