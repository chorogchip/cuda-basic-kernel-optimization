#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

#include "../../../common/culib.h"
#include <cusparse.h>
#include <cuda_runtime.h>

#define CHECK_CUSPARSE(expr)                                                                          \
    do {                                                                                              \
        cusparseStatus_t status__ = (expr);                                                           \
        if (status__ != CUSPARSE_STATUS_SUCCESS) {                                                    \
            std::fprintf(stderr, "cuSPARSE error at %s:%d: %s\n", __FILE__, __LINE__, #expr);          \
            std::exit(1);                                                                             \
        }                                                                                             \
    } while (0)

namespace {

#ifndef MY_BLOCKDIM
#define MY_BLOCKDIM 256
#endif

#ifndef MY_ROW_NNZ
#define MY_ROW_NNZ 8
#endif

#ifndef SPMV_VARIANT
#define SPMV_VARIANT 1
#endif

struct Data {
    size_t rows;
    size_t nnz;
    int* h_row_offsets;
    int* h_col_indices;
    float* h_values;
    float* h_x;
    float* h_y;
    int* d_row_offsets;
    int* d_col_indices;
    float* d_values;
    float* d_x;
    float* d_y;
    cusparseHandle_t handle;
    cusparseSpMatDescr_t mat;
    cusparseDnVecDescr_t vec_x;
    cusparseDnVecDescr_t vec_y;
    void* d_buffer;
    size_t buffer_size;
} data_;

__global__ void spmv_naive_kernel(
    const int* row_offsets,
    const int* col_indices,
    const float* values,
    const float* x,
    float* y,
    size_t rows
) {
    const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    float sum = 0.0f;
    for (int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx) {
        sum += values[idx] * x[col_indices[idx]];
    }
    y[row] = sum;
}

}  // namespace

size_t get_work_items() {
    return data_.nnz;
}

void init_problem(size_t n) {
    data_.rows = n;
    data_.nnz = n * MY_ROW_NNZ;
    data_.h_row_offsets = static_cast<int*>(std::malloc((n + 1) * sizeof(int)));
    data_.h_col_indices = static_cast<int*>(std::malloc(data_.nnz * sizeof(int)));
    data_.h_values = static_cast<float*>(std::malloc(data_.nnz * sizeof(float)));
    data_.h_x = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.h_y = static_cast<float*>(std::malloc(n * sizeof(float)));
    data_.d_row_offsets = nullptr;
    data_.d_col_indices = nullptr;
    data_.d_values = nullptr;
    data_.d_x = nullptr;
    data_.d_y = nullptr;
    data_.handle = nullptr;
    data_.mat = nullptr;
    data_.vec_x = nullptr;
    data_.vec_y = nullptr;
    data_.d_buffer = nullptr;
    data_.buffer_size = 0;
    if (data_.h_row_offsets == nullptr || data_.h_col_indices == nullptr || data_.h_values == nullptr ||
        data_.h_x == nullptr || data_.h_y == nullptr) {
        std::fprintf(stderr, "Host allocation failed for spmv buffers (rows=%zu)\n", n);
        std::exit(1);
    }
    for (size_t row = 0; row <= n; ++row) {
        data_.h_row_offsets[row] = static_cast<int>(row * MY_ROW_NNZ);
    }
    for (size_t i = 0; i < data_.nnz; ++i) {
        data_.h_col_indices[i] = static_cast<int>(i % n);
        data_.h_values[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
    for (size_t i = 0; i < n; ++i) {
        data_.h_x[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
    CHECK_CUDA(cudaMalloc(&data_.d_row_offsets, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&data_.d_col_indices, data_.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&data_.d_values, data_.nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&data_.d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&data_.d_y, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(data_.d_row_offsets, data_.h_row_offsets, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data_.d_col_indices, data_.h_col_indices, data_.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data_.d_values, data_.h_values, data_.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data_.d_x, data_.h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUSPARSE(cusparseCreate(&data_.handle));
    CHECK_CUSPARSE(cusparseCreateCsr(
        &data_.mat,
        static_cast<int64_t>(data_.rows),
        static_cast<int64_t>(data_.rows),
        static_cast<int64_t>(data_.nnz),
        data_.d_row_offsets,
        data_.d_col_indices,
        data_.d_values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    ));
    CHECK_CUSPARSE(cusparseCreateDnVec(&data_.vec_x, static_cast<int64_t>(data_.rows), data_.d_x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&data_.vec_y, static_cast<int64_t>(data_.rows), data_.d_y, CUDA_R_32F));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        data_.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        data_.mat,
        data_.vec_x,
        &beta,
        data_.vec_y,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &data_.buffer_size
    ));
    CHECK_CUDA(cudaMalloc(&data_.d_buffer, data_.buffer_size));
}

void exec_problem() {
    const int grid = static_cast<int>((data_.rows + MY_BLOCKDIM - 1) / MY_BLOCKDIM);
    spmv_naive_kernel<<<grid, MY_BLOCKDIM>>>(
        data_.d_row_offsets,
        data_.d_col_indices,
        data_.d_values,
        data_.d_x,
        data_.d_y,
        data_.rows
    );
    CHECK_CUDA(cudaGetLastError());
}

bool validate_problem() {
    return validate_baseline();
}

void exec_baseline() {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUDA(cudaMemset(data_.d_y, 0, data_.rows * sizeof(float)));
    CHECK_CUSPARSE(cusparseSpMV(
        data_.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        data_.mat,
        data_.vec_x,
        &beta,
        data_.vec_y,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        data_.d_buffer
    ));
}

bool validate_baseline() {
    CHECK_CUDA(cudaMemcpy(data_.h_y, data_.d_y, data_.rows * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t row = 0; row < data_.rows; ++row) {
        float expected = 0.0f;
        for (int idx = data_.h_row_offsets[row]; idx < data_.h_row_offsets[row + 1]; ++idx) {
            expected += data_.h_values[idx] * data_.h_x[data_.h_col_indices[idx]];
        }
        if (std::abs(data_.h_y[row] - expected) > 1.0e-4f) {
            std::fprintf(stderr, "Mismatch at row %zu: got %.8f expected %.8f\n", row, data_.h_y[row], expected);
            return false;
        }
    }
    return true;
}

void clear_problem() {
    CHECK_CUSPARSE(cusparseDestroyDnVec(data_.vec_y));
    CHECK_CUSPARSE(cusparseDestroyDnVec(data_.vec_x));
    CHECK_CUSPARSE(cusparseDestroySpMat(data_.mat));
    CHECK_CUSPARSE(cusparseDestroy(data_.handle));
    CHECK_CUDA(cudaFree(data_.d_buffer));
    CHECK_CUDA(cudaFree(data_.d_row_offsets));
    CHECK_CUDA(cudaFree(data_.d_col_indices));
    CHECK_CUDA(cudaFree(data_.d_values));
    CHECK_CUDA(cudaFree(data_.d_x));
    CHECK_CUDA(cudaFree(data_.d_y));
    std::free(data_.h_row_offsets);
    std::free(data_.h_col_indices);
    std::free(data_.h_values);
    std::free(data_.h_x);
    std::free(data_.h_y);
}
