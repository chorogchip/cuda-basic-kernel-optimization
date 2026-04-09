#ifndef CUDA_PORTFOLIO_CULIB_H
#define CUDA_PORTFOLIO_CULIB_H

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace my_cuda_opt {

constexpr int kBenchmarkIterations = 10;

inline void check_cuda(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA error at %s:%d: %s failed with %s\n",
            file,
            line,
            expr,
            cudaGetErrorString(err)
        );
        std::exit(1);
    }
}

inline bool parse_size_t_arg(int argc, const char** argv, size_t* out_n) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return false;
    }

    size_t n = 0;
    if (std::sscanf(argv[1], "%zu", &n) != 1) {
        std::fprintf(stderr, "Error: invalid number format [%s]\n", argv[1]);
        return false;
    }

    if (n == 0) {
        std::fprintf(stderr, "Error: invalid n: [%zu]\n", n);
        return false;
    }

    *out_n = n;
    return true;
}

inline size_t floor_log2_ceil(size_t n) {
    size_t logn = 0;
    for (size_t nn = 1; nn < n; nn <<= 1U) {
        ++logn;
    }
    return logn;
}

unsigned long long next_rand64();
unsigned long long xorshift64(unsigned long long* state);
float* gen_sparse_buf(size_t n);

}  // namespace my_cuda_opt

size_t get_work_items();
void init_problem(size_t n);
void exec_problem();
bool validate_problem();
void clear_problem();

#define CHECK_CUDA(expr) my_cuda_opt::check_cuda((expr), #expr, __FILE__, __LINE__)

#ifndef MY_CUDA_SOURCE_NAME
#define MY_CUDA_SOURCE_NAME "unknown_source"
#endif

#ifndef MY_CUDA_BUILD_CONFIG
#define MY_CUDA_BUILD_CONFIG "default"
#endif

#ifndef MY_CUDA_METRIC_NAME
#define MY_CUDA_METRIC_NAME "elem_per_sec"
#endif

#endif
