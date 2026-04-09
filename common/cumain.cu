#include "culib.h"

int main(int argc, const char** argv) {
    size_t n = 0;
    if (!my_cuda_opt::parse_size_t_arg(argc, argv, &n)) {
        return 1;
    }

    init_problem(n);
    exec_problem();

    if (!validate_problem()) {
        std::fprintf(stderr, "Validation failed\n");
        clear_problem();
        return 1;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double total_ms = 0.0;
    for (int iter = 0; iter < my_cuda_opt::kBenchmarkIterations; ++iter) {
        CHECK_CUDA(cudaEventRecord(start));
        exec_problem();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_ms += elapsed_ms;
    }

    const double metric_value =
        static_cast<double>(get_work_items()) * my_cuda_opt::kBenchmarkIterations * 1000.0 / total_ms;

    std::printf(
        "%s\t%s\tN %10zu\tLOGN %zu\t%s\t%.6f\tTOTALMS %.6f\n",
        MY_CUDA_SOURCE_NAME,
        MY_CUDA_BUILD_CONFIG,
        n,
        my_cuda_opt::floor_log2_ceil(n),
        MY_CUDA_METRIC_NAME,
        metric_value,
        total_ms
    );

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    clear_problem();
    return 0;
}
