#include "culib.h"

int main(int argc, const char** argv) {
    size_t n = 0;
    my_cuda_opt::ImplMode mode = my_cuda_opt::ImplMode::kProblem;
    if (!my_cuda_opt::parse_args(argc, argv, &n, &mode)) {
        return 1;
    }

    init_problem(n);
    const bool use_baseline = mode == my_cuda_opt::ImplMode::kBaseline;
    void (*exec_impl)() = use_baseline ? exec_baseline : exec_problem;
    bool (*validate_impl)() = use_baseline ? validate_baseline : validate_problem;

    exec_impl();

    if (!validate_impl()) {
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
        exec_impl();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_ms += elapsed_ms;
    }

    const double metric_value =
        static_cast<double>(get_work_items()) * my_cuda_opt::kBenchmarkIterations * 1000.0 / total_ms;

    std::printf(
        "%s\timpl=%s\t%s\tN %10zu\tLOGN %zu\t%s\t%.6f\tTOTALMS %.6f\n",
        MY_CUDA_SOURCE_NAME,
        my_cuda_opt::impl_mode_name(mode),
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
