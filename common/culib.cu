#include "culib.h"

namespace my_cuda_opt {

namespace {
unsigned long long g_rand_state = 88172645463325252ULL;
}

unsigned long long xorshift64(unsigned long long* state) {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return *state;
}

unsigned long long next_rand64() {
    return xorshift64(&g_rand_state);
}

float* gen_sparse_buf(size_t n) {
    float* ret = static_cast<float*>(std::malloc(n * sizeof(float)));
    if (ret == nullptr) {
        std::fprintf(stderr, "Host allocation failed for sparse_input (count=%zu)\n", n);
        return nullptr;
    }

    for (size_t i = 0; i < n; ++i) {
        unsigned long long r = next_rand64();
        if (r % n < 10000) {
            ret[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        } else {
            ret[i] = 0.0f;
        }
    }

    return ret;
}

}  // namespace my_cuda_opt
