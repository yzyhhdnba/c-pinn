#include "pinn/nn/gemm.hpp"

#include <cstring>

#if defined(PINN_USE_CBLAS)
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    // Assumes a CBLAS provider (e.g., MKL/OpenBLAS) is available.
    #include <cblas.h>
  #endif
#endif

namespace pinn::nn {

int gemm(TYPE_VAL* A, TYPE_VAL* B, TYPE_VAL* C, int m, int k, int n) {
    std::memset(C, 0, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(TYPE_VAL));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            TYPE_VAL sum = 0;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    return 0;
}

int gemm_f32(float* A, float* B, float* C, int m, int k, int n) {
    std::memset(C, 0, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    return 0;
}

void gemm_mkl(float* A, float* B, float* C, int M, int K, int N) {
#if defined(PINN_USE_CBLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K,
                B, N,
                0.0f, C, N);
#else
    (void)gemm_f32(A, B, C, M, K, N);
#endif
}

}  // namespace pinn::nn
