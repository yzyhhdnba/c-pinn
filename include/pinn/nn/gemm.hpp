#pragma once

#include <cstddef>

namespace pinn::nn {

// Match the user's TYPE_VAL-based signature (project uses float64 Tensor currently).
using TYPE_VAL = double;

// Naive row-major GEMM with optional OpenMP on outer loop.
// C[m*n] = A[m*k] * B[k*n]
int gemm(TYPE_VAL* A, TYPE_VAL* B, TYPE_VAL* C, int m, int k, int n);

// Float32 variant (useful for optional cblas_sgemm).
int gemm_f32(float* A, float* B, float* C, int m, int k, int n);

// High-performance GEMM via CBLAS if available (Intel MKL / Accelerate / OpenBLAS).
// If CBLAS is not enabled at build time, this falls back to gemm_f32.
void gemm_mkl(float* A, float* B, float* C, int M, int K, int N);

}  // namespace pinn::nn
