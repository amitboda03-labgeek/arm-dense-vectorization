#ifndef ARM_DENSE_VECTORIZATION_DENSE_H
#define ARM_DENSE_VECTORIZATION_DENSE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#define DENSE_RESTRICT __restrict
#else
#define DENSE_RESTRICT restrict
#endif

enum {
    DENSE_ALIGNMENT_BYTES = 64,
    DENSE_FLOATS_PER_ALIGNMENT = DENSE_ALIGNMENT_BYTES / (int)sizeof(float)
};

typedef void (*dense_kernel_f32)(const float *input, const float *weights,
                                 const float *bias, float *output, size_t batch,
                                 size_t input_features, size_t output_features);

/* Row-major dense layer: output[B,N] = input[B,K] * weights[K,N] + bias[N]. */
void dense_ijk_baseline_f32(const float *input, const float *weights, const float *bias,
                            float *output, size_t batch, size_t input_features,
                            size_t output_features);

void dense_ijk_restrict_f32(const float *DENSE_RESTRICT input,
                            const float *DENSE_RESTRICT weights,
                            const float *DENSE_RESTRICT bias,
                            float *DENSE_RESTRICT output, size_t batch,
                            size_t input_features, size_t output_features);

void dense_ikj_f32(const float *input, const float *weights, const float *bias,
                   float *output, size_t batch, size_t input_features,
                   size_t output_features);

void dense_ikj_restrict_f32(const float *DENSE_RESTRICT input,
                            const float *DENSE_RESTRICT weights,
                            const float *DENSE_RESTRICT bias,
                            float *DENSE_RESTRICT output, size_t batch,
                            size_t input_features, size_t output_features);

/*
 * The aligned variants require all four pointers to be 64-byte aligned and
 * input_features/output_features to be multiples of 16 floats. The buffers
 * must also be pairwise disjoint, as required by restrict.
 */
void dense_ikj_aligned_f32(const float *DENSE_RESTRICT input,
                           const float *DENSE_RESTRICT weights,
                           const float *DENSE_RESTRICT bias,
                           float *DENSE_RESTRICT output, size_t batch,
                           size_t input_features, size_t output_features);

/* Same source structure as dense_ikj_aligned_f32, built with vectorization off. */
void dense_ikj_aligned_scalar_f32(const float *DENSE_RESTRICT input,
                                  const float *DENSE_RESTRICT weights,
                                  const float *DENSE_RESTRICT bias,
                                  float *DENSE_RESTRICT output, size_t batch,
                                  size_t input_features, size_t output_features);

#ifdef __cplusplus
}
#endif

#undef DENSE_RESTRICT

#endif
