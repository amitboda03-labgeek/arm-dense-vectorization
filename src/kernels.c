#include "dense.h"

#if defined(__clang__) || defined(__GNUC__)
#define DENSE_NOINLINE __attribute__((noinline))
#else
#define DENSE_NOINLINE
#endif

DENSE_NOINLINE
void dense_ijk_baseline_f32(const float *input, const float *weights, const float *bias,
                            float *output, size_t batch, size_t input_features,
                            size_t output_features)
{
    for (size_t b = 0; b < batch; ++b) {
        for (size_t o = 0; o < output_features; ++o) {
            float sum = bias[o];

            for (size_t k = 0; k < input_features; ++k) {
                sum += input[b * input_features + k] * weights[k * output_features + o];
            }

            output[b * output_features + o] = sum;
        }
    }
}

DENSE_NOINLINE
void dense_ijk_restrict_f32(const float *restrict input, const float *restrict weights,
                            const float *restrict bias, float *restrict output,
                            size_t batch, size_t input_features, size_t output_features)
{
    for (size_t b = 0; b < batch; ++b) {
        for (size_t o = 0; o < output_features; ++o) {
            float sum = bias[o];

            for (size_t k = 0; k < input_features; ++k) {
                sum += input[b * input_features + k] * weights[k * output_features + o];
            }

            output[b * output_features + o] = sum;
        }
    }
}

DENSE_NOINLINE
void dense_ikj_f32(const float *input, const float *weights, const float *bias,
                   float *output, size_t batch, size_t input_features,
                   size_t output_features)
{
    for (size_t b = 0; b < batch; ++b) {
        const float *input_row = input + b * input_features;
        float *output_row = output + b * output_features;

        for (size_t o = 0; o < output_features; ++o) {
            output_row[o] = bias[o];
        }

        for (size_t k = 0; k < input_features; ++k) {
            const float scale = input_row[k];
            const float *weight_row = weights + k * output_features;

            for (size_t o = 0; o < output_features; ++o) {
                output_row[o] += scale * weight_row[o];
            }
        }
    }
}

DENSE_NOINLINE
void dense_ikj_restrict_f32(const float *restrict input, const float *restrict weights,
                            const float *restrict bias, float *restrict output,
                            size_t batch, size_t input_features, size_t output_features)
{
    for (size_t b = 0; b < batch; ++b) {
        const float *restrict input_row = input + b * input_features;
        float *restrict output_row = output + b * output_features;

        for (size_t o = 0; o < output_features; ++o) {
            output_row[o] = bias[o];
        }

        for (size_t k = 0; k < input_features; ++k) {
            const float scale = input_row[k];
            const float *restrict weight_row = weights + k * output_features;

            for (size_t o = 0; o < output_features; ++o) {
                output_row[o] += scale * weight_row[o];
            }
        }
    }
}

DENSE_NOINLINE
void dense_ikj_aligned_f32(const float *restrict input, const float *restrict weights,
                           const float *restrict bias, float *restrict output,
                           size_t batch, size_t input_features, size_t output_features)
{
    const float *restrict aligned_input =
        (const float *)__builtin_assume_aligned(input, DENSE_ALIGNMENT_BYTES);
    const float *restrict aligned_weights =
        (const float *)__builtin_assume_aligned(weights, DENSE_ALIGNMENT_BYTES);
    const float *restrict aligned_bias =
        (const float *)__builtin_assume_aligned(bias, DENSE_ALIGNMENT_BYTES);
    float *restrict aligned_output =
        (float *)__builtin_assume_aligned(output, DENSE_ALIGNMENT_BYTES);

    for (size_t b = 0; b < batch; ++b) {
        const float *restrict input_row = (const float *)__builtin_assume_aligned(
            aligned_input + b * input_features, DENSE_ALIGNMENT_BYTES);
        float *restrict output_row = (float *)__builtin_assume_aligned(
            aligned_output + b * output_features, DENSE_ALIGNMENT_BYTES);

        for (size_t o = 0; o < output_features; ++o) {
            output_row[o] = aligned_bias[o];
        }

        for (size_t k = 0; k < input_features; ++k) {
            const float scale = input_row[k];
            const float *restrict weight_row = (const float *)__builtin_assume_aligned(
                aligned_weights + k * output_features, DENSE_ALIGNMENT_BYTES);

            for (size_t o = 0; o < output_features; ++o) {
                output_row[o] += scale * weight_row[o];
            }
        }
    }
}
