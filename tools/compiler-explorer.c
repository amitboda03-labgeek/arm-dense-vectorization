/*
 * Self-contained Compiler Explorer probe.
 *
 * Suggested compiler: ARM64 Clang 23 (or newer)
 * Suggested options:
 *   -O3 -ffp-contract=fast -mcpu=neoverse-v2 -msve-vector-bits=256
 *
 * Remove -msve-vector-bits=256 to compare LLVM's default NEON cost-model
 * choice. The full, tested implementation lives in src/kernels.c.
 */

#include <stddef.h>

#if defined(__clang__) || defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

NOINLINE
void dense_ikj_aligned_ce(const float *restrict input, const float *restrict weights,
                          const float *restrict bias, float *restrict output,
                          size_t batch, size_t input_features, size_t output_features)
{
    const float *restrict aligned_input =
        (const float *)__builtin_assume_aligned(input, 64);
    const float *restrict aligned_weights =
        (const float *)__builtin_assume_aligned(weights, 64);
    const float *restrict aligned_bias =
        (const float *)__builtin_assume_aligned(bias, 64);
    float *restrict aligned_output = (float *)__builtin_assume_aligned(output, 64);

    for (size_t b = 0; b < batch; ++b) {
        const float *restrict input_row = (const float *)__builtin_assume_aligned(
            aligned_input + b * input_features, 64);
        float *restrict output_row =
            (float *)__builtin_assume_aligned(aligned_output + b * output_features, 64);

        for (size_t o = 0; o < output_features; ++o) {
            output_row[o] = aligned_bias[o];
        }

        for (size_t k = 0; k < input_features; ++k) {
            const float scale = input_row[k];
            const float *restrict weight_row = (const float *)__builtin_assume_aligned(
                aligned_weights + k * output_features, 64);

            for (size_t o = 0; o < output_features; ++o) {
                output_row[o] += scale * weight_row[o];
            }
        }
    }
}
