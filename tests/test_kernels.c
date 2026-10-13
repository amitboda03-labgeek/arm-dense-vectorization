#define _POSIX_C_SOURCE 200112L

#include "dense.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    dense_kernel_f32 function;
    int requires_alignment_shape;
} kernel_case;

static uint32_t next_random(uint32_t *state)
{
    uint32_t value = *state;
    value ^= value << 13U;
    value ^= value >> 17U;
    value ^= value << 5U;
    *state = value;
    return value;
}

static float random_float(uint32_t *state)
{
    const uint32_t bits = next_random(state) >> 8U;
    return ((float)bits / 16777216.0f) - 0.5f;
}

static void *allocate_floats(size_t count)
{
    void *memory = NULL;

    if (count > SIZE_MAX / sizeof(float)) {
        return NULL;
    }
    if (posix_memalign(&memory, DENSE_ALIGNMENT_BYTES, count * sizeof(float)) != 0) {
        return NULL;
    }
    return memory;
}

static void reference_dense(const float *input, const float *weights, const float *bias,
                            float *output, size_t batch, size_t input_features,
                            size_t output_features)
{
    for (size_t b = 0; b < batch; ++b) {
        for (size_t o = 0; o < output_features; ++o) {
            double sum = (double)bias[o];
            for (size_t k = 0; k < input_features; ++k) {
                sum += (double)input[b * input_features + k] *
                       (double)weights[k * output_features + o];
            }
            output[b * output_features + o] = (float)sum;
        }
    }
}

static int compare_outputs(const float *actual, const float *expected, size_t count,
                           const char *kernel_name, const char *shape_name)
{
    for (size_t i = 0; i < count; ++i) {
        const float absolute_error = fabsf(actual[i] - expected[i]);
        const float tolerance = 5.0e-5f + 2.0e-5f * fabsf(expected[i]);

        if (!isfinite(actual[i]) || absolute_error > tolerance) {
            fprintf(stderr,
                    "FAIL %s on %s at %zu: got %.9g, expected %.9g, error %.9g\n",
                    kernel_name, shape_name, i, actual[i], expected[i], absolute_error);
            return 0;
        }
    }
    return 1;
}

static int run_shape(size_t batch, size_t input_features, size_t output_features,
                     const char *shape_name)
{
    static const kernel_case kernels[] = {
        {"ijk-baseline", dense_ijk_baseline_f32, 0},
        {"ijk-restrict", dense_ijk_restrict_f32, 0},
        {"ikj", dense_ikj_f32, 0},
        {"ikj-restrict", dense_ikj_restrict_f32, 0},
        {"ikj-aligned", dense_ikj_aligned_f32, 1},
        {"ikj-aligned-scalar", dense_ikj_aligned_scalar_f32, 1},
    };
    const size_t input_count = batch * input_features;
    const size_t weight_count = input_features * output_features;
    const size_t output_count = batch * output_features;
    const int aligned_shape = input_features % DENSE_FLOATS_PER_ALIGNMENT == 0 &&
                              output_features % DENSE_FLOATS_PER_ALIGNMENT == 0;
    float *input = allocate_floats(input_count);
    float *weights = allocate_floats(weight_count);
    float *bias = allocate_floats(output_features);
    float *expected = allocate_floats(output_count);
    float *actual = allocate_floats(output_count);
    uint32_t random_state = 0xC001D00Du;
    int passed = 1;

    if (input == NULL || weights == NULL || bias == NULL || expected == NULL ||
        actual == NULL) {
        fprintf(stderr, "FAIL %s: allocation failed\n", shape_name);
        passed = 0;
        goto cleanup;
    }

    for (size_t i = 0; i < input_count; ++i) {
        input[i] = random_float(&random_state);
    }
    for (size_t i = 0; i < weight_count; ++i) {
        weights[i] = random_float(&random_state);
    }
    for (size_t i = 0; i < output_features; ++i) {
        bias[i] = random_float(&random_state);
    }

    reference_dense(input, weights, bias, expected, batch, input_features,
                    output_features);

    for (size_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); ++i) {
        if (kernels[i].requires_alignment_shape && !aligned_shape) {
            continue;
        }

        memset(actual, 0xA5, output_count * sizeof(float));
        kernels[i].function(input, weights, bias, actual, batch, input_features,
                            output_features);
        if (!compare_outputs(actual, expected, output_count, kernels[i].name,
                             shape_name)) {
            passed = 0;
        }
    }

cleanup:
    free(input);
    free(weights);
    free(bias);
    free(expected);
    free(actual);
    return passed;
}

int main(void)
{
    int passed = 1;

    if (!run_shape(1, 16, 16, "tiny-aligned-1x16x16")) {
        passed = 0;
    }
    if (!run_shape(3, 17, 19, "odd-tail-3x17x19")) {
        passed = 0;
    }
    if (!run_shape(4, 64, 48, "batched-aligned-4x64x48")) {
        passed = 0;
    }

    if (!passed) {
        return EXIT_FAILURE;
    }

    puts("PASS: 6 kernels matched the double-accumulation reference");
    puts("PASS: odd tail handling covered every non-alignment-contract kernel");
    return EXIT_SUCCESS;
}
