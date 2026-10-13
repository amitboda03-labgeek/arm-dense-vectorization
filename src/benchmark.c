#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif

#include "dense.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

#define DENSE_STRINGIFY_INNER(value) #value
#define DENSE_STRINGIFY(value) DENSE_STRINGIFY_INNER(value)

#if defined(__clang__)
#define DENSE_COMPILER_NAME                                                            \
    "Clang " DENSE_STRINGIFY(__clang_major__) "." DENSE_STRINGIFY(                     \
        __clang_minor__) "." DENSE_STRINGIFY(__clang_patchlevel__)
#elif defined(__GNUC__)
#define DENSE_COMPILER_NAME "GCC " __VERSION__
#else
#define DENSE_COMPILER_NAME "unknown compiler"
#endif

#define MAX_SAMPLES 15
#define WARMUP_COUNT 5

#ifndef DENSE_NATIVE_FLAGS_TEXT
#define DENSE_NATIVE_FLAGS_TEXT "unknown native-target flags"
#endif

#ifndef DENSE_FP_FLAGS_TEXT
#define DENSE_FP_FLAGS_TEXT "unknown floating-point flags"
#endif

#ifndef DENSE_SCALAR_FLAGS_TEXT
#define DENSE_SCALAR_FLAGS_TEXT "unknown scalar-control flags"
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#define DENSE_REPORT_TITLE "Native Apple Silicon benchmark evidence"
#else
#define DENSE_REPORT_TITLE "Native benchmark evidence"
#endif

enum { CASE_COUNT = 3, KERNEL_COUNT = 6 };

typedef struct {
    const char *shape;
    size_t batch;
    size_t input_features;
    size_t output_features;
} benchmark_case;

typedef struct {
    const char *name;
    dense_kernel_f32 function;
} benchmark_kernel;

typedef struct {
    float *input;
    float *weights;
    float *bias;
    float *output;
    float *reference;
    size_t batch;
    size_t input_features;
    size_t output_features;
    size_t output_count;
} benchmark_data;

typedef struct {
    double seconds[MAX_SAMPLES];
    size_t iterations;
    double median_seconds;
    double mad_seconds;
    double max_absolute_error;
} kernel_result;

static const benchmark_case benchmark_cases[CASE_COUNT] = {
    {"1x256x128", 1, 256, 128},
    {"1x256x256", 1, 256, 256},
    {"4x256x256", 4, 256, 256},
};

static const benchmark_kernel kernels[KERNEL_COUNT] = {
    {"ijk-baseline", dense_ijk_baseline_f32},
    {"ijk-restrict", dense_ijk_restrict_f32},
    {"ikj", dense_ikj_f32},
    {"ikj-restrict", dense_ikj_restrict_f32},
    {"ikj-aligned-scalar", dense_ikj_aligned_scalar_f32},
    {"ikj-aligned", dense_ikj_aligned_f32},
};

static volatile double checksum_sink;

static void configure_benchmark_thread(void)
{
#if defined(__APPLE__)
    const int error = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    if (error != 0) {
        fprintf(stderr, "warning: could not request user-interactive QoS: %s\n",
                strerror(error));
    }
#endif
}

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

static int checked_product(size_t left, size_t right, size_t *result)
{
    if (left != 0 && right > SIZE_MAX / left) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static float *allocate_floats(size_t count)
{
    void *memory = NULL;
    size_t bytes = 0;

    if (!checked_product(count, sizeof(float), &bytes)) {
        return NULL;
    }
    if (posix_memalign(&memory, DENSE_ALIGNMENT_BYTES, bytes) != 0) {
        return NULL;
    }
    return memory;
}

static double now_seconds(void)
{
    struct timespec timestamp;

    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)timestamp.tv_sec + (double)timestamp.tv_nsec * 1.0e-9;
}

static int compare_double(const void *left, const void *right)
{
    const double lhs = *(const double *)left;
    const double rhs = *(const double *)right;
    return (lhs > rhs) - (lhs < rhs);
}

static double median(const double *values, int count)
{
    double sorted[MAX_SAMPLES];

    for (int i = 0; i < count; ++i) {
        sorted[i] = values[i];
    }
    qsort(sorted, (size_t)count, sizeof(sorted[0]), compare_double);
    return sorted[count / 2];
}

static double median_absolute_deviation(const double *values, int count, double center)
{
    double deviations[MAX_SAMPLES];

    for (int i = 0; i < count; ++i) {
        deviations[i] = fabs(values[i] - center);
    }
    return median(deviations, count);
}

static void reference_dense(benchmark_data *data)
{
    for (size_t b = 0; b < data->batch; ++b) {
        for (size_t o = 0; o < data->output_features; ++o) {
            double sum = (double)data->bias[o];

            for (size_t k = 0; k < data->input_features; ++k) {
                sum += (double)data->input[b * data->input_features + k] *
                       (double)data->weights[k * data->output_features + o];
            }
            data->reference[b * data->output_features + o] = (float)sum;
        }
    }
}

static int initialize_data(const benchmark_case *benchmark, benchmark_data *data)
{
    size_t input_count = 0;
    size_t weight_count = 0;
    uint32_t random_state = 0xA11CE5E5u;

    memset(data, 0, sizeof(*data));
    data->batch = benchmark->batch;
    data->input_features = benchmark->input_features;
    data->output_features = benchmark->output_features;

    if (!checked_product(data->batch, data->input_features, &input_count) ||
        !checked_product(data->input_features, data->output_features, &weight_count) ||
        !checked_product(data->batch, data->output_features, &data->output_count)) {
        return 0;
    }

    data->input = allocate_floats(input_count);
    data->weights = allocate_floats(weight_count);
    data->bias = allocate_floats(data->output_features);
    data->output = allocate_floats(data->output_count);
    data->reference = allocate_floats(data->output_count);
    if (data->input == NULL || data->weights == NULL || data->bias == NULL ||
        data->output == NULL || data->reference == NULL) {
        return 0;
    }

    for (size_t i = 0; i < input_count; ++i) {
        data->input[i] = random_float(&random_state);
    }
    for (size_t i = 0; i < weight_count; ++i) {
        data->weights[i] = random_float(&random_state);
    }
    for (size_t i = 0; i < data->output_features; ++i) {
        data->bias[i] = random_float(&random_state);
    }

    reference_dense(data);
    return 1;
}

static void destroy_data(benchmark_data *data)
{
    free(data->input);
    free(data->weights);
    free(data->bias);
    free(data->output);
    free(data->reference);
}

static double consume_output(const benchmark_data *data)
{
    double checksum = 0.0;

    for (size_t i = 0; i < data->output_count; ++i) {
        checksum += (double)data->output[i] * (double)((i % 17U) + 1U);
    }
    checksum_sink += checksum;
    return checksum;
}

static double validate_kernel(const benchmark_kernel *kernel, benchmark_data *data,
                              int *passed)
{
    double max_absolute_error = 0.0;

    kernel->function(data->input, data->weights, data->bias, data->output, data->batch,
                     data->input_features, data->output_features);

    for (size_t i = 0; i < data->output_count; ++i) {
        const double actual = (double)data->output[i];
        const double expected = (double)data->reference[i];
        const double error = fabs(actual - expected);
        const double tolerance = 5.0e-5 + 2.0e-5 * fabs(expected);

        if (error > max_absolute_error) {
            max_absolute_error = error;
        }
        if (!isfinite(actual) || error > tolerance) {
            fprintf(stderr,
                    "validation failed for %s at output %zu: got %.9g, "
                    "expected %.9g, error %.3g\n",
                    kernel->name, i, actual, expected, error);
            *passed = 0;
            return max_absolute_error;
        }
    }
    consume_output(data);
    return max_absolute_error;
}

static void call_kernel(const benchmark_kernel *kernel, benchmark_data *data,
                        size_t iterations)
{
    for (size_t i = 0; i < iterations; ++i) {
        kernel->function(data->input, data->weights, data->bias, data->output,
                         data->batch, data->input_features, data->output_features);
    }
}

static size_t calibrate_iterations(const benchmark_kernel *kernel, benchmark_data *data,
                                   double target_seconds)
{
    size_t iterations = 1;
    double elapsed = 0.0;

    do {
        const double start = now_seconds();
        call_kernel(kernel, data, iterations);
        elapsed = now_seconds() - start;
        consume_output(data);

        if (elapsed < 0.010 && iterations <= SIZE_MAX / 2U) {
            iterations *= 2U;
        } else {
            break;
        }
    } while (iterations < 100000000U);

    if (elapsed > 0.0) {
        const double estimated = target_seconds * (double)iterations / elapsed;
        if (estimated >= 1.0 && estimated < 100000000.0) {
            iterations = (size_t)estimated;
        } else if (estimated >= 100000000.0) {
            iterations = 100000000U;
        }
    }

    return iterations == 0 ? 1 : iterations;
}

static int benchmark_one_case(const benchmark_case *benchmark,
                              kernel_result results[KERNEL_COUNT], int sample_count,
                              double target_seconds)
{
    benchmark_data data;
    int passed = 1;

    if (!initialize_data(benchmark, &data)) {
        fprintf(stderr, "failed to allocate benchmark shape %s\n", benchmark->shape);
        destroy_data(&data);
        return 0;
    }

    for (int kernel_index = 0; kernel_index < KERNEL_COUNT; ++kernel_index) {
        results[kernel_index].max_absolute_error =
            validate_kernel(&kernels[kernel_index], &data, &passed);
        for (int warmup = 0; warmup < WARMUP_COUNT; ++warmup) {
            call_kernel(&kernels[kernel_index], &data, 1);
        }
        consume_output(&data);
        results[kernel_index].iterations =
            calibrate_iterations(&kernels[kernel_index], &data, target_seconds);
    }

    if (!passed) {
        destroy_data(&data);
        return 0;
    }

    /* Rotate the first kernel each sample to reduce order/thermal bias. */
    for (int sample = 0; sample < sample_count; ++sample) {
        for (int slot = 0; slot < KERNEL_COUNT; ++slot) {
            const int kernel_index = (slot + sample) % KERNEL_COUNT;
            const size_t iterations = results[kernel_index].iterations;
            const double start = now_seconds();

            call_kernel(&kernels[kernel_index], &data, iterations);
            results[kernel_index].seconds[sample] =
                (now_seconds() - start) / (double)iterations;
            consume_output(&data);
        }
    }

    for (int kernel_index = 0; kernel_index < KERNEL_COUNT; ++kernel_index) {
        results[kernel_index].median_seconds =
            median(results[kernel_index].seconds, sample_count);
        results[kernel_index].mad_seconds =
            median_absolute_deviation(results[kernel_index].seconds, sample_count,
                                      results[kernel_index].median_seconds);
    }

    destroy_data(&data);
    return 1;
}

static double environment_double(const char *name, double fallback)
{
    const char *text = getenv(name);
    char *end = NULL;
    double value;

    if (text == NULL || *text == '\0') {
        return fallback;
    }
    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value) || value <= 0.0 ||
        value > 60000.0) {
        fprintf(stderr, "ignoring invalid %s=%s\n", name, text);
        return fallback;
    }
    return value;
}

static int environment_samples(int fallback)
{
    const char *text = getenv("DENSE_BENCH_SAMPLES");
    char *end = NULL;
    long value;

    if (text == NULL || *text == '\0') {
        return fallback;
    }
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 3 || value > MAX_SAMPLES ||
        value % 2 == 0) {
        fprintf(stderr, "ignoring invalid DENSE_BENCH_SAMPLES=%s (use odd 3..%d)\n",
                text, MAX_SAMPLES);
        return fallback;
    }
    return (int)value;
}

static void print_results(kernel_result results[CASE_COUNT][KERNEL_COUNT],
                          int sample_count, double target_milliseconds,
                          double maximum_mad_percent)
{
    struct utsname system_information;
    char timestamp[64] = "unavailable";
    const time_t now = time(NULL);
    struct tm local_time;

    if (now != (time_t)-1 && localtime_r(&now, &local_time) != NULL) {
        (void)strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S%z",
                       &local_time);
    }

    printf("# %s\n", DENSE_REPORT_TITLE);
    puts("");
    printf("- Compiler: `%s`\n", DENSE_COMPILER_NAME);
    printf("- Kernel flags: `-std=c17 -O3 %s %s` (no `-ffast-math`; no LTO)\n",
           DENSE_NATIVE_FLAGS_TEXT, DENSE_FP_FLAGS_TEXT);
    printf("- Scalar-control flags: `%s`\n", DENSE_SCALAR_FLAGS_TEXT);
    if (uname(&system_information) == 0) {
        printf("- Host: `%s %s %s`\n", system_information.sysname,
               system_information.release, system_information.machine);
    }
    printf("- Run time: `%s`\n", timestamp);
    printf("- Protocol: %d warmups, auto-calibrated %.0f ms batches, "
           "%d samples, median and MAD; sample order rotates\n",
           WARMUP_COUNT, target_milliseconds, sample_count);
    printf("- Quality gate: every kernel/shape must have MAD <= %.1f%%\n",
           maximum_mad_percent);
    puts("- Data: deterministic FP32 inputs; 64-byte-aligned, disjoint buffers; "
         "warm-cache inference");
#if defined(__APPLE__)
    puts("- Scheduling: benchmark thread requests macOS user-interactive QoS");
#endif
    puts("");
    puts("| Shape BxKxN | Kernel | Median us | MAD % | GFLOP/s | Speedup vs IJK |");
    puts("|---|---|---:|---:|---:|---:|");

    for (int case_index = 0; case_index < CASE_COUNT; ++case_index) {
        const benchmark_case *benchmark = &benchmark_cases[case_index];
        const double operations = 2.0 * (double)benchmark->batch *
                                  (double)benchmark->input_features *
                                  (double)benchmark->output_features;
        const double baseline = results[case_index][0].median_seconds;

        for (int kernel_index = 0; kernel_index < KERNEL_COUNT; ++kernel_index) {
            const kernel_result *result = &results[case_index][kernel_index];
            const double microseconds = result->median_seconds * 1.0e6;
            const double mad_percent =
                result->median_seconds > 0.0
                    ? 100.0 * result->mad_seconds / result->median_seconds
                    : 0.0;
            const double gflops = operations / result->median_seconds / 1.0e9;
            const double speedup = baseline / result->median_seconds;

            printf("| %s | `%s` | %.3f | %.2f | %.3f | %.2fx |\n", benchmark->shape,
                   kernels[kernel_index].name, microseconds, mad_percent, gflops,
                   speedup);
        }
    }

    puts("");
    puts("## Isolating SIMD from loop-order/locality gains");
    puts("");
    puts("The scalar control has the same aligned IKJ source shape but is compiled "
         "with both loop and SLP vectorization disabled.");
    puts("");
    puts("| Shape BxKxN | Aligned scalar us | Aligned SIMD us | SIMD uplift |");
    puts("|---|---:|---:|---:|");
    for (int case_index = 0; case_index < CASE_COUNT; ++case_index) {
        const double scalar = results[case_index][4].median_seconds;
        const double vector = results[case_index][5].median_seconds;
        printf("| %s | %.3f | %.3f | %.2fx |\n", benchmark_cases[case_index].shape,
               scalar * 1.0e6, vector * 1.0e6, scalar / vector);
    }

    puts("");
    puts("## Raw calibrated samples");
    puts("");
    puts("```csv");
    puts("shape,kernel,sample,iterations,ns_per_call,max_absolute_error");
    for (int case_index = 0; case_index < CASE_COUNT; ++case_index) {
        for (int kernel_index = 0; kernel_index < KERNEL_COUNT; ++kernel_index) {
            const kernel_result *result = &results[case_index][kernel_index];
            for (int sample = 0; sample < sample_count; ++sample) {
                printf("%s,%s,%d,%zu,%.3f,%.9g\n", benchmark_cases[case_index].shape,
                       kernels[kernel_index].name, sample + 1, result->iterations,
                       result->seconds[sample] * 1.0e9, result->max_absolute_error);
            }
        }
    }
    puts("```");
}

static int results_are_stable(kernel_result results[CASE_COUNT][KERNEL_COUNT],
                              double maximum_mad_percent)
{
    int stable = 1;

    for (int case_index = 0; case_index < CASE_COUNT; ++case_index) {
        for (int kernel_index = 0; kernel_index < KERNEL_COUNT; ++kernel_index) {
            const kernel_result *result = &results[case_index][kernel_index];
            const double mad_percent =
                result->median_seconds > 0.0
                    ? 100.0 * result->mad_seconds / result->median_seconds
                    : 0.0;

            if (mad_percent > maximum_mad_percent) {
                fprintf(stderr, "unstable benchmark: %s/%s MAD %.2f%% exceeds %.2f%%\n",
                        benchmark_cases[case_index].shape, kernels[kernel_index].name,
                        mad_percent, maximum_mad_percent);
                stable = 0;
            }
        }
    }
    return stable;
}

int main(int argc, char **argv)
{
    kernel_result results[CASE_COUNT][KERNEL_COUNT] = {0};
    int sample_count = environment_samples(11);
    double target_milliseconds = environment_double("DENSE_BENCH_TARGET_MS", 100.0);
    double maximum_mad_percent = environment_double("DENSE_BENCH_MAX_MAD_PERCENT", 5.0);
    int quick = 0;

    if (argc == 2 && strcmp(argv[1], "--quick") == 0) {
        sample_count = 3;
        target_milliseconds = 5.0;
        quick = 1;
    } else if (argc != 1) {
        fprintf(stderr,
                "usage: %s [--quick]\n"
                "optional environment: DENSE_BENCH_TARGET_MS, "
                "DENSE_BENCH_SAMPLES (odd 3..%d), "
                "DENSE_BENCH_MAX_MAD_PERCENT\n",
                argv[0], MAX_SAMPLES);
        return EXIT_FAILURE;
    }

    configure_benchmark_thread();

    for (int case_index = 0; case_index < CASE_COUNT; ++case_index) {
        if (!benchmark_one_case(&benchmark_cases[case_index], results[case_index],
                                sample_count, target_milliseconds / 1000.0)) {
            return EXIT_FAILURE;
        }
    }

    print_results(results, sample_count, target_milliseconds, maximum_mad_percent);
    if (!isfinite(checksum_sink)) {
        fputs("non-finite checksum\n", stderr);
        return EXIT_FAILURE;
    }
    if (!quick && !results_are_stable(results, maximum_mad_percent)) {
        fputs("benchmark quality gate failed; existing reports were not replaced\n",
              stderr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
