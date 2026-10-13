// Annotated excerpt, not a standalone assembly file.
// Generated from src/kernels.c with:
//   clang -Iinclude -std=c17 -O3 -ffp-contract=fast -mcpu=apple-m2 \
//     -S src/kernels.c
// Compiler: Homebrew Clang 23.1.3
// Host: Apple M2 (Mac14,2)

// ---------------------------------------------------------------------------
// IJK baseline: the benchmarked N=128/N=256 path is scalar and strided.
// ---------------------------------------------------------------------------
// x21 = k, x16 = output_features * sizeof(float), x22 = k's byte offset.
// The input load advances contiguously; the weight address jumps by an entire
// KxN row on every iteration. One accumulator (s0) creates a dependency chain.
LBB0_19:
    ldr     s1, [x0, x21, lsl #2]  // input_row[k]
    ldr     s2, [x19, x22]         // weights[k * N + output]
    fmadd   s0, s1, s2, s0         // scalar sum += input * weight
    add     x21, x21, #1
    add     x22, x22, x16          // stride by N floats, not one float
    cmp     x5, x21
    b.ne    LBB0_19

// LLVM also emitted a runtime-specialized vector reduction for N=1. The Clang
// optimization report therefore says the source loop was vectorized, but that
// path is not selected by this project's N=128/N=256 benchmark shapes. This is
// why the report and the relevant assembly path must be read together.

// ---------------------------------------------------------------------------
// IKJ + restrict + 64-byte alignment: 16 output channels per iteration.
// ---------------------------------------------------------------------------
// q registers are 128-bit NEON/Advanced SIMD registers: four FP32 lanes each.
// x12 walks a contiguous weight row; x11 walks the contiguous output row.
LBB4_12:
    ldp     q1, q2, [x12, #-32]    // load weights[o + 0 .. o + 7]
    ldp     q3, q4, [x12], #64     // load weights[o + 8 .. o + 15]
    ldp     q5, q6, [x11, #-32]    // load outputs[o + 0 .. o + 7]
    ldp     q7, q16, [x11]         // load outputs[o + 8 .. o + 15]
    fmla.4s v5, v1, v0[0]          // y[0:4]  += weight[0:4]  * input[k]
    fmla.4s v6, v2, v0[0]          // y[4:8]  += weight[4:8]  * input[k]
    fmla.4s v7, v3, v0[0]          // y[8:12] += weight[8:12] * input[k]
    fmla.4s v16, v4, v0[0]         // y[12:16] += weight[12:16] * input[k]
    stp     q5, q6, [x11, #-32]    // store the first eight updated outputs
    stp     q7, q16, [x11], #64    // store the next eight and advance
    subs    x13, x13, #16
    b.ne    LBB4_12

// A four-lane epilogue handles N % 16 before the final scalar tail:
LBB4_16:
    ldr     q1, [x13], #16
    ldr     q2, [x12]
    fmla.4s v2, v1, v0[0]
    str     q2, [x12], #16
    adds    x11, x11, #4
    b.ne    LBB4_16
