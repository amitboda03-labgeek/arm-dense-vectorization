// Annotated excerpt, not a standalone assembly file.
// Generated from src/kernels.c with:
//   clang --target=aarch64-none-elf -Iinclude -std=c17 -O3 -ffp-contract=fast \
//     -mcpu=neoverse-v2 -msve-vector-bits=256 -S src/kernels.c
// Compiler: Homebrew Clang 23.1.3

// Neoverse V2 is SVE2-capable. These FP32 operations are from the SVE subset;
// they are not SVE2-exclusive opcodes. The explicit 256-bit code-generation
// assumption makes LLVM select SVE for inspection and is not a claim about the
// vector length or measured performance of NVIDIA Grace hardware.

// p0 activates every FP32 lane. z registers are scalable vector registers;
// "mul vl" addresses memory in units of the selected vector length.
    ptrue   p0.s

// Broadcast this input activation so every lane updates a different output.
    mov     z1.s, s0

// Four unrolled SVE vectors: with the inspection setting above, this covers
// 4 vectors * 8 FP32 lanes = 32 output channels per loop iteration.
.LBB4_12:
    ldr     z2, [x12, #-2, mul vl] // contiguous weight vector 0
    ldr     z3, [x12, #-1, mul vl] // contiguous weight vector 1
    ldr     z4, [x12]              // contiguous weight vector 2
    ldr     z5, [x12, #1, mul vl]  // contiguous weight vector 3
    ldr     z6, [x11, #-2, mul vl] // current output vector 0
    ldr     z7, [x11, #-1, mul vl] // current output vector 1
    fmad    z2.s, p0/m, z1.s, z6.s // result = weight * input + output
    ldr     z16, [x11]             // current output vector 2
    ldr     z17, [x11, #1, mul vl] // current output vector 3
    fmad    z3.s, p0/m, z1.s, z7.s
    fmad    z4.s, p0/m, z1.s, z16.s
    fmad    z5.s, p0/m, z1.s, z17.s
    str     z2, [x11, #-2, mul vl]
    str     z3, [x11, #-1, mul vl]
    str     z4, [x11]
    str     z5, [x11, #1, mul vl]
    add     x11, x11, #128
    b.ne    .LBB4_12

// The compiler also emits NEON and scalar epilogues for smaller remainders.
// SVE code was generated and inspected only; it was not executed here.
