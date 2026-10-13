# Arm Dense-Layer Vectorization

A compiler-guided optimization case study for an FP32 dense layer used in
embedded inference. The project starts with a legal but cache-hostile
matrix-multiply loop, explains why compilers struggle with it, restructures it
for contiguous Arm SIMD, and verifies the result at four levels:

```text
compiler remarks → source transformation → emitted assembly → measured speedup
```

On an Apple M2, the final aligned IKJ kernel is **12.75–18.72× faster** than the
IJK baseline across the tested shapes and compilers. A scalarized control shows
that NEON vectorization alone contributes **2.54–4.47×**; the rest comes from
the loop order and memory-access pattern. These are measured results, not
estimates.

## The kernel

The operation is a row-major dense layer, equivalently a small GEMM with bias:

```text
Y[B,N] = X[B,K] × W[K,N] + bias[N]
```

The weights use the common input-major layout `W[k*N + n]`. The naïve B→N→K
loop walks that matrix with an N-float stride and carries one scalar reduction.
Changing the loop order to B→K→N makes weights and outputs contiguous in the
inner loop, while preserving the ascending K order of every output sum.

The experiment keeps six variants instead of showing only a before/after:

| Variant | Question it answers |
|---|---|
| `ijk-baseline` | What does the straightforward dense loop cost? |
| `ijk-restrict` | Is alias information alone enough? |
| `ikj` | What does loop interchange/locality unlock? |
| `ikj-restrict` | What changes when overlap checks are unnecessary? |
| `ikj-aligned` | Does a truthful 64-byte alignment contract help further? |
| `ikj-aligned-scalar` | How fast is the same optimized structure without SIMD? |

`restrict` is a caller contract, not a hint. The aligned kernel additionally
requires 64-byte-aligned, disjoint buffers and K/N dimensions divisible by 16;
the test and benchmark allocations satisfy those preconditions.

## Reproduce it on Apple Silicon

Install genuine LLVM Clang and GNU GCC—on macOS, `/usr/bin/gcc` is Apple Clang:

```sh
brew install llvm gcc
make test
make diagnostics
make assembly
make benchmark
```

The Makefile detects Homebrew's versioned `gcc-16` and supplies the active
macOS SDK path when Homebrew GCC needs it. Tool paths and native flags remain
overridable, for example:

```sh
make clean
make CLANG=/path/to/clang GCC=/path/to/gcc NATIVE_FLAGS=-march=native test
```

Run `make clean` before changing command-line compiler or flag overrides; Make
tracks source/Makefile changes, but not the values of variables from a previous
invocation.

Useful targets:

| Target | Output |
|---|---|
| `make test` | correctness under both Clang and GCC, including an odd tail |
| `make diagnostics` | checked-in Clang/GCC vectorizer decisions |
| `make assembly` | full Apple-M2 NEON and Neoverse-V2 listings under `build/` |
| `make benchmark` | median/MAD tables plus every raw calibrated sample |
| `make evidence` | all of the above in one command |

## Measured results

Measured 2026-10-13 on an Apple M2 (`Mac14,2`, 8 CPU cores, 16 GiB) running
macOS 27.0. Each row compares the same compiler's IJK baseline with the final
aligned kernel.

| Compiler | Shape B×K×N | Baseline µs | Aligned IKJ µs | End-to-end speedup | Optimized GFLOP/s | SIMD uplift vs scalar IKJ |
|---|---:|---:|---:|---:|---:|---:|
| Clang 23.1.3 | 1×256×128 | 30.274 | 2.059 | 14.71× | 31.836 | 3.08× |
| Clang 23.1.3 | 1×256×256 | 62.199 | 4.878 | 12.75× | 26.870 | 2.55× |
| Clang 23.1.3 | 4×256×256 | 246.074 | 19.242 | 12.79× | 27.247 | 2.54× |
| GCC 16.2.0 | 1×256×128 | 29.583 | 1.580 | 18.72× | 41.478 | 4.47× |
| GCC 16.2.0 | 1×256×256 | 61.125 | 4.087 | 14.95× | 32.067 | 3.44× |
| GCC 16.2.0 | 4×256×256 | 271.352 | 16.796 | 16.16× | 31.215 | 3.40× |

The complete [Clang report](reports/clang-benchmark.md) and
[GCC report](reports/gcc-benchmark.md) include all six variants and every raw
sample. Alignment itself was neutral within normal run-to-run variation on this
M2; that is expected because modern AArch64 handles unaligned vector accesses
well. It remains useful as an explicit contract and as an experiment, not as a
fabricated speedup claim.

### Benchmark protocol

- Kernels are `noinline` in separate translation units; no LTO is used.
- Every variant uses `-std=c17 -O3 -mcpu=native -ffp-contract=fast`. FMA
  contraction is explicit and equal; reassociation/`-ffast-math` is not used.
- Inputs are deterministic and allocated in separate 64-byte-aligned buffers.
- A double-accumulation reference validates every output before timing.
- Each kernel gets five warmups and an independently calibrated ~100 ms batch.
- Eleven samples are collected with rotated kernel order; the table reports the
  median and median absolute deviation (MAD).
- A 5% MAD quality gate rejects noisy runs before they can replace the committed
  reports.
- The benchmark thread requests macOS user-interactive QoS to reduce P/E-core
  placement variance; macOS still provides no hard user-level core pinning.
- A weighted checksum outside the timed region prevents dead-code elimination.
- Results are warm-cache inference measurements. `2*B*K*N` is used for the
  conventional approximate FLOP count, while bias initialization stays timed.

This protocol reduces—but cannot eliminate—DVFS, scheduler, and thermal noise.
The committed raw samples make that variation auditable.

## What the compilers decided

Generate the reports with exactly the flags requested for this investigation:

```sh
# Clang
-Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
-Rpass-analysis=loop-vectorize

# GCC
-fopt-info-vec-optimized=...  # one compile
-fopt-info-vec-missed=...     # a second compile
```

The [Clang remarks](reports/clang-vectorization.txt) and
[GCC remarks](reports/gcc-vectorization.txt) reveal a useful compiler
difference:

| Source variant | Clang 23 | GCC 16 |
|---|---|---|
| IJK baseline/restrict | Creates specialized vector paths, but benchmarked N>1 path remains scalar/strided | Rejects as unprofitable/unsupported |
| IKJ without `restrict` | Vectorizes | Vectorizes with runtime alias versioning |
| IKJ with `restrict` | 4-wide, 4-way-interleaved loop | 16-byte vectors without alias versioning |
| IKJ aligned | Same direct NEON loop | Same direct NEON loop |

The Clang result is why remarks alone are insufficient: a remark proves that a
vector path exists, not that the production shape selects it. The
[annotated Apple-M2 assembly](asm/apple-m2-neon-annotated.s) shows both the
scalar strided IJK path and the four-unrolled `fmla.4s` NEON loop used after
interchange. The deeper reasoning is recorded in the
[vectorization notebook](docs/vectorization-notebook.md).

## Neoverse V2 and SVE2 inspection

`make assembly` also cross-compiles the pure kernel translation unit without a
sysroot:

```sh
clang --target=aarch64-none-elf -Iinclude -std=c17 -O3 -ffp-contract=fast \
  -mcpu=neoverse-v2 -msve-vector-bits=256 \
  -S src/kernels.c -o build/assembly/neoverse-v2-sve.s
```

The [annotated Neoverse-V2 excerpt](asm/neoverse-v2-sve-annotated.s) identifies
`z` registers, `ptrue` predication, vector-length addressing, loads/stores, and
predicated FP32 `fmad`. These arithmetic instructions are from SVE, even though
Neoverse V2 is an SVE2-capable target; calling the FMA “SVE2-only” would be
incorrect.

For Compiler Explorer, paste the self-contained
[`tools/compiler-explorer.c`](tools/compiler-explorer.c) probe into an ARM64
Clang pane and use:

```text
-O3 -ffp-contract=fast -mcpu=neoverse-v2 -msve-vector-bits=256
```

Remove the vector-bits option to compare LLVM 23's default NEON cost-model
choice. The extra option is a code-generation experiment, not a statement about
the physical vector length of a Grace CPU.

> Native performance was measured on Apple Silicon using AArch64 Advanced SIMD
> (NEON). I generated and inspected code targeting `-mcpu=neoverse-v2`, but I
> did not benchmark SVE2 because I do not have NVIDIA Grace/Neoverse-V2 hardware.

## Correctness and scope

`make test` checks all six kernels against a double-precision accumulator on
aligned shapes and checks every non-alignment-contract kernel on `3×17×19` to
exercise vector tails. Both Clang and genuine GNU GCC must pass.

The dimensions model a small STM32-style fully connected inference workload,
but the performance numbers are **not STM32 measurements**. Apple M2 and STM32
devices differ in ISA, cache, memory, clocking, and power limits. The portable
skill demonstrated here is using compiler evidence to reshape a kernel. A real
deployment would add CMSIS-NN/Helium comparisons, quantization, SRAM-aware
tiling, and DWT cycle measurements on the target board.

This is intentionally not a replacement for an optimized BLAS library. It is a
small, auditable experiment that exposes the reasoning normally hidden inside
one benchmark number.

## Repository map

```text
include/dense.h                    public contracts and six kernel APIs
src/kernels.c                      progressive source variants
src/scalar_control.c               optimized shape with vectorizers disabled
src/benchmark.c                    validation, calibration, statistics, raw data
tests/test_kernels.c               aligned and odd-tail correctness tests
reports/                           compiler remarks and measured raw samples
asm/                               annotated NEON and SVE evidence
tools/compiler-explorer.c          self-contained CE inspection probe
docs/vectorization-notebook.md     decisions, caveats, and interview reasoning
```

## Resume bullet

> Optimized an FP32 Arm dense-layer kernel using loop interchange, `restrict`,
> and explicit alignment contracts; validated Clang/GCC auto-vectorization with
> compiler diagnostics and AArch64 assembly, measured 12.75–18.72× speedups on
> Apple M2, and analyzed Neoverse-V2 SVE/SVE2-targeted code generation.

## Interview discussion points

- Why `restrict` removes alias uncertainty but cannot turn strided loads into
  contiguous ones.
- Why loop interchange is legal here and retains each output's FP accumulation
  order.
- Why GCC's runtime-versioned IKJ loop benefits from `restrict`, while Clang's
  M2 result is almost unchanged.
- Why a vectorized compiler remark does not prove the measured path uses SIMD.
- How the scalarized control separates SIMD uplift from locality improvements.
- Why alignment can be semantically valuable even when its timing impact is
  below measurement noise.
- Why Apple Silicon NEON results cannot be projected onto STM32 or Grace.

Released under the [MIT License](LICENSE).
