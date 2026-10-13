# Vectorization notebook

This notebook records what the source transformation changes and what the two
compilers actually did. It is deliberately more precise than “SIMD made it
faster.”

## Experiment matrix

| Kernel | Loop order | `restrict` | 64-byte contract | Expected inner access |
|---|---|---:|---:|---|
| `ijk-baseline` | B → N → K | no | no | weight stride = N |
| `ijk-restrict` | B → N → K | yes | no | weight stride = N |
| `ikj` | B → K → N | no | no | contiguous, possible alias checks |
| `ikj-restrict` | B → K → N | yes | no | contiguous, disjoint buffers |
| `ikj-aligned` | B → K → N | yes | yes | contiguous and aligned |
| `ikj-aligned-scalar` | B → K → N | yes | yes | same source shape, vectorizers off |

The scalar control is compiled in its own translation unit. Clang gets
`-fno-vectorize -fno-slp-vectorize -fno-builtin-memcpy`; GCC gets
`-fno-tree-vectorize -fno-tree-loop-distribute-patterns`. The extra pattern
flags keep the bias loop from becoming a call to a potentially vectorized libc
`memcpy`, so the control contains no hidden SIMD. Everything else uses the same
optimization level, target, FP semantics, data, and timing harness.

## Why interchange is legal

For a fixed batch row and output channel, the only loop-carried dependency is
the accumulation over K. Interchanging N and K does not change K's order for
any output:

```text
IJK: for each n, visit k = 0, 1, 2, ...
IKJ: for each k = 0, 1, 2, ..., update every n
```

It does change memory behavior. IJK asks for `weights[k*N+n]` while K changes,
so adjacent iterations jump N floats. IKJ changes N in the inner loop, making
both the weight row and output row contiguous. Each SIMD lane owns a different
output accumulator, so vectorization does not require reassociating one FP
reduction.

## What the reports say

The checked-in reports are generated with `make diagnostics`.

- GCC 16 reports the benchmark-relevant IJK loops as missed/unprofitable. It
  vectorizes plain IKJ with runtime alias versioning, then vectorizes the
  `restrict` and aligned variants directly with 16-byte vectors.
- Clang 23 reports vectorized IJK source loops because it creates specialized
  runtime paths, including a vector reduction for `N == 1`. Inspection of the
  generated M2 assembly shows the `N=128/256` benchmark path still using a
  scalar strided load and dependent scalar FMA. Its IKJ inner loop is a direct
  four-way-unrolled NEON vector loop.

That Clang detail is important: an optimization remark proves that the compiler
created a vector path, not that a specific production shape takes that path.
The relevant assembly and measured workload complete the argument.

## `restrict` and alignment are contracts

`restrict` promises that accesses through the four qualified pointer roots do
not overlap for the duration of the call. Violating that promise is undefined
behavior; it is not a best-effort hint.

The aligned kernel similarly promises 64-byte-aligned allocations and K/N
dimensions that are multiples of 16 floats, so every derived row retains the
alignment. The tests and benchmark honor those preconditions. Modern AArch64
handles unaligned vector accesses well, so alignment is expected to be a
smaller effect than loop interchange; the measurements are reported as-is.

## Floating-point policy

The experiment does not use `-ffast-math` or permit reassociation. It explicitly
uses `-ffp-contract=fast`, so a multiply and its adjacent add may become one FMA
in every variant. Every output still visits K in ascending order after
interchange. The reference uses double accumulation, and each kernel is
validated before timing with an absolute-plus-relative tolerance. Separate
translation units and no LTO keep the timed calls visible to the optimizer
while the checksum prevents dead-code elimination.

## SVE/SVE2 interpretation

`-mcpu=neoverse-v2` selects an SVE2-capable core, but ordinary FP32 loads and
FMA are SVE instructions rather than SVE2-exclusive operations. The Makefile's
inspection target adds `-msve-vector-bits=256` so LLVM 23 selects `z` registers
instead of its default NEON cost-model choice. This is code-generation analysis,
not a model of a specific Grace implementation.

Native performance was measured on Apple Silicon using AArch64 Advanced SIMD
(NEON). Neoverse-V2/SVE-targeted assembly was generated and inspected, but SVE2
performance was not measured because no NVIDIA Grace/Neoverse-V2 hardware was
available.
