# Reproducible reports

These checked-in artifacts are machine evidence, not hand-written claims:

- `clang-vectorization.txt` and `gcc-vectorization.txt` come from
  `make diagnostics`.
- `clang-benchmark.md` and `gcc-benchmark.md` come from `make benchmark` and
  include every calibrated timing sample after the summary table.

Compiler remarks contain source line numbers, so regenerate them whenever the
kernel changes. Benchmark values are specific to the named host and toolchain;
do not silently replace them with results from a different machine.
