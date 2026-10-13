SHELL := /bin/sh

CLANG ?= /opt/homebrew/opt/llvm/bin/clang
GCC ?= $(shell command -v gcc-16 2>/dev/null)
NATIVE_FLAGS ?= -mcpu=native
FP_FLAGS ?= -ffp-contract=fast
MACOS_SYSROOT := $(shell xcrun --show-sdk-path 2>/dev/null)
SYSROOT_FLAGS := $(if $(MACOS_SYSROOT),-isysroot $(MACOS_SYSROOT),)

CPPFLAGS := -Iinclude
WARNINGS := -Wall -Wextra -Wpedantic -Wshadow -Wconversion
COMMON_FLAGS := -std=c17 -O3 $(WARNINGS) $(NATIVE_FLAGS) $(FP_FLAGS) $(SYSROOT_FLAGS)
CLANG_NOVEC_FLAGS := -fno-vectorize -fno-slp-vectorize -fno-builtin-memcpy
GCC_NOVEC_FLAGS := -fno-tree-vectorize -fno-tree-loop-distribute-patterns
CLANG_REMARK_FLAGS := -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
	-Rpass-analysis=loop-vectorize
BENCHMARK_COMMON_DEFINES := -D'DENSE_NATIVE_FLAGS_TEXT="$(NATIVE_FLAGS)"' \
	-D'DENSE_FP_FLAGS_TEXT="$(FP_FLAGS)"'

.PHONY: all check-tools test test-clang test-gcc benchmark benchmark-clang \
	benchmark-gcc diagnostics assembly evidence clean

all: test

check-tools:
	@test -x "$(CLANG)" || { echo "error: Homebrew Clang not found at $(CLANG)" >&2; exit 1; }
	@test -n "$(GCC)" && test -x "$(GCC)" || { \
		echo "error: GNU GCC not found; run 'brew install gcc' or set GCC=/path/to/gcc" >&2; \
		exit 1; \
	}

build/clang/kernels.o: src/kernels.c include/dense.h Makefile
	@mkdir -p $(@D)
	$(CLANG) $(CPPFLAGS) $(COMMON_FLAGS) -c $< -o $@

build/clang/scalar_control.o: src/scalar_control.c include/dense.h Makefile
	@mkdir -p $(@D)
	$(CLANG) $(CPPFLAGS) $(COMMON_FLAGS) $(CLANG_NOVEC_FLAGS) -c $< -o $@

build/clang/test_kernels.o: tests/test_kernels.c include/dense.h Makefile
	@mkdir -p $(@D)
	$(CLANG) $(CPPFLAGS) $(COMMON_FLAGS) -c $< -o $@

build/clang/benchmark.o: src/benchmark.c include/dense.h Makefile
	@mkdir -p $(@D)
	$(CLANG) $(CPPFLAGS) $(COMMON_FLAGS) $(BENCHMARK_COMMON_DEFINES) \
		-D'DENSE_SCALAR_FLAGS_TEXT="$(CLANG_NOVEC_FLAGS)"' -c $< -o $@

build/clang/test_kernels: build/clang/kernels.o build/clang/scalar_control.o \
	build/clang/test_kernels.o
	$(CLANG) $(COMMON_FLAGS) $^ -o $@

build/clang/benchmark: build/clang/kernels.o build/clang/scalar_control.o \
	build/clang/benchmark.o
	$(CLANG) $(COMMON_FLAGS) $^ -o $@

build/gcc/kernels.o: src/kernels.c include/dense.h Makefile
	@mkdir -p $(@D)
	$(GCC) $(CPPFLAGS) $(COMMON_FLAGS) -c $< -o $@

build/gcc/scalar_control.o: src/scalar_control.c include/dense.h Makefile
	@mkdir -p $(@D)
	$(GCC) $(CPPFLAGS) $(COMMON_FLAGS) $(GCC_NOVEC_FLAGS) -c $< -o $@

build/gcc/test_kernels.o: tests/test_kernels.c include/dense.h Makefile
	@mkdir -p $(@D)
	$(GCC) $(CPPFLAGS) $(COMMON_FLAGS) -c $< -o $@

build/gcc/benchmark.o: src/benchmark.c include/dense.h Makefile
	@mkdir -p $(@D)
	$(GCC) $(CPPFLAGS) $(COMMON_FLAGS) $(BENCHMARK_COMMON_DEFINES) \
		-D'DENSE_SCALAR_FLAGS_TEXT="$(GCC_NOVEC_FLAGS)"' -c $< -o $@

build/gcc/test_kernels: build/gcc/kernels.o build/gcc/scalar_control.o \
	build/gcc/test_kernels.o
	$(GCC) $(COMMON_FLAGS) $^ -o $@

build/gcc/benchmark: build/gcc/kernels.o build/gcc/scalar_control.o \
	build/gcc/benchmark.o
	$(GCC) $(COMMON_FLAGS) $^ -o $@

test: check-tools test-clang test-gcc

test-clang: build/clang/test_kernels
	./build/clang/test_kernels

test-gcc: build/gcc/test_kernels
	./build/gcc/test_kernels

benchmark: check-tools benchmark-clang benchmark-gcc

benchmark-clang: build/clang/benchmark
	@mkdir -p reports
	./build/clang/benchmark > reports/clang-benchmark.md.tmp
	mv reports/clang-benchmark.md.tmp reports/clang-benchmark.md
	@cat reports/clang-benchmark.md

benchmark-gcc: build/gcc/benchmark
	@mkdir -p reports
	./build/gcc/benchmark > reports/gcc-benchmark.md.tmp
	mv reports/gcc-benchmark.md.tmp reports/gcc-benchmark.md
	@cat reports/gcc-benchmark.md

diagnostics: check-tools
	@mkdir -p build/diagnostics reports
	@{ \
		echo "Compiler: $$($(CLANG) --version | sed -n '1p')"; \
		echo "Command flags: -std=c17 -O3 $(NATIVE_FLAGS) $(FP_FLAGS) $(CLANG_REMARK_FLAGS)"; \
		echo; \
		$(CLANG) $(CPPFLAGS) $(COMMON_FLAGS) $(CLANG_REMARK_FLAGS) \
			-c src/kernels.c -o build/diagnostics/clang-kernels.o 2>&1 >/dev/null; \
	} > reports/clang-vectorization.txt.tmp
	@mv reports/clang-vectorization.txt.tmp reports/clang-vectorization.txt
	@: > build/diagnostics/gcc-optimized.raw
	@: > build/diagnostics/gcc-missed.raw
	@$(GCC) $(CPPFLAGS) $(COMMON_FLAGS) \
		-fopt-info-vec-optimized=build/diagnostics/gcc-optimized.raw \
		-c src/kernels.c -o build/diagnostics/gcc-optimized.o
	@$(GCC) $(CPPFLAGS) $(COMMON_FLAGS) \
		-fopt-info-vec-missed=build/diagnostics/gcc-missed.raw \
		-c src/kernels.c -o build/diagnostics/gcc-missed.o
	@{ \
		echo "Compiler: $$($(GCC) --version | sed -n '1p')"; \
		echo "Command flags: -std=c17 -O3 $(NATIVE_FLAGS) $(FP_FLAGS) -fopt-info-vec-{optimized,missed}"; \
		echo; \
		echo "== Optimized loops =="; \
		sed 's|^src/|src/|' build/diagnostics/gcc-optimized.raw; \
		echo; \
		echo "== Missed loops =="; \
		sed 's|^src/|src/|' build/diagnostics/gcc-missed.raw; \
	} > reports/gcc-vectorization.txt.tmp
	@mv reports/gcc-vectorization.txt.tmp reports/gcc-vectorization.txt
	@echo "wrote reports/clang-vectorization.txt and reports/gcc-vectorization.txt"

assembly: check-tools
	@mkdir -p build/assembly
	$(CLANG) $(CPPFLAGS) -std=c17 -O3 $(FP_FLAGS) -mcpu=apple-m2 -S src/kernels.c \
		-o build/assembly/apple-m2.s
	$(CLANG) --target=aarch64-none-elf $(CPPFLAGS) -std=c17 -O3 $(FP_FLAGS) \
		-mcpu=neoverse-v2 -msve-vector-bits=256 -S src/kernels.c \
		-o build/assembly/neoverse-v2-sve.s
	@echo "wrote build/assembly/apple-m2.s and build/assembly/neoverse-v2-sve.s"

evidence: test diagnostics assembly benchmark

clean:
	rm -rf build
