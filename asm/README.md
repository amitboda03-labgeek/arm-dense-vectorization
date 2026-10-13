# Assembly evidence

The checked-in files are short, annotated excerpts copied from assembly that
the `Makefile` generated from the current kernel source. They connect the
compiler remarks to concrete instructions without checking in thousands of
lines of generated output.

- `apple-m2-neon-annotated.s` contrasts the strided scalar IJK path with the
  four-way-unrolled 128-bit NEON `fmla.4s` loop.
- `neoverse-v2-sve-annotated.s` shows `z` registers, predicates, vector-length
  addressing, and predicated `fmad` for an SVE2-capable Neoverse-V2 target.

Regenerate the full, unannotated listings with:

```sh
make assembly
```

The raw files land in `build/assembly/`, which is intentionally ignored. A
diff between an annotated excerpt and regenerated output should be treated as
evidence that the source, compiler, or cost model changed and the annotations
need review.
