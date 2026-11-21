---
name: ascendc-kernel-tuning
description: Tune an Ascend C hotspot through target-specific tiling, buffer budgets and applicable single-core or inter-core pipelines.
---

# Ascend C tiling and pipeline tuning

1. Start with the pipeline guide and the tiling guide for the measured operator pattern.
2. Read architecture-specific performance or Blaze/RegBase material when selected by the actual SoC and implementation.
3. If results change, investigate accuracy comparison and debugging before interpreting performance.

## Working references

Read the reference for the current phase; the original technical procedures, examples and resources are retained below. Resolve a documented script relative to its reference guide's directory.

- [ascendc-perf-optimize](references/native/ascendc-perf-optimize/GUIDE.md)
- [ascendc-tiling-design](references/native/ascendc-tiling-design/GUIDE.md)
- [ascendc-performance-best-practices](references/native/ascendc-performance-best-practices/GUIDE.md)
- [ascendc-regbase-best-practice](references/native/ascendc-regbase-best-practice/GUIDE.md)
- [ascendc-blaze-best-practice](references/native/ascendc-blaze-best-practice/GUIDE.md)
- [ascendc-blaze-migration](references/native/ascendc-blaze-migration/GUIDE.md)
