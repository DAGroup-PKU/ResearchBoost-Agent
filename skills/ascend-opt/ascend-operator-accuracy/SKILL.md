---
name: ascend-operator-accuracy
description: Locate Ascend operator mismatches, tail/alignment faults and unstable intermediate values, then verify a targeted kernel repair.
---

# Ascend operator accuracy debugging

1. Use the task's supplied reference and numerical contract, then follow the precision guide and relevant pattern.
2. For non-aligned tails, use the corrected writeback example and the Ascend C API alignment guidance.

## Working references

Read the reference for the current phase; the original technical procedures, examples and resources are retained below. Resolve a documented script relative to its reference guide's directory.

- [ascendc-precision-debug](references/native/ascendc-precision-debug/GUIDE.md)
