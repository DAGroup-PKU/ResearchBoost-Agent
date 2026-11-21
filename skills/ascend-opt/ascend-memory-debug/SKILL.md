---
name: ascend-memory-debug
description: Run supported Ascend C or Triton sanitizer checks for bounds, initialization, races and synchronization faults.
---

# Ascend kernel memory diagnostics

1. Follow the sanitizer guide for the actual operator project type and supported toolchain.
2. Use its reproducible case and report to distinguish a detected kernel issue from an incomplete diagnostic run.

## Working references

Read the reference for the current phase; the original technical procedures, examples and resources are retained below. Resolve a documented script relative to its reference guide's directory.

- [ops-mssanitizer](references/native/ops-mssanitizer/GUIDE.md)
