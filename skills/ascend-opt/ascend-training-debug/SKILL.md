---
name: ascend-training-debug
description: Locate Ascend training OOM, nonfinite values, finite reference mismatch or repeated-run nondeterminism using the relevant evidence.
---

# Ascend training OOM and numerical debugging

1. Classify the first failing event and choose only its OOM, nonfinite, finite-mismatch or determinism guide.
2. If a collective timeout follows an OOM on another rank, diagnose that first failure before an independent communication test.
3. Use the frozen task numerical contract; these guides locate causes rather than replace its verdict.

## Working references

Read the reference for the current phase; the original technical procedures, examples and resources are retained below. Resolve a documented script relative to its reference guide's directory.

- [model-train-oom-analysis](references/native/model-train-oom-analysis/GUIDE.md)
- [model-train-precision-nonfinite](references/native/model-train-precision-nonfinite/GUIDE.md)
- [model-train-precision-numerical-mismatch](references/native/model-train-precision-numerical-mismatch/GUIDE.md)
- [model-train-precision-determinism](references/native/model-train-precision-determinism/GUIDE.md)
- [model-train-log-visualization](references/native/model-train-log-visualization/GUIDE.md)
