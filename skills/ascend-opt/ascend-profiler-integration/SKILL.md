---
name: ascend-profiler-integration
description: Integrate torch_npu.profiler into training, inference or RL code with framework-specific injection and control examples.
---

# Ascend training and inference profiler integration

1. Choose the training, inference or RL reference and locate the process executing NPU work.
2. Preserve forward/backward/optimizer or role-specific execution while adding the profiler controls and configuration propagation.
3. Report an instrumentation patch separately from an executed collection with verified artifacts.

## Working references

Read the reference for the current phase; the original technical procedures, examples and resources are retained below. Resolve a documented script relative to its reference guide's directory.

- [ascend-profiler-collect-adaption](references/native/ascend-profiler-collect-adaption/GUIDE.md)
