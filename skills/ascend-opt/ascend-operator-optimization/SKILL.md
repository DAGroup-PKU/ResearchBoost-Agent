---
name: ascend-operator-optimization
description: Optimize or minimally modify an Ascend C operator or its PyTorch integration after profiling identifies a kernel bottleneck.
---

# Ascend C operator optimization

Start with an observed hotspot and the existing RBA action ladder: test supported switches or library operators before a custom kernel. Keep the operator's mathematical definition, shape/dtype/layout, aliasing and numerical contract explicit. Use the existing operator project and installed target headers; avoid introducing a template project solely for a small change.

## Select the implementation

- Identify the SoC generation, available API/overload and compiler/CANN version. Consult the installed headers and version-matched official Ascend C documentation; do not transfer buffer sizes, instructions or alignment rules between chip generations.
- Preserve the host-side schema, output allocation, device/stream, tiling metadata and framework registration contract for the chosen direct-launch or ACLNN path. Keep PyTorch invocation and graph behavior consistent with the existing application.

## Analyze and change

1. Classify the measured bottleneck: launch/scheduling, GM traffic, core underutilization, vector/Cube compute, synchronization or communication. Pick a single hypothesis and a representative shape.
2. Partition work across cores without gaps or overlaps. Choose tile size from the target's real UB capacity, element size, queue depth, alignment and temporary buffer requirements. For double buffering account for both live buffers and scratch space; avoid assuming maximum occupancy always wins.
3. Review data movement, compute and synchronization as a pipeline. Overlap only independent work; ensure buffers are not reused before their consumers finish. For reductions preserve the intended accumulation order and dtype; changes may affect numerical results.
4. Handle tails and boundary shapes explicitly. A DMA alignment condition is measured in bytes, whereas an algorithmic tile tail is measured in elements; check the target overload's units. Avoid reading or writing beyond valid GM ranges. Compare aligned and unaligned shapes.
5. Test the changed operator on a minimal representative case and its actual application path. For any mismatch use the Ascend accuracy/debugging skill and the task's frozen numerical checks. Use profiling to explain effects and the task evaluator for the final result.

References: [Ascend C API guidance](https://gitcode.com/cann/cannbot-skills/tree/2af66d732e595297cf34159cf8e1d49fa7354c2c/ops/ascendc-api-best-practices), [tiling design](https://gitcode.com/cann/cannbot-skills/tree/2af66d732e595297cf34159cf8e1d49fa7354c2c/ops/ascendc-tiling-design), [PyTorch extension](https://gitcode.com/cann/cannbot-skills/tree/2af66d732e595297cf34159cf8e1d49fa7354c2c/ops/torch-ascendc-op-extension).
