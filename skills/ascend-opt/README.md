# Ascend performance optimization skills

These skills help an optimization agent run, profile and improve workloads on Ascend NPUs. Start with the task's baseline and correctness contract. Profiling and experiments explain results, and the task's evaluator determines the final performance and numerical verdict.

## Environment

| Skill | Use it for |
|---|---|
| [ascend-environment](ascend-environment/SKILL.md) | NPU and SoC identification, CANN/OPP readiness, and container device visibility. |
| [ascend-communication](ascend-communication/SKILL.md) | HCCL collective tests with the workload's rank mapping, topology, and payload sizes. |

## Best Practice

| Skill | Cases and guidance |
|---|---|
| [ascendc-perf-optimize](ascendc-perf-optimize/SKILL.md) | Use measured metrics to choose inter-core or intra-core pipeline changes, including ping-pong prefetch. |
| [ascendc-performance-best-practices](ascendc-performance-best-practices/SKILL.md) | Choose implementation approaches by NPU generation and operator family, including MatMul, FlashAttention, and gather/scatter. |
| [ascend-npugraph-performance](ascend-npugraph-performance/SKILL.md) | Diagnose redundant copies or clones in graph capture and replay using FX graphs and reinplace diagnostics. |
| [ascendc-mc2-best-practice](ascendc-mc2-best-practice/SKILL.md) | Pick an MC2 route and consult the pipeline and synchronization examples for computation–communication fusion. |
| [ascendc-api-best-practices](ascendc-api-best-practices/SKILL.md) | Look up API contracts for buffers, copies, alignment, vector operations, and synchronization. |

## Profiling

| Skill | Start here when |
|---|---|
| [ascend-model-profiling](ascend-model-profiling/SKILL.md) | You have a runnable model and need to enable a framework hook, integrate `torch_npu.profiler`, or collect and validate a training, inference, or RL trace. |
| [ascend-ops-profiling](ascend-ops-profiling/SKILL.md) | You have a runnable operator case or msOpProf collection and need operator timing, metric analysis, runtime triage, or a visual report. |
| [ascend-profiling-analysis](ascend-profiling-analysis/SKILL.md) | You have existing model profiling artifacts and need to locate bottlenecks or timeline gaps, or [attribute time to model components](ascend-profiling-analysis/references/model-perf-breakdown.md). |

## Ascend C Operator Development

| Skill | Use it for |
|---|---|
| [ascendc-tiling-design](ascendc-tiling-design/SKILL.md) | Design core partitioning and UB tiling, with elementwise and Welford examples. |
| [torch-ascendc-op-extension](torch-ascendc-op-extension/SKILL.md) | Connect an Ascend C operator to PyTorch through direct launch or ACLNN. |
| [ascend-npu-objdump](ascend-npu-objdump/SKILL.md) | Inspect operator binaries and correlate device PCs with instruction windows. |

Some specialized tools require a matching CANN installation or additional packages. See [skills/upstream.md](upstream.md) for source revisions and external resources.
