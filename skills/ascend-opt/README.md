# Ascend optimization skills

Five focused skills supply Ascend-specific guidance to a ResearchBoost optimization task. Use the existing RBA measurement, numerical validation and action-ladder cards for the task-wide workflow and frozen judge.

| Skill | Use when |
|---|---|
| [ascend-environment](ascend-environment/SKILL.md) | Check assigned NPU, compatible software and distributed setup. |
| [ascend-deployment-migration](ascend-deployment-migration/SKILL.md) | Bring an existing PyTorch or training/inference framework workload to Ascend. |
| [ascend-profiling](ascend-profiling/SKILL.md) | Collect NPU model and operator traces and investigate a measured bottleneck. |
| [ascend-operator-optimization](ascend-operator-optimization/SKILL.md) | Change an existing Ascend C operator or its integration after profiling. |
| [ascend-accuracy-runtime-debug](ascend-accuracy-runtime-debug/SKILL.md) | Diagnose accuracy, runtime, memory or graph regressions. |

The cards adapt and summarize task-relevant guidance and link to pinned source material for specialized procedures and examples. Original upstream notices are retained in [licenses](licenses/). The previous full collection remains in the repository history at commit `bc1ef53`.
