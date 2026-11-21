---
name: ascend-profiling
description: Collect and analyze Ascend NPU model or operator traces to locate optimization bottlenecks with torch_npu.profiler, msprof or msOpProf.
---

# Ascend profiling

Use the existing RBA tiered profiling and measurement cards to select a representative phase and define the workload. Ascend-specific traces explain where time goes; the frozen task evaluator determines correctness and performance.

## Model and training traces

1. Record NPU, software versions, process/rank, input shape, dtype, parallelism and exact workload window. Profile the process that actually executes NPU work, not a controller that only schedules it.
2. Prefer an existing framework profiler hook. If none exists, instrument the actual training or inference loop using the installed `torch_npu.profiler` version's documented schedule and trace handler. Include the relevant forward, backward and optimizer steps; do not alter their semantics. Keep collection bounded and call the profiler step at the framework's intended iteration boundary.
3. Inspect the output actually requested by the chosen export mode. Text exports can contain `trace_view.json`, `op_statistic.csv` and `kernel_details.csv`; DB exports use a profiler database. Check timestamps, ranks and completeness before drawing conclusions. Missing metrics are unavailable, not zero.
4. Start with the task phase and rank, then examine operator/kernel time, CPU-to-device gaps, synchronization, copies and communication. Compare like windows and avoid summing overlapping streams as elapsed time. Use timeline context to distinguish launch gaps from long kernels.
5. Use available counters to test a concrete bandwidth, compute, communication or launch-overhead hypothesis. Confirm a hotspot on a representative input, then measure the change with the task's own evaluator outside profiler overhead.

## Operator traces

Use the installed CANN version's `msprof`/`msprof op` or msOpProf workflow when a measured operator needs kernel-level detail. Check SoC, tool availability and supported metrics before collection. Record command, input shape, repetitions and generated artifacts; use timeline/pipe and memory counters only when provided. A microbenchmark is diagnostic and cannot establish whole-workload benefit.

References: [torch_npu profiler collection](https://gitcode.com/Ascend/agent-skills/tree/25ff511dc01e4db903e5ffdfd178425671881086/official/MindStudio/skills/profiler/ascend-profiler-collect-adaption), [operator profiling](https://gitcode.com/cann/cannbot-skills/tree/2af66d732e595297cf34159cf8e1d49fa7354c2c/ops/ops-profiling), [trace anomaly analysis](https://gitcode.com/Ascend/agent-skills/tree/25ff511dc01e4db903e5ffdfd178425671881086/official/Common/ascend-profiling-anomaly).
