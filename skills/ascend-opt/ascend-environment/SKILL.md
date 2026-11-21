---
name: ascend-environment
description: Inspect and prepare an assigned Ascend NPU environment for an existing optimization task, including versions, devices and distributed communication.
---

# Ascend environment for an optimization run

Record the target NPU model, allocated device IDs, driver, firmware, CANN, torch/torch_npu and Python versions. Check the framework's support matrix before installing or changing any package. Use the job's assigned devices and directories.

1. Inspect `npu-smi info` and the existing task/container configuration. Map visible logical devices to the allocation; do not assume host IDs match container IDs.
2. Identify CANN and OPP paths from the active environment and check that the installed toolkit, driver and framework builds are compatible with the selected SoC. Confirm the target inside the execution environment, not just on the host.
3. Run the task's existing minimal functional path before changing code. Preserve model, data, dtype, shape, seed, parallelism and launch parameters as the baseline. Report exact version/shape and failure boundary when initialization fails.
4. For a container, use the project's approved image and device/driver mounts. Verify availability from inside the container. For distributed jobs, use the framework's existing HCCL smoke test on the assigned ranks; record rank-to-device mapping and the earliest per-rank error. Avoid changing cluster-wide settings.
5. Once functional, use the task's frozen measurement and numerical contracts. Profiling and environment checks diagnose; they do not replace the judge.

See pinned [CANNBot environment guide](https://gitcode.com/cann/cannbot-skills/tree/2af66d732e595297cf34159cf8e1d49fa7354c2c/ops/ascendc-env-check) and [Ascend container guidance](https://gitcode.com/Ascend/agent-skills/tree/25ff511dc01e4db903e5ffdfd178425671881086/official/Common/ascend-docker) for version-specific commands.
