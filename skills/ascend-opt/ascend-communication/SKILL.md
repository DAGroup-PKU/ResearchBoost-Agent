---
name: ascend-communication
description: Diagnose and measure HCCL collective communication on allocated Ascend NPUs using the installed HCCL test tool.
---

# Ascend HCCL communication

Use for a distributed workload with suspected collective failure or communication bottleneck. First profile the actual training run to identify the rank group, collective, payload size and time window. Run a focused HCCL test only on assigned devices with the compatible toolkit/MPI version; its result diagnoses the communication path and is not a whole-workload performance verdict.

1. Record the hardware, driver/CANN version, rank-to-device mapping, node topology, MPI version, collective, data type and payload range. Check device readiness and CANN compatibility in the environment skill.
2. Confirm the installed `hccl_test` binary and its options for the current CANN version. Choose AllReduce, AllGather or another collective based on the observed workload. See [HCCL test procedure](#hccl-test-procedure) for a bounded example and interpretation.
3. Check correctness as well as latency/bandwidth, investigate first errors per rank, and compare the same topology and payload sizes. An isolated collective test cannot prove communication/computation overlap or end-to-end speedup.
4. Make the smallest workload-side change suggested by the evidence, then verify semantics and use the task's existing evaluator for final correctness and timing.

## Contents

- [HCCL test procedure](#hccl-test-procedure)
  - [Select a representative test](#select-a-representative-test)
  - [Example: eight allocated devices on one node](#example-eight-allocated-devices-on-one-node)
  - [Interpret and act](#interpret-and-act)

## HCCL test procedure

### Select a representative test

Look for `${INSTALL_DIR}/tools/hccl_test` in the CANN installation. Installed paths, supported devices, MPI and CLI flags vary by release: inspect the local installation and its help before using the examples. Match the real training collective and tensor size; AllReduce is common for gradient aggregation, AllGather for gathering parameters, and ReduceScatter / AlltoAll may be relevant to other distributed layouts.

Check the allocated device/rank mapping, comparable CANN and driver configuration across participating nodes, interface configuration and NPU health. Use the existing launcher and account permissions for a multi-node job. A hostfile and accessible node setup are launcher-specific; do not assume MPICH and Open MPI accept identical flags. Do not change shared-node network or SSH configuration just to perform a benchmark.

If the tool is supplied as source and the local CANN distribution instructs building it, use the following build command, with the actual MPI and Ascend toolkit directories substituted:

```bash
cd "${INSTALL_DIR}/tools/hccl_test"
make MPI_HOME="${MPI_HOME}" ASCEND_DIR="${INSTALL_DIR}"
```

### Example: eight allocated devices on one node

The following source-based example uses MPICH-style `mpirun`; verify the tool's installed options first and substitute the actual allocation. Start with small payloads and validation enabled:

```bash
mpirun -n 8 ./bin/all_reduce_test \
  -p 8 -b 8K -e 64M -f 2 -d fp32 -o sum -c 1
```

- `mpirun -n` is the total number of ranks; `-p` is the number of participating NPUs per node for this interface. The example assumes one process per NPU and eight assigned NPUs on one node.
- `-b` and `-e` select minimum and maximum payload; `-f 2` doubles it at each step. Use a fixed workload size with matching `-b` and `-e` when investigating a specific collective.
- `-d` selects dtype, `-o` the reduction operation, and `-c 1` requests result checking. Supported options and defaults must match the installed version.
- For multiple nodes, use the job's MPI-specific hostfile/launcher and consistent rank mapping. Do not copy a root-based SSH example or a network address from an unrelated system.

### Interpret and act

For each payload, record average collective time, reported algorithm bandwidth and correctness result. The test reports algorithm bandwidth as data size divided by time; compare like units, payload definitions, rank counts, dtypes and topology. A failed or missing correctness result is not evidence of a valid fast collective. Record the first failure on every affected rank with CANN/HCCL logs and timestamps.

A bandwidth plateau or a slow isolated collective suggests where to investigate, but does not establish end-to-end benefit. Correlate the collective with the workload timeline, including scheduling and overlap. Re-run comparable cases after a targeted change, then check the original workload with its frozen numerical and timing contract.
