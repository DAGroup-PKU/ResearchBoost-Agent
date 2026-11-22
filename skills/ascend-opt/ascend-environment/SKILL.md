---
name: ascend-environment
description: Inspect Ascend NPU devices, CANN/OPP setup and architecture, and configure the selected devices in a development container.
---

# Ascend NPU environment

Use for environment readiness, architecture identification or NPU container setup before working on a workload. Record the observed device, versions and configuration so that a baseline and its candidate use the same allocated resources.

1. Inspect the allocated devices, driver, CANN toolkit and OPP paths; distinguish missing software from missing container device mappings. For practical checks, see [device and toolkit setup](#device-and-toolkit-readiness).
2. Identify both runtime `SocVersion` and `NpuArch` using the target SDK. Use architecture information to choose the correct compiler and operator branch; do not infer a full chip model from `npu-smi`'s generic Chip Name. See [architecture and containers](#architecture-and-container-setup).
3. For a container, choose an image matching the workload and host driver, pass only the assigned NPU devices and required Ascend manager devices and driver libraries, and run the same readiness checks inside it. Map profiling/log directories only when the workflow needs them.
4. Validate the workload's existing minimal execution path. If a collective fails or dominates the workload, consult [ascend-communication](../ascend-communication/SKILL.md); investigate runtime or numerical failure using the applicable debugging guidance.

## Contents

- [Device and toolkit readiness](#device-and-toolkit-readiness)
- [Architecture and container setup](#architecture-and-container-setup)

## Device and toolkit readiness

### Inspect the assigned device

```bash
npu-smi info
# Optional: inspect utilization for an assigned physical device ID.
npu-smi info -t usages -i <device_id>
```

`npu-smi` reports device inventory and running processes. A small amount of HBM may be used by the driver on an otherwise idle device; use process and utilization evidence rather than memory usage alone to infer occupancy. Device IDs exposed in a container may differ from the host. Reconcile physical IDs, logical visible IDs, ranks and the scheduler allocation before using them.

If the executable or device is absent, check the host driver and the container device mounts; no device visible inside the container does not by itself prove a hardware fault. Inspect the error and the first failing step rather than installing or updating packages speculatively.

### CANN toolkit and operators

Determine the installation path used by the current workload, not a different toolkit installed on the host. The standard toolkit setup is sourced from its installation directory, for example:

```bash
# Replace with the toolkit installation path for this environment.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
printf 'Toolkit: %s\nOPP: %s\n' "$ASCEND_HOME_PATH" "$ASCEND_OPP_PATH"
```

The toolkit path is normally exposed by `ASCEND_HOME_PATH`; operator packages use `ASCEND_OPP_PATH`. Confirm these values and the corresponding directories/libraries inside the process environment. For a custom operator, also verify its installed operator package and `op_api` library. Tools such as `msprof` are required only for profiling, not for every runtime setup.

Record the NPU/SoC, driver and CANN toolkit versions, OPP and custom operator packages, Python and framework/`torch_npu` versions, and the image tag if applicable. Compare the selected framework branch's compatibility requirements with the actually installed combination. Preserve exact versions in baseline and candidate runs.

### Capture the first failure

- If the driver device is visible on the host but not in a container, review the selected `/dev/davinci<N>` devices, Ascend manager nodes and driver mounts first.
- If CANN is missing inside a container, check the image and toolkit environment rather than assuming host environment variables were inherited.
- If an operator cannot load, distinguish the toolkit/OPP path from the custom `op_api` package and look for the first shared-library or registration error.
- Validate a small existing workload invocation with the same execution environment before proceeding to profiling or code changes.

## Architecture and container setup

### Architecture names serve different purposes

- `SocVersion` identifies a software SoC version. `NpuArch` identifies an architecture; the device-side `__NPU_ARCH__` macro is a numeric compile-time condition.
- Names such as `arch35` are repository directory shorthand; compiler architecture tokens and build-system macros are different identifiers. Do not derive one by spelling alone.
- When working with Ascend C host tiling code, `PlatformAscendC(context->GetPlatformInfo())` exposes `GetCurNpuArch()` and `GetSocVersion()`. The returned reserved values (`DAV_RESV` and `RESERVED_VERSION`) indicate identification failure. Check the actual target SDK header and version before relying on the API or enum names.
- The compiler mapping associates `DAV_3510` with `arch35` and `__NPU_ARCH__=3510`, but product/SKU and compiler mappings require target-specific verification. `npu-smi` Chip Name can be generic on some A3 systems; do not treat it as an authoritative full SoC identifier.
- Read core counts, UB capacity and supported instructions from the actual device/SDK for tiling decisions; do not assume capacities are interchangeable across DAV generations.

For full SoC and architecture identification, check the Chip Info and Arch Info fields from `asys info -r=hardware`. You can also use driver/DSMI information and match the full SoC name against the installed configuration. Preserve the evidence source and report unresolved fields rather than deriving them from a similar name. Set up the target CANN environment before using toolkit utilities.

### Container mappings

Use the workload's documented Ascend-compatible image and verify host driver/container toolkit compatibility. Typical device nodes include the following:

| Host resource | Purpose |
|---|---|
| `/dev/davinci<N>` | A particular allocated NPU |
| `/dev/davinci_manager` | Device manager |
| `/dev/devmm_svm` | Device memory management |
| `/dev/hisi_hdc` | Host-device communication |
| `/usr/local/Ascend/driver` | Driver libraries, typically mounted read-only |
| `/usr/local/sbin/npu-smi` | Device inspection utility, when exposed |

A minimal selected-device example, assuming the first NPU is allocated and the paths exist on the host, is:

```bash
docker run --rm -it \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  <compatible-image> /bin/bash
```

Replace the image and device node with the actual assigned values. Device and driver layouts, mount permissions, runtime libraries and NPU profiling/dump access depend on the installed driver and container runtime; adapt them using the local platform's documented setup. Do not assume basic device access automatically provides profiling or device dumps. Inside the container run `npu-smi info`, inspect toolkit/OPP paths, and test the real workload's Python environment and device visibility.
