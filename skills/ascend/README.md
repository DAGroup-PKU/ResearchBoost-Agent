# Ascend performance optimization skills

Twenty task-oriented skills for performance optimization agents working on Huawei Ascend NPUs. Select a task below and load only its relevant phase references. All twenty skill folders are peers. Deployment includes environment preparation, framework training and inference serving.

## Deployment · environment, training and serving

| Skill | Purpose |
|---|---|
| [ascend-npu-environment](ascend-npu-environment/SKILL.md) | Inspect assigned Ascend devices, architecture, CANN/OPP and development or HCCL prerequisites. |
| [ascend-mindspeed-llm-training](ascend-mindspeed-llm-training/SKILL.md) | Run the Ascend MindSpeed-LLM environment, model/weight, data and distributed-training pipeline for a selected model and backend. |
| [ascend-mindspeed-mm-training](ascend-mindspeed-mm-training/SKILL.md) | Deploy MindSpeed-MM understanding or generative workloads on Ascend with model-specific dependencies, data and weight conversion. |
| [ascend-verl-training](ascend-verl-training/SKILL.md) | Prepare a compatible Ascend VERL environment, data and model, then launch a selected training/rollout backend combination. |
| [ascend-om-deployment](ascend-om-deployment/SKILL.md) | Analyze source IO/state contracts, convert an appropriate ONNX subgraph with ATC and reconnect the OM path into the application. |
| [ascend-vllm-quantized-serving](ascend-vllm-quantized-serving/SKILL.md) | Inspect existing msModelSlim or compressed-tensors weights and start a single-instance vLLM-Ascend service with format-specific parameters. |

## Migration

| Skill | Purpose |
|---|---|
| [ascend-pytorch-migration](ascend-pytorch-migration/SKILL.md) | Migrate PyTorch training or inference code to torch_npu, preserving device, dependency and input/output contracts. |
| [ascend-fsdp2-migration](ascend-fsdp2-migration/SKILL.md) | Adapt model registration, datasets and configuration to MindSpeed-MM FSDP2 and verify distributed functional execution. |

## Ascend C operators

| Skill | Purpose |
|---|---|
| [ascendc-api-guide](ascendc-api-guide/SKILL.md) | Resolve Ascend C API overloads, DMA alignment and units, vector/Cube operations, buffers and synchronization. |
| [ascendc-operator-project](ascendc-operator-project/SKILL.md) | Create a direct-launch or registered Ascend C operator project with matching build, tiling and test resources. |
| [ascendc-pytorch-integration](ascendc-pytorch-integration/SKILL.md) | Connect a working Ascend C direct-launch or ACLNN operator to PyTorch with the corresponding schema and graph interface. |
| [ascendc-kernel-tuning](ascendc-kernel-tuning/SKILL.md) | Tune an Ascend C hotspot through target-specific tiling, buffer budgets and applicable single-core or inter-core pipelines. |

## Profiling

| Skill | Purpose |
|---|---|
| [ascend-operator-profiling](ascend-operator-profiling/SKILL.md) | Collect and interpret msprof or msOpProf operator traces, counters and pipeline evidence. |
| [ascend-profiler-integration](ascend-profiler-integration/SKILL.md) | Integrate torch_npu.profiler into training, inference or RL code with framework-specific injection and control examples. |
| [ascend-trace-analysis](ascend-trace-analysis/SKILL.md) | Interpret Ascend kernel_details/trace artifacts, AICPU exposure, device gaps and model-component execution. |

## Accuracy and runtime debugging

| Skill | Purpose |
|---|---|
| [ascend-operator-accuracy](ascend-operator-accuracy/SKILL.md) | Locate Ascend operator mismatches, tail/alignment faults and unstable intermediate values, then verify a targeted kernel repair. |
| [ascend-runtime-debug](ascend-runtime-debug/SKILL.md) | Investigate ACLNN/CANN failures, kernel crashes or hangs using plog, exception dumps and AI Core fault evidence. |
| [ascend-memory-debug](ascend-memory-debug/SKILL.md) | Run supported Ascend C or Triton sanitizer checks for bounds, initialization, races and synchronization faults. |
| [ascend-npugraph-debug](ascend-npugraph-debug/SKILL.md) | Diagnose npugraph_ex/aclgraph capture, replay and graph-generated copy/clone performance regressions. |
| [ascend-training-debug](ascend-training-debug/SKILL.md) | Locate Ascend training OOM, nonfinite values, finite reference mismatch or repeated-run nondeterminism using the relevant evidence. |

## Sources

Phase manuals, scripts and examples remain available within each skill’s references. Several entries combine multiple phases; specialist entries can retain one source skill. LLM and MM retain distinct workflows. See [SOURCES.md](SOURCES.md), [upstream.json](upstream.json) and [licenses/](licenses/) for pinned source revisions and attribution.
