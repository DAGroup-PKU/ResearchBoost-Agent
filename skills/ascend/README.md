# Ascend backend skills

Twenty task-oriented Ascend cards supplement the nine existing ResearchBoost cards. The original measurement, action-ladder and numerical-validation cards continue to govern the task; its frozen evaluator remains the verdict. This adds backend knowledge and workflows, not a new benchmark or a new NPU judge.

Choose the current task from the tree, then read its card and the referenced phase. Upstream phase manuals are retained inside each card as references, without separate skill registrations or platform-specific dispatchers. High task relatedness is used for navigation; distinct procedures are not discarded as duplicates.

## Task tree

```text
ascend/
├── environment/
│   └── ascend-npu-environment/
├── migration/
│   ├── ascend-pytorch-migration/
│   └── ascend-fsdp2-migration/
├── training/
│   ├── ascend-mindspeed-llm-training/
│   ├── ascend-mindspeed-mm-training/
│   └── ascend-verl-training/
├── deployment/
│   ├── ascend-om-deployment/
│   └── ascend-vllm-quantized-serving/
├── operators/
│   ├── ascendc-api-guide/
│   ├── ascendc-operator-project/
│   ├── ascendc-pytorch-integration/
│   └── ascendc-kernel-tuning/
├── profiling/
│   ├── ascend-operator-profiling/
│   ├── ascend-profiler-integration/
│   └── ascend-trace-analysis/
└── debugging/
    ├── ascend-operator-accuracy/
    ├── ascend-runtime-debug/
    ├── ascend-memory-debug/
    ├── ascend-npugraph-debug/
    └── ascend-training-debug/
```

## Ascend NPU environment

| Card | Use for | Original entries |
|---|---|---|
| [ascend-npu-environment](environment/ascend-npu-environment/SKILL.md) | Inspect assigned Ascend devices, architecture, CANN/OPP and development or HCCL prerequisites. | <sub>`ascendc-env-check`, `npu-arch`, `ascend-docker`, `hccl-test`</sub> |

## Ascend model migration

| Card | Use for | Original entries |
|---|---|---|
| [ascend-pytorch-migration](migration/ascend-pytorch-migration/SKILL.md) | Migrate PyTorch training or inference code to torch_npu, preserving device, dependency and input/output contracts. | <sub>`migration-ascend-torchnpu-skills`, `model-infer-migrator`</sub> |
| [ascend-fsdp2-migration](migration/ascend-fsdp2-migration/SKILL.md) | Adapt model registration, datasets and configuration to MindSpeed-MM FSDP2 and verify distributed functional execution. | <sub>`mindspeed-fsdp2-config-migration`, `mindspeed-fsdp2-data-migration`, `mindspeed-fsdp2-migration-main`, `mindspeed-fsdp2-model-migration`, `mindspeed-fsdp2-verification`</sub> |

## Ascend framework training

| Card | Use for | Original entries |
|---|---|---|
| [ascend-mindspeed-llm-training](training/ascend-mindspeed-llm-training/SKILL.md) | Run the Ascend MindSpeed-LLM environment, model/weight, data and distributed-training pipeline for a selected model and backend. | <sub>`mindspeed-llm-pipeline`, `mindspeed-llm-env-setup`, `mindspeed-llm-weight-prep`, `mindspeed-llm-data-prep`, `mindspeed-llm-training`</sub> |
| [ascend-mindspeed-mm-training](training/ascend-mindspeed-mm-training/SKILL.md) | Deploy MindSpeed-MM understanding or generative workloads on Ascend with model-specific dependencies, data and weight conversion. | <sub>`mindspeed-mm-pipeline`, `mindspeed-mm-env-setup`, `mindspeed-mm-weight-prep`, `mindspeed-mm-vlm`, `mindspeed-mm-generative`</sub> |
| [ascend-verl-training](training/ascend-verl-training/SKILL.md) | Prepare a compatible Ascend VERL environment, data and model, then launch a selected training/rollout backend combination. | <sub>`ascend-verl-training`, `ascend-verl-env-preparation`, `ascend-verl-image-list`, `ascend-verl-prepare-data`, `ascend-verl-model-download`, `ascend-verl-env-check`</sub> |

## Ascend inference deployment

| Card | Use for | Original entries |
|---|---|---|
| [ascend-om-deployment](deployment/ascend-om-deployment/SKILL.md) | Analyze source IO/state contracts, convert an appropriate ONNX subgraph with ATC and reconnect the OM path into the application. | <sub>`ascend-om-deployer`, `ascend-onnx-atc-pipeline`, `ascend-om-pipeline-adapter`, `tool-atc`</sub> |
| [ascend-vllm-quantized-serving](deployment/ascend-vllm-quantized-serving/SKILL.md) | Inspect existing msModelSlim or compressed-tensors weights and start a single-instance vLLM-Ascend service with format-specific parameters. | <sub>`vllm-ascend-quant-serving`</sub> |

## Ascend C implementation

| Card | Use for | Original entries |
|---|---|---|
| [ascendc-api-guide](operators/ascendc-api-guide/SKILL.md) | Resolve Ascend C API overloads, DMA alignment and units, vector/Cube operations, buffers and synchronization. | <sub>`ascendc-api-best-practices`, `ascendc-docs-search`</sub> |
| [ascendc-operator-project](operators/ascendc-operator-project/SKILL.md) | Create a direct-launch or registered Ascend C operator project with matching build, tiling and test resources. | <sub>`ascendc-direct-invoke-template`, `ascendc-registry-invoke-template`</sub> |
| [ascendc-pytorch-integration](operators/ascendc-pytorch-integration/SKILL.md) | Connect a working Ascend C direct-launch or ACLNN operator to PyTorch with the corresponding schema and graph interface. | <sub>`torch-ascendc-op-extension`</sub> |
| [ascendc-kernel-tuning](operators/ascendc-kernel-tuning/SKILL.md) | Tune an Ascend C hotspot through target-specific tiling, buffer budgets and applicable single-core or inter-core pipelines. | <sub>`ascendc-perf-optimize`, `ascendc-tiling-design`, `ascendc-performance-best-practices`, `ascendc-regbase-best-practice`, `ascendc-blaze-best-practice`, `ascendc-blaze-migration`</sub> |

## Ascend profiling tools

| Card | Use for | Original entries |
|---|---|---|
| [ascend-operator-profiling](profiling/ascend-operator-profiling/SKILL.md) | Collect and interpret msprof or msOpProf operator traces, counters and pipeline evidence. | <sub>`ops-profiling`</sub> |
| [ascend-profiler-integration](profiling/ascend-profiler-integration/SKILL.md) | Integrate torch_npu.profiler into training, inference or RL code with framework-specific injection and control examples. | <sub>`ascend-profiler-collect-adaption`</sub> |
| [ascend-trace-analysis](profiling/ascend-trace-analysis/SKILL.md) | Interpret Ascend kernel_details/trace artifacts, AICPU exposure, device gaps and model-component execution. | <sub>`ascend-profiling-anomaly`, `model-infer-perf-breakdown`</sub> |

## Ascend accuracy and runtime debugging

| Card | Use for | Original entries |
|---|---|---|
| [ascend-operator-accuracy](debugging/ascend-operator-accuracy/SKILL.md) | Locate Ascend operator mismatches, tail/alignment faults and unstable intermediate values, then verify a targeted kernel repair. | <sub>`ascendc-precision-debug`</sub> |
| [ascend-runtime-debug](debugging/ascend-runtime-debug/SKILL.md) | Investigate ACLNN/CANN failures, kernel crashes or hangs using plog, exception dumps and AI Core fault evidence. | <sub>`ascendc-runtime-debug`, `ascendc-crash-debug`, `tool-msaicerr`, `tool-asys`, `tool-exception-dump`, `npu-objdump-toolkit`</sub> |
| [ascend-memory-debug](debugging/ascend-memory-debug/SKILL.md) | Run supported Ascend C or Triton sanitizer checks for bounds, initialization, races and synchronization faults. | <sub>`ops-mssanitizer`</sub> |
| [ascend-npugraph-debug](debugging/ascend-npugraph-debug/SKILL.md) | Diagnose npugraph_ex/aclgraph capture, replay and graph-generated copy/clone performance regressions. | <sub>`torch-npugraph-ex-dfx-triage`, `torch-npugraph-ex-compile-error-diagnosis`, `torch-npugraph-ex-runtime-error-diagnosis`, `torch-npugraph-ex-performance-diagnosis`, `torch-npugraph-ex-knowledge`, `torch-npugraph-ex-template`</sub> |
| [ascend-training-debug](debugging/ascend-training-debug/SKILL.md) | Locate Ascend training OOM, nonfinite values, finite reference mismatch or repeated-run nondeterminism using the relevant evidence. | <sub>`model-train-oom-analysis`, `model-train-precision-nonfinite`, `model-train-precision-numerical-mismatch`, `model-train-precision-determinism`, `model-train-log-visualization`</sub> |

## Sources and adaptation

See [SOURCES.md](SOURCES.md) for pinned upstream revisions, notices and the import map. Technical reference bodies and supporting examples/scripts are retained. Local changes group phase entry points, replace brittle cross-skill dispatch/path requirements with document subjects, and correct concrete examples. The card's own reference/resource links remain direct so they can be found and used.

The training cards use the selected framework branch's model examples and software pairing rather than assuming that a pinned example's version matrix is current. The VERL card includes the recently verified official quick-start reference.
