---
name: ascend-deployment-migration
description: Adapt or deploy an existing PyTorch training or inference workload on Ascend NPU, including torch_npu, MindSpeed and VERL setup.
---

# Ascend deployment and migration

Use the checked-out workload's documented launch path and supported version matrix. Record SoC, CANN, driver, torch/torch_npu, framework revision, image, dataset and model revision. Make the smallest change that makes the original workload run on the assigned NPU.

## PyTorch migration

- Find CUDA-specific device selection, streams/events, synchronization, distributed backends, fused/custom operators and hard-coded device assumptions. Replace only where the installed torch_npu and workload's supported APIs require it; check the target-version documentation for availability and semantics.
- Preserve input/output contracts, dtype, layout, random-state behavior and distributed topology. Keep the same model path and task-level inputs for baseline and candidate. Verify a representative forward, backward and optimizer step if the task trains.
- Validate per-rank device assignment and collective initialization. Profile after functional correctness; do not equate successful startup with numerical equivalence or speedup.

## Frameworks and model artifacts

- For MindSpeed-LLM or MindSpeed-MM, choose the actual framework branch's supported model/backend example and software combination. Preserve tokenizer, model config, model/data parallel settings and checkpoint format across conversion, training and export. Export the checkpoint produced by training; LoRA requires the correct base and adapter, not an unrelated checkpoint.
- For VERL, select a documented training/rollout backend pair and the matching image. Prepare the dataset and model in the paths expected by the selected example. Capture the launched command, process status and logs. Treat a quick start as a functional smoke test, not performance evidence.
- For ONNX to OM or quantized vLLM serving, use the existing framework's documented conversion/serving recipe for its CANN version and SoC. Check shape/dtype, dynamic dimensions, input/output order, state/cache behavior and weight format before integrating it into the application. Do not infer general deployment support from a quantized-weight example.

If a result differs, investigate the accuracy and runtime debugging skill. If it is slower, use the Ascend profiling skill and the existing RBA measurement card. Keep the task's frozen numerical and timing verdicts.

References: [torch_npu migration](https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/75ec89ce9d401be8ac95980e1c3f42f8d4315057/skills/inference/migration-ascend-torchnpu-skills), [MindSpeed-LLM](https://github.com/Ascend/MindSpeed-LLM), [MindSpeed-MM](https://github.com/Ascend/MindSpeed-MM), [VERL Ascend quick start](https://verl.readthedocs.io/en/latest/ascend_tutorial/en/get_start/quick_start.html), [OM deployment](https://gitcode.com/Ascend/agent-skills/tree/25ff511dc01e4db903e5ffdfd178425671881086/official/Common/ascend-om-deployer).
