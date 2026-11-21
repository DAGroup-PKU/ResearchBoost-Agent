# Current Ascend VERL quick-start reference

The official [Ascend quick-start guide](https://verl.readthedocs.io/en/latest/ascend_tutorial/en/get_start/quick_start.html) was checked on 2026-11-21 and states an update date of 2026-11-12. It covers A2/A3 GRPO startup with Qwen3-0.6B and GSM8K.

Use the matching launch example from the checked-out VERL branch:

| Training backend | Rollout backend | Example basename under `tests/special_npu/quick_start/` |
|---|---|---|
| FSDP2 | vLLM-Ascend | `run_qwen3_0_6b_fsdp2_vllm_ascend.sh` |
| Megatron | vLLM-Ascend | `run_qwen3_0_6b_megatron_vllm_ascend.sh` |
| FSDP2 | SGLang | `run_qwen3_0_6b_fsdp2_sglang_ascend.sh` |
| Megatron | SGLang | `run_qwen3_0_6b_megatron_sglang_ascend.sh` |

Select the image and backend pair together. Point model/data settings to actual assets, activate the installed CANN environment and use the branch's documented launch arguments. The bundled phase guides provide the preparation procedure; they are documents to apply, not shell commands or a platform-specific skill API. The first completed training step is startup evidence, while the ResearchBoost evaluator determines optimization acceptance.
