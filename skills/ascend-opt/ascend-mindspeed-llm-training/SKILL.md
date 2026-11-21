---
name: ascend-mindspeed-llm-training
description: Run the Ascend MindSpeed-LLM environment, model/weight, data and distributed-training pipeline for a selected model and backend.
---

# Ascend MindSpeed-LLM training

1. Choose the model and Megatron/FSDP2 backend from the checked-out MindSpeed-LLM branch's supported examples and software pairing table.
2. Follow environment, weight, data and training guides; keep architecture, tokenizer and parallel settings consistent across stages.
3. For export, use the checkpoint actually produced by training. The pipeline distinguishes full-weight conversion from LoRA base-plus-adapter merge.

## Working references

Read the reference for the current phase; the original technical procedures, examples and resources are retained below. Resolve a documented script relative to its reference guide's directory.

- [mindspeed-llm-pipeline](references/native/mindspeed-llm-pipeline/GUIDE.md)
- [mindspeed-llm-env-setup](references/native/mindspeed-llm-env-setup/GUIDE.md)
- [mindspeed-llm-weight-prep](references/native/mindspeed-llm-weight-prep/GUIDE.md)
- [mindspeed-llm-data-prep](references/native/mindspeed-llm-data-prep/GUIDE.md)
- [mindspeed-llm-training](references/native/mindspeed-llm-training/GUIDE.md)
