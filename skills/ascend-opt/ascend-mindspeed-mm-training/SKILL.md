---
name: ascend-mindspeed-mm-training
description: Deploy MindSpeed-MM understanding or generative workloads on Ascend with model-specific dependencies, data and weight conversion.
---

# Ascend MindSpeed-MM training

1. Use the model router to select understanding/VLM or generative training, then verify the supported backend and example on the framework branch.
2. Apply that model's dependency, dataset and checkpoint contracts using the environment and weight guides.
3. For a new FSDP2 model rather than launching an existing example, consult the model/data/config migration category.

## Working references

Read the reference for the current phase; the original technical procedures, examples and resources are retained below. Resolve a documented script relative to its reference guide's directory.

- [mindspeed-mm-pipeline](references/native/mindspeed-mm-pipeline/GUIDE.md)
- [mindspeed-mm-env-setup](references/native/mindspeed-mm-env-setup/GUIDE.md)
- [mindspeed-mm-weight-prep](references/native/mindspeed-mm-weight-prep/GUIDE.md)
- [mindspeed-mm-vlm](references/native/mindspeed-mm-vlm/GUIDE.md)
- [mindspeed-mm-generative](references/native/mindspeed-mm-generative/GUIDE.md)
