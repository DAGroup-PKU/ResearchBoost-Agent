---
name: ascendc-pytorch-integration
description: Connect a working Ascend C direct-launch or ACLNN operator to PyTorch with the corresponding schema and graph interface.
---

# Ascend C PyTorch integration

1. Follow the route matching the existing operator project and actual PyTorch caller.
2. Apply its schema, Meta/fake and stream/queue conventions; use accuracy comparison and debugging guidance when integration changes results.

## Working references

Read the reference for the current phase; the original technical procedures, examples and resources are retained below. Resolve a documented script relative to its reference guide's directory.

- [torch-ascendc-op-extension](references/native/torch-ascendc-op-extension/GUIDE.md)
