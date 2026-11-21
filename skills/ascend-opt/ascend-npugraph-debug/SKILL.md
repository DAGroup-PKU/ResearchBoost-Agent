---
name: ascend-npugraph-debug
description: Diagnose npugraph_ex/aclgraph capture, replay and graph-generated copy/clone performance regressions.
---

# Ascend npugraph_ex debugging

1. Identify the actual graph backend, then use the triage guide for npugraph_ex/aclgraph evidence.
2. Choose compile, runtime or performance analysis from that evidence. For changed results, investigate model/operator accuracy comparison and debugging.

## Working references

Read the reference for the current phase; the original technical procedures, examples and resources are retained below. Resolve a documented script relative to its reference guide's directory.

- [torch-npugraph-ex-dfx-triage](references/native/torch-npugraph-ex-dfx-triage/GUIDE.md)
- [torch-npugraph-ex-compile-error-diagnosis](references/native/torch-npugraph-ex-compile-error-diagnosis/GUIDE.md)
- [torch-npugraph-ex-runtime-error-diagnosis](references/native/torch-npugraph-ex-runtime-error-diagnosis/GUIDE.md)
- [torch-npugraph-ex-performance-diagnosis](references/native/torch-npugraph-ex-performance-diagnosis/GUIDE.md)
- [torch-npugraph-ex-knowledge](references/native/torch-npugraph-ex-knowledge/GUIDE.md)
- [torch-npugraph-ex-template](references/native/torch-npugraph-ex-template/GUIDE.md)
