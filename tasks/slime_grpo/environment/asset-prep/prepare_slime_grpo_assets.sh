#!/bin/bash
# Provision RESEARCHBOOST_ASSETS_ROOT/slime_grpo on a host (public files only, pinned revisions, checked against
# SHA256SUMS next to this script), then convert the checkpoint for Megatron with the task image:
#   Qwen3.5-9B/               Hugging Face Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a
#   data/dapo-math-17k.jsonl  Hugging Face dataset zhuzilin/dapo-math-17k @ 2e65612930298bde4c5d58fd97b3f23a483aaff9
#                             (the dataset of upstream examples/fully_async/run-qwen3.5-9B-fully_async.sh)
#   Qwen3.5-9B_torch_dist/    tools/convert_hf_to_torch_dist.py of Qwen3.5-9B
#                             of the reference slime tree (scripts/models/qwen3.5-9B.sh) with the reference Megatron-LM, on one GPU
# Usage: prepare_slime_grpo_assets.sh <task image> <assets root> [gpu index, default 0]
# HF_ENDPOINT selects a Hugging Face mirror (default https://huggingface.co).
set -euo pipefail
image=$1; root=$2; gpu=${3:-0}
here=$(cd "$(dirname "$0")" && pwd)
D=$root/slime_grpo; E=${HF_ENDPOINT:-https://huggingface.co}
mkdir -p "$D/Qwen3.5-9B" "$D/data"
M=Qwen/Qwen3.5-9B; MR=c202236235762e1c871ad0ccb60c8ee5ba337b9a
for f in $(awk "{print \$2}" "$here/SHA256SUMS" | grep "^Qwen3.5-9B/" | sed "s|^Qwen3.5-9B/||"); do
  [ -s "$D/Qwen3.5-9B/$f" ] || { curl -fsSL --retry 5 -C - -o "$D/Qwen3.5-9B/$f.part" "$E/$M/resolve/$MR/$f" \
                                 && mv "$D/Qwen3.5-9B/$f.part" "$D/Qwen3.5-9B/$f"; }
done
[ -s "$D/data/dapo-math-17k.jsonl" ] || curl -fsSL --retry 5 -o "$D/data/dapo-math-17k.jsonl" \
  "$E/datasets/zhuzilin/dapo-math-17k/resolve/2e65612930298bde4c5d58fd97b3f23a483aaff9/dapo-math-17k.jsonl"
(cd "$D" && sha256sum -c --quiet "$here/SHA256SUMS")
if [ ! -f "$D/Qwen3.5-9B_torch_dist/latest_checkpointed_iteration.txt" ]; then
  rm -rf "$D/Qwen3.5-9B_torch_dist.part"
  docker run --rm --gpus "\"device=$gpu\"" --shm-size 64g --entrypoint bash -v "$D:/out" "$image" -c "
set -euo pipefail
driver=\$(nvidia-smi --query-gpu=driver_version --format=csv,noheader); driver=\${driver%%\$'\\n'*}
case \${driver%%.*} in 535|550|565|570|575) echo /usr/local/cuda/compat > /etc/ld.so.conf.d/000-cuda-compat.conf; ldconfig ;; esac
cd /opt/researchboost/ws/targets/reference
source scripts/models/qwen3.5-9B.sh
PYTHONPATH=/opt/researchboost/ws/targets/reference-Megatron-LM:/opt/researchboost/ws/targets/reference \
  python3 tools/convert_hf_to_torch_dist.py \"\${MODEL_ARGS[@]}\" --hf-checkpoint /out/Qwen3.5-9B \
  --save /out/Qwen3.5-9B_torch_dist.part
chmod -R a+rX /out/Qwen3.5-9B_torch_dist.part"
  mv "$D/Qwen3.5-9B_torch_dist.part" "$D/Qwen3.5-9B_torch_dist"
fi
chmod -R a+rX "$D"
du -sh "$D"/*
