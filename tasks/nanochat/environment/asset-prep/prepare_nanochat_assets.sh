#!/bin/bash
# Provision RESEARCHBOOST_ASSETS_ROOT/nanochat on a host, with the task image (it has upstream's environment):
#   data/base_data_climbmix/  ClimbMix shards 0-11 and the validation shard 6542 (karpathy/climbmix-400b-shuffle,
#                             upstream's `python -m nanochat.dataset -n 12`)
#   data/tokenizer/           the tokenizer trained on them (upstream's `python -m scripts.tok_train`, defaults:
#                             vocabulary 32768, the first 2B characters of the training shards)
# Usage: prepare_nanochat_assets.sh <image> <assets root>
set -euo pipefail
image=$1
root=$2
mkdir -p "$root/nanochat/data"
docker run --rm --network host --entrypoint bash -e NANOCHAT_BASE_DIR=/out/data -e HF_HUB_OFFLINE=0 \
    -v "$root/nanochat:/out" "$image" -c '
set -euo pipefail
cd /opt/researchboost/ws/targets/reference
/opt/nanochat-env/bin/python -m nanochat.dataset -n 12 -w 8
test "$(ls /out/data/base_data_climbmix/*.parquet | wc -l)" = 13
if [ ! -f /out/data/tokenizer/tokenizer.pkl ]; then
    /opt/nanochat-env/bin/python -m scripts.tok_train
fi
ls -la /out/data/base_data_climbmix /out/data/tokenizer
chmod -R a+rX /out'
