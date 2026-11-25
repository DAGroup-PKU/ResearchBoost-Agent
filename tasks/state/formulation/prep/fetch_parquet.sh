#!/bin/bash
# Fetch N shards of Tahoe-100M parquet with limited parallelism.
set -u
N=${1:-120}
OUT=${2:-/data/fy/state-rba/data/parquet}
BASE=https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/data
mkdir -p "$OUT"
fetch() {
  i=$1; f=$(printf "train-%05d-of-03388.parquet" "$i")
  [ -s "$OUT/$f" ] && return 0
  for try in 1 2 3; do
    curl -sS -L --retry 2 --max-time 900 -o "$OUT/$f.part" "$BASE/$f" && mv "$OUT/$f.part" "$OUT/$f" && return 0
  done
  echo "FAILED $f"; return 1
}
export -f fetch; export OUT BASE
seq 0 $((N-1)) | xargs -P 8 -I{} bash -c 'fetch {}'
echo "done: $(ls "$OUT"/*.parquet 2>/dev/null | wc -l) shards, $(du -sh "$OUT" | cut -f1)"
