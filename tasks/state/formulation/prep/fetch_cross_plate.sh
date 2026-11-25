#!/bin/bash
# Fetch a plate-diverse slice of Tahoe-100M: shards at a fixed stride across the 3388 shards,
# so the slice spans many plates instead of one.
set -u
N=${1:-200}
OUT=${2:-/data/fy/state-rba/data/parquet_all}
BASE=https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/data
TOTAL=3388
mkdir -p "$OUT"
fetch() {
  i=$1; f=$(printf "train-%05d-of-03388.parquet" "$i")
  [ -s "$OUT/$f" ] && return 0
  curl -sS -L --retry 2 --max-time 1200 -o "$OUT/$f.part" "$BASE/$f" && mv "$OUT/$f.part" "$OUT/$f"
}
export -f fetch; export OUT BASE
# stride so the N shards spread evenly over the whole file order (= plate order)
stride=$(( TOTAL / N ))
seq 0 "$stride" $((stride * N - 1)) | head -n "$N" | xargs -P 10 -I{} bash -c 'fetch {}'
echo "done: $(ls "$OUT"/*.parquet 2>/dev/null | wc -l) shards, $(du -sh "$OUT" | cut -f1)"
