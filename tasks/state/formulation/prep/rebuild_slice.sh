#!/bin/bash
# Rebuild the benchmark slice end to end for the cross-plate Tahoe shards, then re-freeze everything the judge
# compares against. Every step is idempotent; stop at the first failure.
set -euo pipefail
R=/data/fy/state-rba
WS=$R/ws
export ASSETS=$R/assets DATASET_DIR=$R/assets/data/dataset CUDA_VISIBLE_DEVICES=4
export PATH=$R/dataenv/bin:$R/venv/bin:$PATH

say() { echo "[rebuild $(date +%T)] $*"; }

until ! pgrep -f "[f]ilter_genes.py" >/dev/null; do sleep 10; done
say "gene filter done"

say "preprocessing (HVG selection)"
cd $WS/state
$R/venv/bin/python -m state tx preprocess_train \
  --adata $R/data/h5ad_v2/tahoe_slice_nz.h5ad \
  --output $R/data/h5ad_v2/tahoe_pre.h5ad --num_hvgs 2000
say "preprocess done"

# AFTER preprocessing, not before: `preprocess_train` round-trips through AnnData and rewrites the file, which
# drops any column added by hand and re-emits `_index` in anndata's nullable-string form.
say "normalising the preprocessed h5ad for cell_load"
$R/dataenv/bin/python $R/prep/normalize_h5ad.py $R/data/h5ad_v2/tahoe_pre.h5ad

# repoint the assets symlink at the new slice; the old draft slice is retired
ln -sfn $R/data/h5ad_v2/tahoe_pre.h5ad $R/assets/data/dataset/tahoe_slice.h5ad
say "assets -> $(readlink -f $R/assets/data/dataset/tahoe_slice.h5ad)"

cd $WS
say "rebuilding the frozen batch fixture"
$R/venv/bin/python bench/build_fixture.py --tree state --toml bench/dataset.toml \
  --out bench/fixture.pt --batches 6
python3 - <<'PY'
import json, hashlib
from pathlib import Path
ws = Path('/data/fy/state-rba/ws')
fx = ws / 'bench' / 'fixture.pt'
(ws / 'bench' / 'fixture_manifest.json').write_text(
    json.dumps({'bench/fixture.pt': hashlib.sha256(fx.read_bytes()).hexdigest()}, indent=1) + '\n')
print('fixture manifest refreshed')
PY

say "re-recording the reference twice"
rm -rf bench/records/ref bench/records/ref2
$R/venv/bin/python bench/bench.py record --out bench/records/ref   > runs/rec_ref.log  2>&1
$R/venv/bin/python bench/bench.py record --out bench/records/ref2  > runs/rec_ref2.log 2>&1

python3 - <<'PY'
import json, sys
from pathlib import Path
a = json.loads(Path('bench/records/ref/train2.r0.json').read_text())
b = json.loads(Path('bench/records/ref2/train2.r0.json').read_text())
n = sum(1 for ea, eb in zip(a['steps'], b['steps'])
        for k in ea['tensors'] if ea['tensors'][k]['sha256'] != eb['tensors'][k]['sha256'])
f = [k for k in ('rng_before', 'rng_after', 'invariants', 'counters', 'math_mode')
     if a['steps'][0].get(k) != b['steps'][0].get(k)]
print(f'REPRODUCIBILITY: differing fields {f or "NONE"}, differing tensors {n}')
if f or n:
    sys.exit('the reference does not reproduce itself; the new slice is not frozen')
PY

say "recalibrating thresholds and re-freezing the reference timing"
$R/venv/bin/python -m numerics calibrate --ref bench/records/ref --variants bench/records/ref --out bench/thresholds.json
rm -rf runs/freeze
$R/venv/bin/python bench/bench.py freeze-timing --out runs/freeze

say "freezing input hashes"
rm -rf bench/records/ref2
$R/venv/bin/python bench/bench.py freeze-inputs \
  $R/assets/data/dataset/tahoe_slice.h5ad $WS/bench/fixture.pt $R/assets/data/gene_metadata.parquet

say "ALL DONE"
python3 -c "import json;d=json.load(open('task/timing_ref.json'));print('frozen timing:',d['s_per_iteration'],'s/iter over',d['iterations_timed'],'iterations')"
