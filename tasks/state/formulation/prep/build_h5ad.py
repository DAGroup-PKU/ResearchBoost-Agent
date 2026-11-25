"""Build one AnnData from Tahoe-100M parquet shards (raw counts), for a reproducible benchmark slice.

usage: build_h5ad.py --shards DIR --out X.h5ad --max-cells N [--seed 0]
"""
import argparse, glob, os, random
import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import anndata as ad
import pandas as pd

ap_ = argparse.ArgumentParser()
ap_.add_argument("--shards", required=True)
ap_.add_argument("--out", required=True)
ap_.add_argument("--gene-meta", default="/data/fy/state-rba/data/gene_metadata.parquet")
ap_.add_argument("--max-cells", type=int, default=2_000_000)
ap_.add_argument("--seed", type=int, default=0)
a = ap_.parse_args()

files = sorted(glob.glob(os.path.join(a.shards, "*.parquet")))
if not files:
    raise SystemExit(f"no parquet in {a.shards}")
print(f"{len(files)} shards", flush=True)

# gene vocabulary from token_id
gm = pq.read_table(a.gene_meta, columns=["gene_symbol", "token_id"]).to_pandas()
n_genes = int(gm.token_id.max()) + 1
names = np.array([f"gene_{i}" for i in range(n_genes)], dtype=object)
names[gm.token_id.values] = gm.gene_symbol.values.astype(object)
print(f"n_genes={n_genes}", flush=True)

rng = random.Random(a.seed)
blocks, obss, total = [], [], 0
# Take a quota from EVERY shard instead of filling up from the first few: Tahoe's shards are ordered by
# plate, so a greedy prefix is one plate even when the shard list itself spans many.
quota = max(1, a.max_cells // len(files))
for path in files:
    if total >= a.max_cells:
        break
    t = pq.read_table(path, columns=["genes", "expressions", "drug", "cell_line_id", "plate", "sample"])
    n = t.num_rows
    take = min(n, quota, a.max_cells - total)
    idx = list(range(n)) if take >= n else sorted(rng.sample(range(n), take))
    if take < n:
        t = t.take(idx)
    g = t.column("genes").to_pylist()
    e = t.column("expressions").to_pylist()
    indptr = np.zeros(len(g) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(x) for x in g])
    indices = np.fromiter((v for row in g for v in row), dtype=np.int32, count=int(indptr[-1]))
    data = np.fromiter((v for row in e for v in row), dtype=np.float32, count=int(indptr[-1]))
    blocks.append(sp.csr_matrix((data, indices, indptr), shape=(len(g), n_genes)))
    obss.append(pd.DataFrame({
        "perturbation": t.column("drug").to_pandas().values,
        "cell_type": t.column("cell_line_id").to_pandas().values,
        "batch": t.column("plate").to_pandas().values,
        "sample": t.column("sample").to_pandas().values,
    }))
    total += len(g)
    if len(blocks) % 20 == 0:
        print(f"  {len(blocks)} shards, {total} cells", flush=True)

X = sp.vstack(blocks, format="csr")
obs = pd.concat(obss, ignore_index=True)
obs.index = [f"cell{i}" for i in range(X.shape[0])]
var = pd.DataFrame(index=pd.Index(names, name=None))
adata = ad.AnnData(X=X, obs=obs, var=var)
print(f"built {adata.shape}, nnz={X.nnz}, density={X.nnz/np.prod(X.shape):.4f}", flush=True)
print(adata.obs.nunique().to_dict(), flush=True)
adata.write_h5ad(a.out, compression="gzip")
print("wrote", a.out, f"{os.path.getsize(a.out)/1e9:.1f} GB", flush=True)
