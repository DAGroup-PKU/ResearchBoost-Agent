import anndata as ad, numpy as np
a = ad.read_h5ad('/data/fy/state-rba/data/h5ad_v2/tahoe_slice.h5ad')
nz = np.asarray((a.X > 0).sum(axis=0)).ravel()
keep = np.flatnonzero(nz > 0)
print(f'keeping {len(keep)} of {a.n_vars} genes', flush=True)
a = a[:, keep].copy()
a.write_h5ad('/data/fy/state-rba/data/h5ad_v2/tahoe_slice_nz.h5ad', compression='gzip')
print('wrote', a.shape, flush=True)
