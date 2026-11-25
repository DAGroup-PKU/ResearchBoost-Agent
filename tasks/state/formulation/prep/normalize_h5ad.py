"""Make a preprocessed h5ad readable by cell_load.

Recent anndata writes `obs/_index` and `var/_index` as a `nullable-string-array` GROUP (values + mask).
cell_load does `h5_file["obs/_index"][:]`, which needs a plain string dataset, so it raises
`TypeError: Accessing a group is done with bytes or str, not <class 'slice'>`.

It also reads `obs/<batch_col>` with the benchmark's batch_col = `gem_group`, while the Tahoe slice calls that
column `batch`; the column is copied under the expected name.

Should the upstream preprocess step ever emit these forms itself, this script becomes a no-op.

usage: normalize_h5ad.py <in.h5ad> [--batch-col gem_group] [--source-col batch]
"""
import argparse
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument('path')
ap.add_argument('--batch-col', default='gem_group')
ap.add_argument('--source-col', default='batch')
a = ap.parse_args()

with h5py.File(a.path, 'r+') as f:
    for grp in ('obs', 'var'):
        g = f[grp]['_index']
        if hasattr(g, 'keys'):
            vals = g['values'][:]
            attrs = dict(g.attrs)
            del f[grp]['_index']
            f[grp].create_dataset('_index', data=vals, dtype=h5py.string_dtype('utf-8'))
            print(f'{grp}/_index: nullable group -> plain string array ({len(vals)} values, was {attrs})')

    obs = f['obs']
    if a.batch_col not in obs:
        src = obs[a.source_col]
        dst = obs.create_group(a.batch_col)
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.create_dataset('codes', data=src['codes'][:])
        dst.create_dataset('categories', data=src['categories'][:])
        print(f'obs/{a.batch_col}: copied from obs/{a.source_col}')
    else:
        print(f'obs/{a.batch_col}: already present')
