"""Time the cell_load dataloader alone (no GPU): per-batch time and MB/s."""
import time, sys
import torch
from cell_load.utils.modules import get_datamodule

kwargs = dict(
    toml_config_path="/data/fy/state-rba/bench/tahoe_slice.toml",
    embed_key="X_hvg", pert_col="perturbation", cell_type_key="cell_type",
    control_pert="DMSO_TF", batch_col="gem_group", output_space="all",
)
dm = get_datamodule("PerturbationDataModule", kwargs, batch_size=8, cell_sentence_len=512)
dm.setup(stage="fit")
dl = dm.train_dataloader()
print("workers", dl.num_workers, "batch_size", dl.batch_size, flush=True)
it = iter(dl)
n, t0 = 0, time.perf_counter()
for i, batch in enumerate(it):
    if i < 2:
        continue
    t = time.perf_counter()
    if i == 12:
        break
    shp = {k: (v.shape if hasattr(v, "shape") else v) for k, v in list(batch.items())[:4]}
print(f"batches 10, elapsed {time.perf_counter()-t0:.2f}s", flush=True)
print("sample keys/shapes:", shp, flush=True)
