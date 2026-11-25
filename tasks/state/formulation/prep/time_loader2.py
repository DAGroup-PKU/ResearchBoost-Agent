"""Steady-state loader timing: skip warmup, report per-batch mean over 20 batches."""
import time
from cell_load.utils.modules import get_datamodule
kw = dict(toml_config_path="/data/fy/state-rba/bench/tahoe_slice.toml", embed_key="X_hvg",
          pert_col="perturbation", cell_type_key="cell_type", control_pert="DMSO_TF",
          batch_col="gem_group", output_space="all")
dm = get_datamodule("PerturbationDataModule", kw, batch_size=8, cell_sentence_len=512)
dm.setup(stage="fit")
dl = dm.train_dataloader()
it = iter(dl); times = []
for i, batch in enumerate(it):
    t = time.perf_counter()
    b = next(it)
    times.append(time.perf_counter() - t)
    if len(times) >= 20: break
import statistics
print(f"RESULT loader_mean_s={statistics.fmean(times):.4f} median={statistics.median(times):.4f} n={len(times)}", flush=True)
