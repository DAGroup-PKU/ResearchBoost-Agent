# 3D Reconstruction

[← All results](README.md) · [ResearchBoost-Agent](../../README.md#results)

2 projects · [Original 3DGS](#original-3dgs) · [PIDG](#pidg)

Contributed by [@HaoqinHong](https://github.com/HaoqinHong). Both experiments use one NVIDIA A800-SXM4-80GB, original precision settings, and a shared GPU with existing viewers running. Speedup is baseline time divided by experimental time. Validation uses task metrics: held-out-view PSNR for Original 3DGS and module-output numerical checks for PIDG. Judge statuses are listed below.

## Original 3DGS

**Mip-NeRF360 · bonsai, bicycle and garden · complete 30,000-step native pipelines**

| Item | Result |
|---|---|
| Speedup | **1.13×** across the three complete pipelines; **1.07–1.29×** per scene |
| Hardware | 1 × NVIDIA A800-SXM4-80GB |
| Measurement scope | 4269.73 → 3773.86 s total, including initialization, training, native evaluation and saving |
| Validation | Held-out-view PSNR: ΔPSNR −0.014 to +0.038 dB per scene |
| Judge status | `INCORRECT / exit 2`: 11 continuous-value violations among 1,287 checks; `formal_ResearchBoost_acceptance: false` |

| Scene | Baseline (s) | Experimental (s) | Speedup | PSNR baseline → experimental (dB) | ΔPSNR (dB) |
|---|---:|---:|---:|---:|---:|
| bonsai | 1050.32 | 816.46 | **1.29×** | 32.1943 → 32.2318 | +0.0375 |
| bicycle | 1587.13 | 1436.74 | **1.10×** | 25.2363 → 25.2222 | −0.0140 |
| garden | 1632.28 | 1520.65 | **1.07×** | 27.3985 → 27.3963 | −0.0023 |

Timing uses the eligible paired seed-1 runs. PSNR averages seeds 0 and 1 over all held-out views at step 30,000: 37 for bonsai, 25 for bicycle and 24 for garden. Official image scales are `images_2` for bonsai and `images_4` for bicycle/garden. PSNR follows the pinned historical implementation's per-channel averaging convention.

### Timing eligibility, noise and Gaussian counts

The eligibility policy was declared before optimization. Both paired runs must have `full_train.json` status `complete` and return code 0, a successful fresh CPU artifact-check subprocess with empty `CUDA_VISIBLE_DEVICES`, and exact frozen source/data/config/seed/schedule bindings. Native duration is the artifact-check subprocess start minus the full-training process start; the extra CPU check is outside that duration.

All three seed-0 baselines are ineligible for timing: bonsai used the legacy checker and lacks the same fresh-CPU boundary; bicycle/garden's original post-training artifact checks exited SIGBUS (−7). Separate strict recovery qualifies the saved models for image-quality evaluation only and preserves the failed original exits. Seed-1 runs provide comparable boundaries for all scenes.

There is **one eligible baseline/candidate timing pair per scene**. Baselines were collected before optimization and candidate full runs followed; there was no complete-pipeline baseline → candidate → baseline repeat. **Run-to-run noise and confidence intervals for complete-pipeline speedup were not estimated.** Noise from the separate fixed-N probes does not estimate full-pipeline noise.

| Scene | Baseline final Gaussians | Experimental final Gaussians | Change |
|---|---:|---:|---:|
| bonsai | 1,275,632 | 1,271,323 | −0.34% |
| bicycle | 6,071,245 | 5,971,655 | −1.64% |
| garden | 5,927,094 | 5,951,459 | +0.41% |

FP32 accumulation changes affect the training trajectory and densification. The complete-pipeline ratios include these Gaussian-count and work differences; they do not isolate equal-work kernel acceleration.

The patch reduces FP32 gradients for full warps with a common Gaussian ID in rasterization backward and retains the original atomic path otherwise. Separate fixed-Gaussian-count probes at checkpoints 100, 7,000 and 29,900 measured **1.04–2.17×** using baseline → candidate → baseline arms; they exclude densification/reset/evaluation/save events.

### Validation

The 11 numerical violations remain unresolved; exact-state, runtime-error, dtype and math-mode failures were zero. Quality comparison recorded 52 endpoints outside the frozen two-seed baseline descriptive band: 28 worsened and 24 improved. Two quality seeds do not establish statistical equivalence. There is no frozen-judge PASS for this case.

[Code & project README ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost) · [Native timing ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/gs3d/native-timing-results.csv) · [Timing policy, exclusions & noise ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/gs3d/native-timing-notes.json) · [Image quality ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/gs3d/quality-results.csv) · [Numerical checks ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/gs3d/numerics.json)

## PIDG

**Physics-Informed Deformable Gaussian Splatting · dry_ice · optical-flow module-group benchmark**

| Item | Result |
|---|---|
| Speedup | **4.17×**, final R06: retained flow optimization + restored original raster CUDA source |
| Hardware | 1 × NVIDIA A800-SXM4-80GB |
| Measurement scope | 9369.79 → 2244.31 ms for three native renders, the original velocity network, optical-flow/reconstruction losses, backward and five PNG writes |
| Module numerical status | R06 `checks[4]`: 2,435 checked, 1,368 bitwise-identical, 0 unresolved; generic `PASS` |
| Formal judge status | `formal_ResearchBoost_acceptance: false`; `formal_benefit_verdict: UNAVAILABLE_SHARED_WORKLOAD` |

### Historical labels and the failing/passing checks

`both` in the harness selects the flow file **and** raster extension from the candidate tree. It describes module selection, not which selected modules contain optimization code. The raw labels and failed checks remain unchanged in the published JSON.

| Summary index | Run / recorded variant | Effective implementation | Generic status | Bitwise identical | Continuous violations |
|---|---|---|---|---:|---:|
| `checks[0]` | R01 / `flow-only` | Dense embedding gather + original raster | `INCORRECT` | 1,168 | 82 |
| `checks[1]` | R03 / `flow-only` | Exact-zero-eliding gather + original raster | `PASS` | 1,360 | 0 |
| `checks[2]` | R04 / `both` | R03 gather + grouped raster reduction | `INCORRECT` | 1,367 | 84 |
| `checks[3]` | R04 / `raster-only` | Original flow + grouped raster reduction | `INCORRECT` | 1,372 | 84 |
| `checks[4]` | R06 / `both` | R03 gather + original raster source rebuilt in candidate tree | `PASS` | 1,368 | 0 |

Each row has 2,435 checks. R01's embedding gather was rejected; R03 replaced it with the exact-zero-eliding gather implementation. R04's raster-only and accelerated combined implementations were rejected. Before R06, the rejected raster CUDA source was restored byte-for-byte to baseline and its candidate-local extension rebuilt; the R03 flow source stayed unchanged. Thus **1,360** belongs to the R03 flow-only PASS, while the **4.17×** confirmation and **1,368** bitwise checks belong to R06, still recorded as `both`.

The R06 flow source hash is `3566f99a…26a37b`; its restored raster backward is `ea9d57d6…81ac6b`, versus rejected R04 `9edf9e28…273f1a`. The [run-provenance record](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/pidg/run-provenance.json) includes full source/binary hashes and archived run paths. All five checks retain the same thresholds-file SHA256, `ea0ad940…8d8232`; no threshold relaxation was used between these candidates. R02 timing was configuration-invalid; the common GLM include-path build fix precedes subsequent valid measurements.

R06's longer dry_ice confirmation uses baseline → candidate → baseline independent processes, one warmup and three windows of three iterations per arm. Active clocks were comparable and its observed ratio exceeded the 3.27% diagnostic resolution threshold. That threshold and the generic module PASS do not constitute formal benefit acceptance on the shared GPU.

The gather optimization omits exactly-zero gradient vectors, preserving forward computation, nonzero/nonfinite gradients and higher-order differentiation. Five-scene ablations use the original 100,000-point initialization and training frames `0002/0003`. The headline covers the module group above, excluding the deformation network, physical regularizers, densification and complete training. A 601-step native dry_ice run completed with finite saved artifacts.

[Code & project README ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost) · [Full run-history explanation ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/docs/pidg-run-history.md) · [Original summary & checks ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/pidg/summary.json) · [Five-scene ablations ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/pidg/measurements.csv)

## Reproduction

The public repository includes pinned upstream versions, retained and rejected patches, public-data SHA256 manifests, benchmark drivers and result tables. These runs used ResearchBoost methodology, skill cards and its unmodified generic numerical component; they did not launch `clawo`.

[Reproduction commands ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/docs/reproduce.md) · [Environment ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/docs/environment.md)
