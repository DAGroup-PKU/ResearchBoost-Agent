# AI for Science

[← All results](README.md) · [ResearchBoost-Agent](../../README.md#results)

3 projects · [ProteinMPNN](#faster-proteinmpnn) · [Protenix](#faster-protenix) · [FermiNet](#faster-ferminet)

## Faster-ProteinMPNN

Contributed by [@micky-li-hd](https://github.com/micky-li-hd).

**ProteinMPNN v_48_020 training · full public dataset**

| Item | Result |
|---|---|
| Speedup | **2.25×** |
| Hardware | 1 × A100 SXM4 40GB |
| Measurement scope | 450.65 → 200.52 s/epoch, mean of epochs 2–6 |
| Validation | Validation perplexity 5.884 vs 5.934 / 5.897 after six epochs |

Six of the recipe's 150 epochs (8,970 updates) per arm, run reference → candidate → reference with cold compiler caches. Whole-process time, including compilation, data loading, validation and checkpoint writing, went from 3,316.2 to 2,154.9 s (1.54×). Final validation accuracy is 0.447 for the candidate and 0.445 / 0.446 for the references; all 472 checkpoint tensors are finite in every run.

This is a bounded exploratory comparison, not a frozen-judge four-gate PASS or a full-training convergence claim. The source tree contains the pinned upstream implementation with the training optimization applied in place, an unchanged reference arm, and reproduction scripts.

[Code & project README ↗](https://github.com/micky-li-hd/Faster-ProteinMPNN) · [Detailed evidence ↗](https://github.com/micky-li-hd/Faster-ProteinMPNN/blob/main/six_epoch_evidence.json)

## Faster-Protenix

Contributed by [@micky-li-hd](https://github.com/micky-li-hd).

**Protenix 2.0.0 training · 128 optimizer updates · crop 384 · diffusion batch 48**

| Item | Result |
|---|---|
| Speedup | **1.55×** |
| Hardware | 8 × NVIDIA H20 96 GB, single host |
| Measurement scope | 1,503.99 → 968.83 s whole process; mean of two references vs candidate |
| Validation | Manual review PASS: 216/216 training comparisons and 135/138 normal/EMA evaluation comparisons passed; the three accepted EMA deviations were smaller than the reference-repeat deviation |

The fixed workload ran reference → candidate → reference with the same inputs, random-state fingerprints, precision, Adam configuration, effective batch, learning-rate schedule, checkpointing and evaluation. Whole-process timing includes startup, cold native compilation, 128 updates, checkpoint writing and evaluation. Mean max-rank step time over updates 16–127 improved from 10.24 to 6.07 s (**1.69×**).

The candidate keeps selected Pairformer, MSA and diffusion activations instead of recomputing them, caches repeated conditioning and sampling projections, fuses attention softmax and the head-gate backward, uses native CUDA kernels for selected layer-norm and affine work, and parallelizes cold builds and model initialization. Keeping activations costs memory: the candidate peaked at 81.9 GiB allocated and 93.4 GiB reserved per rank.

Comparisons used the frozen tolerance max(1e-4, 5% of the reference value). Manual review accepted the three deviations in `ema0.999_lddt/complex/plddt.rank1.avg` as run-to-run variance: the reference-repeat delta was 0.003168, larger than the 0.001859 and 0.001309 candidate deltas. All six final normal/EMA checkpoints were finite. The public repository includes the implementation snapshot, source and publication manifests, reviewed findings, fixed-scope record and clean-run closure record.

The validation covers this fixed 128-update engineering workload; full-corpus convergence is outside its scope. The exact training fixture and generated checkpoints are not redistributed, and the source snapshot is hash-manifested because the upstream Git commit was not recorded.

[Code & project README ↗](https://github.com/micky-li-hd/Faster-Protenix) · [Summary ↗](https://github.com/micky-li-hd/Faster-Protenix/blob/main/results/summary.json) · [Detailed evidence ↗](https://github.com/micky-li-hd/Faster-Protenix/blob/main/results/FINDINGS.json)

## Faster-FermiNet

Contributed by [@sen-ye](https://github.com/sen-ye).

**FermiNet all-electron training on Fe, Kr, Cd and Xe · KFAC · 1024 walkers per GPU**

| Item | Result |
|---|---|
| Speedup | **1.25×** (geometric mean 1.252× across four atoms) |
| Hardware | 1 × NVIDIA H20 96 GB per atom; four independent single-GPU runs |
| Measurement scope | Steady-state training-step time; compilation and warm-up excluded |
| Validation | Frozen timing and bitwise gates PASS for all four atoms; every recorded step matches in 8-step and 110-step deterministic-mode checks |

| Atom | Baseline (s/step) | Optimized (s/step) | Speedup |
|---|---:|---:|---:|
| Fe | 0.5383 | 0.4319 | **1.2463×** |
| Kr | 1.1901 | 0.9569 | **1.2438×** |
| Cd | 2.3870 | 1.8900 | **1.2630×** |
| Xe | 3.7101 | 2.9564 | **1.2549×** |

Based on upstream FermiNet at `e9d5a604`, with JAX 0.7.2 and cuBLAS 12.9.1.4. Eight ResearchBoost-Agent rounds (16 attempts) produced a bit-exact Pallas triangular-solve kernel that preserves cuBLAS layout and a 2× unroll of the Laplacian loop. All atoms use one generic configuration, with upstream precision, hyperparameters and random streams unchanged.

Timing uses default XLA flags and independent baseline A → candidate → baseline B processes on the same GPU. Each arm runs 18 iterations; compilation and two warm-up steps are excluded, and the median of three 5-step windows determines step time. Bitwise checks run separately with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, using the same optimization switches. Energy bits, acceptance probabilities, walkers, parameters, KFAC optimizer state and MCMC width match at every recorded step, including width adaptation at step 100. The kernel also matches real cuBLAS for sizes 9–64, four variants and three input scales.

The measurements cover single-device, real ground-state training with the default Laplacian on the pinned H20 stack. Pretraining and burn-in are shortened for benchmarking; the timed step is unchanged. Full training to convergence and deterministic-mode timing were not run. The portable reproduction launchers package the recorded protocol but have not themselves been rerun end to end. Revalidate the kernel on a different GPU or software stack.

[Code & project README ↗](https://github.com/sen-ye/faster-ferminet) · [Timing & validation evidence ↗](https://github.com/sen-ye/faster-ferminet/blob/master/docs/validation/h20.json) · [Bitwise records ↗](https://github.com/sen-ye/faster-ferminet/blob/master/docs/validation/bitwise_records.json) · [Frozen judge ↗](https://github.com/sen-ye/faster-ferminet/blob/master/task/formulation.json) · [Optimization history ↗](https://github.com/sen-ye/faster-ferminet/blob/master/docs/history.md) · [Reproduction ↗](https://github.com/sen-ye/faster-ferminet/blob/master/docs/reproduction.md)
