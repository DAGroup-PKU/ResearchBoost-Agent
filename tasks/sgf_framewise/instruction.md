# Self Gradient Forcing: framewise training

## Setup

Work in `/submission/sgf_framewise`, the pinned upstream checkout. The job
runs on four H200 GPUs, with one sample per rank and a global batch of four.
Submit the modified checkout, `/submission/switches.json`, and
`/opt/candidate-env`. Only record permitted settings that are not already on
in the reference; leave the manifest empty if none apply.

## Working time

Your working time is limited. When it ends, the judge evaluates the artifacts
as they stand; there is no further editing during evaluation.

## Workload

Accelerate the implementation of the frozen Self Gradient Forcing framewise
training recipe. The unit of work is one completed upstream trainer-loop
iteration: the critic updates on every iteration, and the generator's
distribution-matching update precedes the critic update when scheduled.
The two updates consume their own batches on these iterations. Keep this
schedule and the real update boundary, not a count of individual optimizer calls.

Keep the dual-pass training semantics. The first pass performs the complete
ordered rollout without gradients, preserving the sampled exits and the
context passed between blocks. The second pass uses the matching context and
preserves the prescribed gradient flow. Preserve every first-pass block and
its denoising work even when a sampled exit selects an earlier state. Keep
the starting weights, per-rank sample order, noise, and sampling semantics.
The generator has a rolling context of six latent frames including the
current frame, with the first three latent frames retained when available.

## Judge

The judge checks exact workload invariants, measured performance benefit,
and statistical equivalence on training trajectories with both critic-only
and generator+critic updates. Only N1 is accepted; bitwise agreement does
not bypass the trajectory check. Final evaluation is offline.

## Rules

- Gains must come from implementation changes that preserve semantics.
  Enabling a capability that the reference already uses is not a submission.
- Keep the model, data, sharding strategy, rank count, batch sizes, latent
  shape, frame grouping, denoising and context semantics, loss, optimizer, and
  update cadence fixed. Do not substitute another forcing recipe or use a
  publicly available optimized implementation or fork.
- Do not lower computation, parameter, reduction, or optimizer-state
  precision below the corresponding reference precision.
- Do not precompute results, use shortcuts keyed to particular inputs, or
  detect a timing or correctness run to change what the code does. Keep the
  judge's assets, configuration, and measurement hooks intact and read-only.
