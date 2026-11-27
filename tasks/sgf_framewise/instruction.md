# Self Gradient Forcing: framewise training

Accelerate the implementation of the frozen SGF framewise training recipe in
`/submission/sgf_framewise` on four H200 GPUs. The unit of work is a complete
optimizer update at the upstream training-loop boundary, preserving the
upstream cadence of generator and critic updates.

Keep the training algorithm, model and data fixed. Preserve the starting
weights, sample identities and order on each rank, noise and sampled exits,
all rollout blocks, context and gradient semantics, denoising schedule, cache
semantics, loss, optimizer and update schedule. Do not substitute a different
training recipe, skip work by detecting evaluation, or use precomputed results.
Preserve the reference precision for each computation and for parameters and
optimizer state. Keep the same per-device batch and effective global batch.

Only tier N1 is awarded: exact workload invariants must match, and numerical
results must pass statistical equivalence against the reference on training
trajectories. Bitwise identity does not waive the trajectory check. Final
evaluation uses the submitted tree with no network access.
