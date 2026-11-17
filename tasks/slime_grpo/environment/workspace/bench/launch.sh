#!/bin/bash
# The judge's only way to run slime: one real train.py process (Ray, sglang engines and/or Megatron actors), started
# by bench.py with the environment it manages (fresh TMPDIR and compile caches, PATH, the GPUs, the candidate's
# switches). Recipe = Qwen3.5-9B GRPO on dapo-math-17k (upstream examples/fully_async/run-qwen3.5-9B-fully_async.sh,
# made synchronous and colocated on 4 GPUs: actor TP2 x CP2 = DP1, 2 sglang engines x TP2; 16 prompts x 4 samples,
# global batch 64, one optimizer step per rollout; 8K response limit; no checkpoint save).
#
# Required env: SLIME SGLANG MEGATRON (trees), MODE (rollout|train|full), OUT, NUM_ROLLOUT, SLIME_GRPO_BENCH_DIR
#   (the judge's records of this run)
# Optional: FORCE_LENGTHS (json sample index -> response length; rollout and full modes), LOAD_DUMPS (dir with
#   <id>.pt; train mode), START_ROLLOUT, LR, PROFILE_DIR, SLIME_GRPO_BENCH_DUMP=1 (rollout mode: the judge's dump of
#   every rollout, outside the timed window), SLIME_GRPO_BENCH_RECORD=1 (train mode: parameter records).
set -euo pipefail
: "${SLIME:?}" "${SGLANG:?}" "${MEGATRON:?}" "${MODE:?}" "${OUT:?}" "${NUM_ROLLOUT:?}" "${ASSETS:?}"
: "${SLIME_GRPO_BENCH_DIR:?}"
BENCH=$(cd "$(dirname "$0")" && pwd); WS=$(dirname "$BENCH")
mkdir -p "$OUT" "$SLIME_GRPO_BENCH_DIR"
export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=$([ "$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)" -gt 0 ] && echo 1 || echo 0)
export PYTHONPATH="$MEGATRON:$SGLANG/python:$SLIME:$WS"
# opt-in: measure Triton autotuning/compilation in every process of the run, the sglang engines included
# (bench/first_use_site/sitecustomize.py; the judge's Megatron hook cannot reach them)
if [ "${SLIME_GRPO_BENCH_FIRST_USE_ALL:-}" = 1 ]; then export PYTHONPATH="$BENCH/first_use_site:$PYTHONPATH"; fi

cd "$SLIME"
source scripts/models/qwen3.5-9B.sh
CKPT=(--hf-checkpoint "$ASSETS/Qwen3.5-9B" --ref-load "$ASSETS/Qwen3.5-9B_torch_dist")
ROLLOUT=(--prompt-data "$ASSETS/data/dapo-math-17k.jsonl" --input-key prompt --label-key label
  --apply-chat-template --rollout-shuffle --rm-type deepscaler
  --num-rollout "$NUM_ROLLOUT" --start-rollout-id "${START_ROLLOUT:-0}" --rollout-batch-size 16 --n-samples-per-prompt 4
  --rollout-max-response-len 8192 --rollout-temperature 1
  --global-batch-size 64 --balance-data --rollout-seed 1234 --seed 1234)
PERF=(--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1
  --context-parallel-size 2 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --log-probs-chunk-size 1024 --max-tokens-per-gpu 4096)
GRPO=(--advantage-estimator grpo --use-score-centering --score-centering-top-k 128
  --pg-loss-type reinforce --use-tis --tis-clip-low 0 --tis-clip 2
  --disable-grpo-std-normalization --calculate-per-token-loss
  --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl --entropy-coef 0.00)
OPT=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
  --optimizer adam --lr "${LR:-1e-6}" --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98)
SGL=(--rollout-num-gpus-per-engine 2 --sglang-mem-fraction-static 0.7)
MISC=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash)
# The judge's instruments: per-step clocks, the parallel layout that actually ran, the rollout dump (rollout mode)
# and the parameter records (train mode), all written to SLIME_GRPO_BENCH_DIR.
JUDGE=(--custom-rollout-log-function-path bench.rollout_hook.log_rollout
  --custom-megatron-before-train-step-hook-path bench.train_hook.before_train_step)
EXTRA=()
case $MODE in
  rollout) EXTRA+=(--debug-rollout-only)
           [ -n "${FORCE_LENGTHS:-}" ] && EXTRA+=(--custom-generate-function-path bench.force_len.generate) ;;
  train)   EXTRA+=(--debug-train-only --load-debug-rollout-data "${LOAD_DUMPS:?}/{rollout_id}.pt") ;;
  full)    [ -n "${FORCE_LENGTHS:-}" ] && EXTRA+=(--custom-generate-function-path bench.force_len.generate) ;;
  *) echo "bad MODE $MODE"; exit 2 ;;
esac
if [ -n "${PROFILE_DIR:-}" ]; then
  export SLIME_GRPO_BENCH_PROFILE_DIR=$PROFILE_DIR
  # train: step PROFILE_STEP is traced (default 2; the judge's profile traces a replayed step, as the timed ones)
  [ "$MODE" = train ] && EXTRA+=(--use-pytorch-profiler --profile-step-start "${PROFILE_STEP:-2}"
                                 --profile-step-end "$(( ${PROFILE_STEP:-2} + 1 ))" --tensorboard-dir "$PROFILE_DIR")
fi
NGPU=$(nvidia-smi -L | wc -l)
set +e
python3 "$SLIME/train.py" --actor-num-nodes 1 --actor-num-gpus-per-node "$NGPU" --colocate \
  "${MODEL_ARGS[@]}" "${CKPT[@]}" "${ROLLOUT[@]}" "${OPT[@]}" "${GRPO[@]}" "${PERF[@]}" "${SGL[@]}" "${MISC[@]}" \
  "${JUDGE[@]}" "${EXTRA[@]}" > "$OUT/train.log" 2>&1
RC=$?
echo "$RC" > "$OUT/exit_code"
exit $RC
