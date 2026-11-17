#!/bin/bash
# The judge's only way to run slime: the teacher's sglang server (rollout, full and teacher modes) and one real
# train.py process (Ray, the student's sglang engines and/or Megatron actors), started by bench.py with the environment
# it manages (fresh TMPDIR and compile caches, PATH, the GPUs, the candidate's switches).
#
# Recipe = on-policy distillation, upstream examples/on_policy_distillation/run-qwen3-4B-opd.sh (sglang teacher mode)
# with the models replaced by the task owner: student Qwen3.5-9B (trained), teacher Qwen3.5-27B (scores the student's
# samples). Colocated on 4 GPUs: student actor TP2 x CP2 = DP1, 2 student sglang engines x TP2, and the teacher's
# stock-configured sglang server TP4 resident on the same GPUs. 16 prompts x 4 samples, global batch 64, one optimizer
# step per rollout; 8K response limit (upstream 128K); no checkpoint save.
#
# Required env: SLIME SGLANG MEGATRON (trees), MODE (rollout|train|full|teacher), OUT, NUM_ROLLOUT, SLIME_OPD_BENCH_DIR
#   (the judge's records of this run)
# Optional: FORCE_LENGTHS (json sample index -> response length; rollout and full modes), LOAD_DUMPS (dir with
#   <id>.pt; train and teacher modes), START_ROLLOUT, LR, PROFILE_DIR, SLIME_OPD_BENCH_DUMP=1 (rollout mode: the
#   judge's dump of every rollout, outside the timed window), SLIME_OPD_BENCH_RECORD=1 (train mode: parameter
#   records), SLIME_OPD_BENCH_REPLAY_CONCURRENCY (teacher mode, default: every sample of a rollout at once).
# Candidate switch read here: OPD_TEACHER_SERVER_ARGS (extra arguments of the teacher's sglang.launch_server, appended
#   after the recipe's; the layout the teacher reports is judged).
set -euo pipefail
: "${SLIME:?}" "${SGLANG:?}" "${MEGATRON:?}" "${MODE:?}" "${OUT:?}" "${NUM_ROLLOUT:?}" "${ASSETS:?}"
: "${SLIME_OPD_BENCH_DIR:?}"
BENCH=$(cd "$(dirname "$0")" && pwd); WS=$(dirname "$BENCH")
mkdir -p "$OUT" "$SLIME_OPD_BENCH_DIR"
export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=$([ "$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)" -gt 0 ] && echo 1 || echo 0)
export PYTHONPATH="$MEGATRON:$SGLANG/python:$SLIME:$WS"
# opt-in: measure Triton autotuning/compilation in every process of the run, the sglang engines included
# (bench/first_use_site/sitecustomize.py; the judge's Megatron hook cannot reach them)
if [ "${SLIME_OPD_BENCH_FIRST_USE_ALL:-}" = 1 ]; then export PYTHONPATH="$BENCH/first_use_site:$PYTHONPATH"; fi
NGPU=$(nvidia-smi -L | wc -l)

# ---- the teacher: upstream's server command (chunked prefill 4096), TP over the 4 GPUs, resident next to the
# colocated student (its static memory fraction leaves room for the student's engines and Megatron).
TEACHER_PORT=13141
TEACHER_URL=http://127.0.0.1:$TEACHER_PORT
TPID=
stop_teacher() { [ -n "$TPID" ] && { kill -TERM -- "-$TPID" 2>/dev/null || kill -TERM "$TPID" 2>/dev/null; sleep 5;
                 kill -KILL -- "-$TPID" 2>/dev/null || true; }; }
trap stop_teacher EXIT
start_teacher() {
  # shellcheck disable=SC2086
  setsid python3 -m sglang.launch_server --model-path "$ASSETS/Qwen3.5-27B" --host 127.0.0.1 --port "$TEACHER_PORT" \
    --tp "$NGPU" --chunked-prefill-size 4096 --mem-fraction-static 0.25 \
    ${OPD_TEACHER_SERVER_ARGS:-} > "$OUT/teacher.log" 2>&1 &
  TPID=$!
  local t0=$SECONDS
  until curl -sf "$TEACHER_URL/health_generate" > /dev/null; do
    kill -0 "$TPID" 2>/dev/null || { echo "teacher server died; see $OUT/teacher.log" >&2; tail -40 "$OUT/teacher.log" >&2; exit 3; }
    [ $((SECONDS - t0)) -gt 1800 ] && { echo "teacher server not ready after 1800 s" >&2; exit 3; }
    sleep 5
  done
  # the layout the teacher actually runs (judged by bench.py against the reference teacher's)
  { curl -sf "$TEACHER_URL/get_server_info" || curl -sf "$TEACHER_URL/server_info" || echo '{"error": "no server info"}'; } \
    > "$SLIME_OPD_BENCH_DIR/teacher_layout.json"
  echo "teacher ready after $((SECONDS - t0)) s" >> "$OUT/teacher.log"
}

if [ "$MODE" = teacher ]; then
  # Teacher alone: replay the token sequences of frozen rollout dumps through the tree's reward path (the custom
  # reward function and its post-processing, as slime calls them), timed per rollout by the judge.
  start_teacher
  REPLAY=(--url "$TEACHER_URL/generate" --dumps "${LOAD_DUMPS:?}" --start "${START_ROLLOUT:-0}" --num "$NUM_ROLLOUT"
          --out "$SLIME_OPD_BENCH_DIR" --custom-rm-path slime.rollout.on_policy_distillation.reward_func
          --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
          --temperature 1 --concurrency "${SLIME_OPD_BENCH_REPLAY_CONCURRENCY:-0}")
  [ -n "${PROFILE_DIR:-}" ] && REPLAY+=(--profile-dir "$PROFILE_DIR")
  set +e
  python3 -m bench.teacher_replay "${REPLAY[@]}" > "$OUT/train.log" 2>&1
  RC=$?
  echo "$RC" > "$OUT/exit_code"
  exit $RC
fi

cd "$SLIME"
source scripts/models/qwen3.5-9B.sh
CKPT=(--hf-checkpoint "$ASSETS/Qwen3.5-9B" --ref-load "$ASSETS/Qwen3.5-9B_torch_dist")
ROLLOUT=(--prompt-data "$ASSETS/data/dapo-math-17k.jsonl" --input-key prompt --apply-chat-template --rollout-shuffle
  --num-rollout "$NUM_ROLLOUT" --start-rollout-id "${START_ROLLOUT:-0}" --rollout-batch-size 16 --n-samples-per-prompt 4
  --rollout-max-response-len 8192 --rollout-temperature 1
  --global-batch-size 64 --balance-data --rollout-seed 1234 --seed 1234)
RM=(--custom-rm-path slime.rollout.on_policy_distillation.reward_func
  --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
  --rm-url "$TEACHER_URL/generate")
PERF=(--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1
  --context-parallel-size 2 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --log-probs-chunk-size 1024 --max-tokens-per-gpu 4096)
OPD=(--advantage-estimator grpo --use-opd --opd-type sglang --opd-kl-coef 1.0
  --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl --entropy-coef 0.00)
OPT=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
  --optimizer adam --lr "${LR:-1e-6}" --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98)
# The router policy is fixed to round robin (upstream: slime's default cache_aware), a deviation for reproducible
# timing: cache_aware sends the 64 requests of a rollout 28/36 or 32/32 to the two engines depending on arrival timing,
# and an uneven rollout lasts ~5 s (7%) longer; round robin splits them 32/32 every time.
SGL=(--rollout-num-gpus-per-engine 2 --sglang-mem-fraction-static 0.45 --router-policy round_robin)
MISC=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash)
# The judge's instruments: per-step clocks, the parallel layout that actually ran, the rollout dump (rollout mode),
# the per-sample generation end (forced-length modes) and the parameter records (train mode), all written to
# SLIME_OPD_BENCH_DIR.
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
  export SLIME_OPD_BENCH_PROFILE_DIR=$PROFILE_DIR
  # train: step PROFILE_STEP is traced (default 2; the judge's profile traces a replayed step, as the timed ones)
  [ "$MODE" = train ] && EXTRA+=(--use-pytorch-profiler --profile-step-start "${PROFILE_STEP:-2}"
                                 --profile-step-end "$(( ${PROFILE_STEP:-2} + 1 ))" --tensorboard-dir "$PROFILE_DIR")
fi
# The teacher scores rollouts: up before the student's engines (rollout and full modes). Train mode needs no
# teacher: the frozen dumps carry the teacher's responses.
[ "$MODE" != train ] && start_teacher
set +e
python3 "$SLIME/train.py" --actor-num-nodes 1 --actor-num-gpus-per-node "$NGPU" --colocate \
  "${MODEL_ARGS[@]}" "${CKPT[@]}" "${ROLLOUT[@]}" "${RM[@]}" "${OPT[@]}" "${OPD[@]}" "${PERF[@]}" "${SGL[@]}" \
  "${MISC[@]}" "${JUDGE[@]}" "${EXTRA[@]}" > "$OUT/train.log" 2>&1
RC=$?
echo "$RC" > "$OUT/exit_code"
exit $RC
