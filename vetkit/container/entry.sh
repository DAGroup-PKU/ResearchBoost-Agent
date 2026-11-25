#!/bin/bash
# entry.sh <build|trial|judge> [command...]: the entrypoint of every reference-pack container. bin/refpack.py mounts
# this directory at /opt/rbb-svc (read-only) and sets RBB_SVC_TASK and RBB_SVC_PYTHON. It does what the task's own
# entrypoint.sh does, except that the startup preparation is replaced:
#   build   the task's own preparation (/opt/researchboost/bootstrap.py, unchanged), then svcpack.py collect writes the
#           pack to $RBB_SVC_OUT/pack (a pack-build run; no agent, no judge)
#   trial   pack mode: svcpack.py verify restores /opt/rbb-svc/pack and only verifies (bitwise record, one timing in
#           the band); then the ready flag and exec "$@" (e.g. `true` for a verify-only run, or an agent's command)
#   judge   pack mode as above, then the submission (/logs/artifacts, read-only, Harbor's artifacts/ layout) is put in
#           place AS THE AGENT USER (restore_bundle.py), then the task's unchanged final judge /tests/test.sh
#           (verify.py) writes /logs/verifier; exit status = test.sh's
# Exit 81: the pack does not verify here (an infrastructure stop; never a silent preparation); 82: no pack mounted;
# 84: the submission was refused (restore_bundle.py); other non-zero: a failure, see $RBB_SVC_OUT.
set -uo pipefail
# PATH (protocol supplement C10). An image's PATH often starts with a directory the agent can write (its candidate
# environment), and in the judge role the submission is put back there. So: entry.sh's own commands, which run as root,
# use the system directories only; the task's own steps before anything of the agent's exists (no_credentials.py, the
# preparation, pack mode, a trial's command) get the image's PATH, as under the task's own entrypoint; the final judge
# gets the image's PATH without the directories that are not root-only (safepath.py), and entry.sh's root-run helper
# interpreter must be root-only itself.
IMAGE_PATH=${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
role=${1:?role}; shift
T=${RBB_SVC_TASK:?}
SVC=/opt/rbb-svc
OUT=${RBB_SVC_OUT:-/opt/rbb-svc/out}
# the task's training interpreter (the one its entrypoint.sh runs bootstrap.py with), e.g. /opt/nanochat-env/bin/python
PY=${RBB_SVC_PYTHON:?set RBB_SVC_PYTHON to the training python of the task image}
stamp() { date '+%F %T %Z'; }
phase() { printf '{"phase": "%s", "time": "%s", "epoch": %s}\n' "$1" "$(stamp)" "$(date +%s)" >> "$OUT/phases.jsonl"; }
mkdir -p "$OUT" /var/log/researchboost
chmod 755 "$OUT"
phase start
# as the task's entrypoint.sh: an empty /tmp, no credentials in the container
find /tmp -mindepth 1 -delete
chmod 1777 /tmp
# the system interpreter for the helpers (no_credentials.py, restore_bundle.py, guard.py: standard library only).
# An image built FROM python:*-slim has no /usr/bin/python3, only /usr/local/bin/python3.
SYSPY=/usr/bin/python3
[ -x "$SYSPY" ] || SYSPY=/usr/local/bin/python3
[ -x "$SYSPY" ] || SYSPY=$PY
PATH=$IMAGE_PATH "$SYSPY" -I /opt/researchboost/no_credentials.py || { echo "entry.sh: no_credentials.py failed" >&2; exit 1; }
compat() {   # CUDA forward compatibility for an older host driver, as some task entrypoints set it up
  echo /usr/local/cuda/compat > /etc/ld.so.conf.d/000-cuda-compat.conf; ldconfig
}
newer_compat() {   # the image's forward-compat libcuda is newer than the host's driver (only then is it of any use)
  c=$(ls /usr/local/cuda/compat/ 2>/dev/null | sed -n 's/^libcuda\.so\.\([0-9][0-9]*\.[0-9.]*\)$/\1/p' | sort -V | tail -1)
  [ -n "$c" ] || return 1
  d=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null); d=${d%%$'\n'*}
  [ -n "$d" ] && [ "$c" != "$d" ] && [ "$(printf '%s\n%s\n' "$d" "$c" | sort -V | tail -1)" = "$c" ]
}
# Default never: as the task's own entrypoint, which usually does not set them up (pack mode must run the libraries the
# trials run). auto: only when the image's are newer than the host driver (loading forward-compat libraries OLDER than
# the kernel driver fails with CUDA error 803); always: for images whose own entrypoint always sets them up.
case ${RBB_SVC_COMPAT:-never} in
  always) compat ;;
  never) ;;
  *) if newer_compat; then compat; fi ;;
esac
# extra directories the task's entrypoint makes world-readable after preparation (space separated), besides ws
extra_chmod=${RBB_SVC_EXTRA_CHMOD:-}
log=/var/log/researchboost/prepare.log
finish_prep() {   # what the task's entrypoint does after a successful preparation
  chmod -R a+rX /opt/researchboost/ws $extra_chmod
  # held-out data stays the judge's
  if [ -d /opt/researchboost/ws/heldout ]; then chmod -R go-rwx /opt/researchboost/ws/heldout; fi
  chmod 644 $log
  touch /run/researchboost-ready
}
if [ "$role" = build ]; then
  PATH=$IMAGE_PATH "$PY" $SVC/svcpack.py pristine --task "$T" --out "$OUT/pristine.json" > "$OUT/collect.log" 2>&1 || exit 1
  phase prepare
  if ! PATH=$IMAGE_PATH "$PY" /opt/researchboost/bootstrap.py > $log 2>&1; then
    cp $log "$OUT/prepare.log"; echo "entry.sh: the task's preparation failed (prepare.log)" >&2; tail -n 60 $log >&2
    exit 1
  fi
  finish_prep
  cp $log "$OUT/prepare.log"
  phase collect
  PATH=$IMAGE_PATH "$PY" $SVC/svcpack.py collect --task "$T" --pristine "$OUT/pristine.json" --out "$OUT/pack" >> "$OUT/collect.log" 2>&1
  rc=$?
  phase end
  exit $rc
fi
phase verify
PATH=$IMAGE_PATH "$PY" $SVC/svcpack.py verify --task "$T" --pack $SVC/pack --report "$OUT/svc-pack.json" > $log 2>&1
rc=$?
cp $log "$OUT/prepare.log"; chmod 644 "$OUT/prepare.log"
if [ $rc != 0 ]; then
  echo "entry.sh: pack mode failed with exit $rc ($OUT/svc-pack.json)" >&2; tail -n 30 $log >&2
  exit $rc
fi
finish_prep
phase ready
if [ "$role" = trial ]; then
  export PATH=$IMAGE_PATH
  exec "$@"
fi
[ "$role" = judge ] || { echo "entry.sh: unknown role $role" >&2; exit 2; }
# Before the submission is back: the root-run helper interpreter (guard.py runs under it) must be root-only, and the
# judge's PATH is the image's without the directories the agent could write.
"$SYSPY" -I $SVC/safepath.py check "$SYSPY" || { echo "entry.sh: $SYSPY may not run as root here" >&2; exit 1; }
JUDGE_PATH=$("$SYSPY" -I $SVC/safepath.py path "$IMAGE_PATH" /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
[ -n "$JUDGE_PATH" ] || { echo "entry.sh: no root-only PATH directory" >&2; exit 1; }
BASH_BIN=$(command -v bash)
echo "$JUDGE_PATH" > "$OUT/judge-path.txt"
# the submission, as the agent user (it can only write where the agent could). A private copy of the read-only
# artifacts, owned by the agent, so that files without read permission for others are readable too.
stage=/var/tmp/rbb-bundle
rm -rf $stage && cp -a /logs/artifacts $stage && chown -R -h agent:agent $stage || { echo "entry.sh: staging failed" >&2; exit 1; }
runuser -u agent -- "$SYSPY" -I $SVC/restore_bundle.py --artifacts $stage \
  --allow ${RBB_SVC_ARTIFACTS:?} > "$OUT/restore.json" 2> "$OUT/restore.err"
rc=$?
rm -rf $stage
phase restored
if [ $rc != 0 ]; then
  echo "entry.sh: the submission was not restored (exit $rc): $(head -c 600 "$OUT/restore.json")" >&2
  [ $rc = 3 ] && exit 84
  exit 1
fi
# the judge's output directories belong to root, whoever created them on the host: the candidate (agent user) must not
# be able to plant files or links there that verify.py (root) would then write through
chown root:root /logs/verifier "$OUT" 2>/dev/null && chmod 755 /logs/verifier "$OUT" \
  || echo "entry.sh: WARNING: could not make /logs/verifier and $OUT root-owned" >&2
# the task's final judge, unchanged: Harbor runs `bash /tests/test.sh` as the verifier user (root) in /tests' parent;
# here with JUDGE_PATH (above) in place of the image's PATH.
# Under guard.py (a root child subreaper): every non-root process orphaned in this container, i.e. a candidate process
# that escaped its run, is killed at once, so none is left to rewrite the bench's check.json before verify.py reads it.
# RBB_SVC_GUARD=0 (refpack.py judge --no-guard, adversarial tests only) runs test.sh without it.
cd /
if [ "${RBB_SVC_GUARD:-1}" = 0 ]; then
  PATH=$JUDGE_PATH "$BASH_BIN" /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1
else
  PATH=$JUDGE_PATH "$SYSPY" -I $SVC/guard.py --report "$OUT/guard.json" -- "$BASH_BIN" /tests/test.sh \
    > /logs/verifier/test-stdout.txt 2>&1
fi
rc=$?
phase judged
echo "$rc" > "$OUT/test.rc"
exit $rc
