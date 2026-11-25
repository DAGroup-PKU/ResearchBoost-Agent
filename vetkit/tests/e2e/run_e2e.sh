#!/bin/bash
# End-to-end test of the reference-pack path on a CPU-only toy task (no GPU needed; Docker with Compose v2 is):
# onboard_check, leak_guard, refpack build (twice: the second finds the pack), verify, judge of four submissions
# (unchanged -> UNAVAILABLE, faster -> PASS, wrong output -> INCORRECT, faster with programs of its own in its
# candidate environment that share names with system tools -> PASS, none of them run as root), and a tampered pack
# (verify -> exit 81).
#   bash vetkit/tests/e2e/run_e2e.sh            # E2E_WORK=<dir> to choose the work directory, KEEP=1 to keep it
# The work directory must be visible to the Docker daemon (on a laptop VM, somewhere under your home directory).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
KIT=$(cd "$HERE/../.." && pwd)
W=${E2E_WORK:-$HERE/.work}
PY=${PYTHON:-python3}
IMG=vetkit-toytask:e2e
rm -rf "$W"; mkdir -p "$W/bin" "$W/assets/toytask" "$W/scratch"
# A CPU-only task takes the pack key's GPU fields from the host GPUs named by --gpu-ids: a stub on a machine without one.
if ! command -v nvidia-smi >/dev/null; then
  printf '#!/bin/sh\necho "0, NVIDIA H200, 570.00.00, GPU-00000000-e2e"\n' > "$W/bin/nvidia-smi"; chmod +x "$W/bin/nvidia-smi"
  export PATH="$W/bin:$PATH"
fi
printf 'RESEARCHBOOST_ASSETS_ROOT=%s\nRESEARCHBOOST_SCRATCH_ROOT=%s\n' "$W/assets" "$W/scratch" > "$W/r.env"
step() { printf '\n== %s\n' "$*"; }
fail() { echo "E2E FAIL: $*" >&2; exit 1; }

step "lint and leak scan"
"$PY" "$KIT/bin/onboard_check.py" "$HERE/tasks/toytask" --allow-dirty > "$W/lint.json" || { cat "$W/lint.json"; fail lint; }
"$PY" "$KIT/bin/leak_guard.py" "$HERE/tasks/toytask" > "$W/leak.json" || { cat "$W/leak.json"; fail leak; }

step "image"
docker build -q -t $IMG "$HERE/tasks/toytask/environment" > /dev/null

RP=("$PY" "$KIT/bin/refpack.py")
COMMON=(--task toytask --tasks-dir "$HERE/tasks" --image $IMG --env-file "$W/r.env" --gpu-ids 0 --store "$W/packs"
        --work "$W/work" --python /usr/local/bin/python3 --adapter generic
        --timing-file task/timing_ref.json --timing-field s_per_update)
[ "$(id -u)" = 0 ] || COMMON+=(--test-non-root)     # on a vetting host, run as root
step "build"
"${RP[@]}" build "${COMMON[@]}" --spread-max 0.5 | tee "$W/build.json"
grep -q '"result": "published"' "$W/build.json" || fail "build did not publish"
"${RP[@]}" build "${COMMON[@]}" | grep -q '"result": "exists"' || fail "second build did not find the pack"
step "verify"
"${RP[@]}" verify "${COMMON[@]}" | tee "$W/verify.json"
grep -q '"result": "verified"' "$W/verify.json" || fail "verify"

# Harbor's artifacts/ layout of a finished trial: manifest.json + the recorded copies
artifacts() {   # artifacts <dir> <train.py source>
  mkdir -p "$1/tree" "$1/env"
  cp "$2" "$1/tree/train.py"; echo '{}' > "$1/switches.json"; echo 'pkg' > "$1/env/pkg.txt"
  cat > "$1/manifest.json" <<JSON
[{"source": "/submission/toytask", "destination": "artifacts/tree", "type": "directory", "status": "ok"},
 {"source": "/submission/switches.json", "destination": "artifacts/switches.json", "type": "file", "status": "ok"},
 {"source": "/opt/candidate-env", "destination": "artifacts/env", "type": "directory", "status": "ok"}]
JSON
}
REF="$HERE/tasks/toytask/environment/reference/train.py"
artifacts "$W/sub-unchanged/artifacts" "$REF"
sed 's/^assert work(N) == acc$/pass  # the re-check removed: same output, half the work/' "$REF" > "$W/fast.py"
artifacts "$W/sub-fast/artifacts" "$W/fast.py"
sed 's/% 1000003/% 1000033/' "$REF" > "$W/wrong.py"
artifacts "$W/sub-wrong/artifacts" "$W/wrong.py"
# The candidate environment comes first on the image's PATH. Programs of the submission's own there, named like the
# tools a judge runs, must never run as root (protocol supplement C10): each leaves a mark in the root-only verifier
# directory if it does, then runs the system tool.
artifacts "$W/sub-own-tools/artifacts" "$W/fast.py"
mkdir -p "$W/sub-own-tools/artifacts/env/bin"
for t in bash sh rm head cat chown chmod mkdir cp mv find date touch ls sed sort tail env timeout kill ps pgrep pkill \
         runuser su git nvidia-smi python python3 id; do
  cat > "$W/sub-own-tools/artifacts/env/bin/$t" <<'SH'
#!/bin/sh
n=${0##*/}
[ "$(/usr/bin/id -u)" = 0 ] && : > "/logs/verifier/ran-as-root-$n"
for d in /usr/local/bin /usr/sbin /usr/bin /sbin /bin; do [ -x "$d/$n" ] && exec "$d/$n" "$@"; done
exit 127
SH
  chmod 755 "$W/sub-own-tools/artifacts/env/bin/$t"
done
for s in unchanged fast wrong own-tools; do
  step "judge $s"
  "${RP[@]}" judge "${COMMON[@]}" --artifacts "$W/sub-$s/artifacts" --out "$W/judged-$s" | tee "$W/judge-$s.json" || true
done
"$PY" - "$W" <<'PYEOF'
import json, sys
w = sys.argv[1]
import glob
want = {'unchanged': ('UNAVAILABLE', 0), 'fast': ('PASS', 1), 'wrong': ('INCORRECT', 0), 'own-tools': ('PASS', 1)}
for s, (status, accepted) in want.items():
    j = json.load(open('%s/judged-%s/judge.json' % (w, s)))
    got = (j['status'], (j['reward'] or {}).get('accepted'))
    print('%-9s %-11s accepted=%s speedup=%s' % (s, got[0], got[1], j['speedup']))
    assert j['result'] == 'judged' and got == (status, accepted), (s, j['result'], got)
marks = sorted(p.rsplit('/', 1)[1] for p in glob.glob('%s/judged-*/**/ran-as-root-*' % w, recursive=True))
print('own-tools: programs of the submission that ran as root: %s' % (marks or 'none'))
assert not marks, marks
PYEOF
step "tampered pack"
d=$(find "$W/packs/toytask" -mindepth 1 -maxdepth 1 -type d -print -quit)
[ "$(stat -c %a "$d" 2>/dev/null || stat -f %Lp "$d")" = 700 ] || fail "pack $d is not 0700"
f=$(find "$W/packs/toytask" -path '*files*' -name ref.json -print -quit)
chmod u+w "$f"; echo '{"label": "ref", "digest": "0"}' > "$f"
set +e; "${RP[@]}" verify "${COMMON[@]}" > "$W/verify-tampered.json"; rc=$?; set -e
[ $rc = 81 ] || { cat "$W/verify-tampered.json"; fail "tampered pack: exit $rc, want 81"; }
echo "tampered pack refused (exit 81)"
[ "${KEEP:-0}" = 1 ] || { docker rmi -f $IMG > /dev/null; rm -rf "$W"; }
echo; echo "E2E OK"
