#!/bin/bash
set -euo pipefail
mkdir -p /var/log/researchboost
# /tmp is the trial's scratch directory (<scratch root>/slime_grpo on the host, one trial per host at a time): start
# it empty, so that nothing of an earlier trial reaches this one.
find /tmp -mindepth 1 -delete
chmod 1777 /tmp
# The agent has internet access; the container must hold no credentials.
/usr/bin/python3 -I /opt/researchboost/no_credentials.py
# CUDA forward compatibility: on a driver older than CUDA 13 (the base image's rule, /opt/slime/cuda-compat.sh),
# every process of the container, the agent's commands included, loads the image's compat libcuda (cuda-compat-13-0).
# (no `| head` under pipefail: nvidia-smi gets SIGPIPE when head exits first, and the container stops with 141)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
driver=${driver%%$'\n'*}
case "${driver%%.*}" in
    535|550|565|570|575) echo /usr/local/cuda/compat > /etc/ld.so.conf.d/000-cuda-compat.conf; ldconfig ;;
esac
# Root-only preparation completes before Harbor's readiness check allows agent setup.
if ! /opt/sglang/bin/python3 /opt/researchboost/bootstrap.py > /var/log/researchboost/prepare.log 2>&1; then
    chmod 644 /var/log/researchboost/prepare.log
    echo "slime_grpo preparation failed; last lines of /var/log/researchboost/prepare.log:" >&2
    tail -n 200 /var/log/researchboost/prepare.log >&2
    exit 1
fi
chmod -R a+rX /opt/researchboost/ws /opt/researchboost/prep
chmod 644 /var/log/researchboost/prepare.log
touch /run/researchboost-ready
exec "$@"
