#!/bin/bash
set -euo pipefail
mkdir -p /var/log/researchboost
# /tmp is the trial's scratch directory (<scratch root>/gaussian_splatting on the host, one trial per host at a time): start it
# empty, so that nothing of an earlier trial reaches this one.
find /tmp -mindepth 1 -delete
chmod 1777 /tmp
# The agent has internet access; the container must hold no credentials.
/usr/bin/python3 -I /opt/researchboost/no_credentials.py
# Root-only preparation completes before Harbor's readiness check allows agent setup.
if ! /opt/gs-env/bin/python /opt/researchboost/bootstrap.py > /var/log/researchboost/prepare.log 2>&1; then
    chmod 644 /var/log/researchboost/prepare.log
    echo "Gaussian Splatting preparation failed; last lines of /var/log/researchboost/prepare.log:" >&2
    tail -n 200 /var/log/researchboost/prepare.log >&2
    exit 1
fi
chmod -R a+rX /opt/researchboost/ws
chmod 644 /var/log/researchboost/prepare.log
touch /run/researchboost-ready
exec "$@"
