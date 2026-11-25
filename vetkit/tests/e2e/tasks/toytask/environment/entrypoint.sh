#!/bin/bash
set -euo pipefail
find /tmp -mindepth 1 -delete; chmod 1777 /tmp
/usr/local/bin/python3 -I /opt/researchboost/no_credentials.py
/usr/local/bin/python3 /opt/researchboost/bootstrap.py > /var/log/researchboost/prepare.log 2>&1
chmod -R a+rX /opt/researchboost/ws
touch /run/researchboost-ready
exec "$@"
