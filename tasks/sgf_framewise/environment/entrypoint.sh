#!/bin/sh
set -eu
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
/usr/bin/chmod 1777 /tmp
/usr/bin/python3 -I /opt/researchboost/no_credentials.py
/usr/bin/python3 -I /opt/researchboost/bootstrap.py
exec "$@"
