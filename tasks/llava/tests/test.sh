#!/bin/bash
set -euo pipefail
# Harbor uploads this directory into the task container after the agent finishes.
exec /usr/bin/python3 -I /tests/verify.py
