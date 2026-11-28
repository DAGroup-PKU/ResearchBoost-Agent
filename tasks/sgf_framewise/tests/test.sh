#!/bin/sh
set -eu
umask 022
exec /usr/bin/python3 -I /tests/verify.py
