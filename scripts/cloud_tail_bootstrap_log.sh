#!/usr/bin/env bash
set -eu

log=$(ls -1t /home/jovyan/hmoe-cloud/logs/bootstrap-torch251-* 2>/dev/null | head -1)
echo "LOG=$log"
tail -n 240 "$log"
