#!/usr/bin/env bash
set -eu

log=$(ls -1t /home/jovyan/logs/te_image_probe-* | head -1)
echo "LOG=$log"
tail -n 100 "$log"
