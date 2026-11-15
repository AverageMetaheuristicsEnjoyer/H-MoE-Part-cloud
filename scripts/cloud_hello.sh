#!/usr/bin/env bash
# Does a job that touches no shared volume at all still complete? If even this hangs in
# Running, the stall is the platform's own logging (policy logs_dir lives on nfs2) and no
# amount of routing our own writes to /tmp will help. If it completes, the stall is our
# scripts touching a full volume, and that we can fix ourselves.
echo "HELLO_FROM_JOB=$(date -u +%H:%M:%SZ) image=${MLSUB_IMAGE:-unset} host=$(hostname)"
echo "HELLO_DONE"
exit 0
