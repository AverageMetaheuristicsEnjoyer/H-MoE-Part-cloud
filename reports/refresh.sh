#!/usr/bin/env bash
# Pull the points both sweeps have recorded so far and rebuild the PDF.
#
#   reports/refresh.sh
#
# Runs from the workstation, not in a job. The workspace disk is not reachable from
# here, so the only way to read a sweep's table is to ask it to print one: each entry
# script has an `export` mode, and this submits both, waits, and keeps the `PT` lines.
# Safe to run while the sweeps are still going -- a missing point is a dash in the PDF.
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/results"

MOE_REPO=https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud
DENSE_REPO=https://github.com/AverageMetaheuristicsEnjoyer/efficient-training-membench-cloud

submit() {
  ssh brain_lab "mlsub run --repo $1 --branch $2 --entry $3 --gpus cpu --no-pip \
    --note membench-export --args export" 2>/dev/null |
    sed -n 's/.*"job_name": "\([^"]*\)".*/\1/p'
}

collect() {  # job id, destination -- keeps the previous file if the job brings nothing
  local job=$1 destination=$2 status=""
  for _ in $(seq 1 40); do
    status=$(ssh brain_lab "mlsub status $job" 2>/dev/null |
      sed -n 's/.*"status": "\([^"]*\)".*/\1/p')
    [[ $status == completed || $status == failed ]] && break
    sleep 45
  done
  local rows
  rows=$(ssh brain_lab "mlsub logs $job --tail 400" 2>/dev/null |
    sed 's/.*<stdout>://' | grep -E '^PT' | sort -u)
  if [[ -z $rows ]]; then
    echo "  $job ($status): no rows; keeping $(basename "$destination")"
    return
  fi
  printf '%s\n' "$rows" > "$destination"
  echo "  $job ($status): $(grep -cvE '^PT\s+model' "$destination") points -> $(basename "$destination")"
}

echo "submitting export jobs"
moe_job=$(submit "$MOE_REPO" bench/fp8-membench scripts/cloud_membench.sh)
dense_job=$(submit "$DENSE_REPO" main scripts/cloud_run.sh)
echo "  moe=$moe_job dense=$dense_job"

collect "$moe_job" "$root/results/moe-points.tsv"
collect "$dense_job" "$root/results/dense-points.tsv"

python3 "$root/reports/make_membench_pdf.py" "$root/reports/membench.tex"
xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory="$root/reports" "$root/reports/membench.tex" >/dev/null 2>&1
status=$?
rm -f "$root/reports/membench.aux" "$root/reports/membench.log"
if [[ $status -ne 0 || ! -f $root/reports/membench.pdf ]]; then
  echo "xelatex failed; rerun it by hand to see why" >&2
  exit 1
fi
echo "reports/membench.pdf: $(du -h "$root/reports/membench.pdf" | cut -f1)"
