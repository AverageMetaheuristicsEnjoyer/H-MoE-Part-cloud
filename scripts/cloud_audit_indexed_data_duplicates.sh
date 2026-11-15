#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
base=${STAGE3_MOE_ORIGINAL_DATA_ROOT:-/home/jovyan/data/fineweb-edu-gpt2-megatron}
extension=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}
output=${STAGE3_MOE_DATA_AUDIT_OUTPUT:-/workspace-SR006.nfs2/hmoe-cloud/data-audit/base-extension-indexed-exact-v1.json}

if [[ -f $output ]]; then
  python - "$output" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
for label in ("base", "extension"):
    value = report[label]
    print(
        f"DUPLICATE_RESULT label={label} documents={value['documents']}"
        f" duplicate_occurrences={value['duplicate_occurrences']}"
        f" duplicate_rate={value['duplicate_occurrence_rate']:.8f}"
        f" adjacent_length_correlation={value['adjacent_length_correlation']:.8f}"
    )
    for split in ("development", "final"):
        overlap = value["heldout_overlap"][split]
        print(
            f"HELDOUT_RESULT label={label} split={split}"
            f" matched_documents={overlap['matched_documents']}"
            f" matched_document_rate={overlap['matched_document_rate']:.8f}"
            f" matched_indexed_tokens={overlap['matched_indexed_tokens']}"
            f" matched_indexed_token_rate={overlap['matched_indexed_token_rate']:.8f}"
        )
    if "base_overlap" in value:
        overlap = value["base_overlap"]
        print(
            f"BASE_OVERLAP_RESULT label={label}"
            f" matched_documents={overlap['matched_documents']}"
            f" matched_document_rate={overlap['matched_document_rate']:.8f}"
            f" matched_unique_hashes={overlap['matched_unique_hashes']}"
        )
print(f"REPORT={sys.argv[1]}")
print("INDEXED_DUPLICATE_AUDIT=reused")
PY
  echo "EXIT=0"
  exit 0
fi

python "$root/scripts/audit_indexed_data_duplicates.py" \
  --base-root "$base" \
  --extension-root "$extension" \
  --output "$output"

echo "EXIT=0"
