#!/usr/bin/env bash
# Were the original and extension corpora packed the same way?
#
#   mlsub run --entry scripts/cloud_compare_corpus_packing.sh --gpus cpu --image te4
#
# Switching training to the extension corpus mid-run costs 0.28 % of validation loss at a
# matched budget, and the replicate noise at that horizon is 0.003-0.008 %, so the effect is
# real. It is not distribution shift: both corpora are disjoint shard ranges of the same
# pre-shuffled HuggingFaceFW/fineweb_edu_100BT-shuffled revision, and neither run repeats
# data. The remaining candidate is the build itself -- the extension was tokenized by a
# separate job with its own pinned datatrove and Megatron, so a difference in packing or
# EOS handling would change the data statistics without changing the document distribution.
#
# Document lengths live in the .idx header alone, so this reads no payload and finishes in
# seconds. It also prints the exact-duplicate report if a previous audit left one, but never
# recomputes it: that walk hashes every document of both corpora and is hours of CPU.
set -u

base_root=${STAGE3_MOE_ORIGINAL_DATA_ROOT:-/home/jovyan/data/fineweb-edu-gpt2-megatron}
extension_root=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}
report=${STAGE3_MOE_DATA_AUDIT_OUTPUT:-/workspace-SR006.nfs2/hmoe-cloud/data-audit/base-extension-indexed-exact-v1.json}

echo "BASE_ROOT=$base_root"
echo "EXTENSION_ROOT=$extension_root"

if [[ -f $report ]]; then
  echo "=== stored exact-duplicate report ==="
  python - "$report" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for label in ("base", "extension"):
    v = r.get(label)
    if not v:
        continue
    print(f"{label}: documents={v['documents']} duplicate_occurrences={v['duplicate_occurrences']}"
          f" rate={v['duplicate_occurrence_rate']:.8f}"
          f" adjacent_length_correlation={v['adjacent_length_correlation']:.8f}")
    for split in ("development", "final"):
        o = v.get("heldout_overlap", {}).get(split)
        if o:
            print(f"   overlap with {split}: documents={o['matched_documents']}"
                  f" rate={o['matched_document_rate']:.8f} tokens={o['matched_indexed_tokens']}")
    if "base_overlap" in v:
        o = v["base_overlap"]
        print(f"   overlap with base: documents={o['matched_documents']} rate={o['matched_document_rate']:.8f}")
PY
else
  echo "NO_STORED_REPORT=$report (not recomputing: that walk hashes every document)"
fi

echo "=== document-length statistics from .idx headers ==="
python - "$base_root" "$extension_root" <<'PY'
import struct, sys
from pathlib import Path
import numpy as np

def lengths(prefix):
    idx = Path(f"{prefix}.idx")
    with idx.open("rb") as stream:
        if stream.read(9) != b"MMIDIDX\x00\x00":
            raise RuntimeError(f"bad index header: {idx}")
        version = struct.unpack("<Q", stream.read(8))[0]
        dtype_code = struct.unpack("<B", stream.read(1))[0]
        sequences = struct.unpack("<Q", stream.read(8))[0]
        doc_indices = struct.unpack("<Q", stream.read(8))[0]
    if version != 1 or doc_indices != sequences + 1:
        raise RuntimeError(f"unexpected layout: {idx} v={version} docs={doc_indices} seqs={sequences}")
    return dtype_code, np.memmap(idx, dtype=np.int32, mode="r", offset=34, shape=(sequences,))

def report(label, prefix):
    try:
        dtype_code, values = lengths(prefix)
    except Exception as exc:
        print(f"{label:34s} UNREADABLE {exc}")
        return
    values = np.asarray(values, dtype=np.int64)
    total = int(values.sum())
    # Adjacent-length correlation catches a different packing order: independently packed
    # documents are uncorrelated, anything grouped or sorted by length is not.
    if values.size > 2:
        a, b = values[:-1].astype(np.float64), values[1:].astype(np.float64)
        denom = a.std() * b.std()
        adjacent = float(((a - a.mean()) * (b - b.mean())).mean() / denom) if denom else float("nan")
    else:
        adjacent = float("nan")
    print(f"{label:34s} dtype={dtype_code} docs={values.size:>10d} tokens={total:>14d} "
          f"mean={values.mean():8.1f} median={np.median(values):8.1f} "
          f"p95={np.percentile(values, 95):9.1f} max={values.max():>8d} "
          f"adj_corr={adjacent:+.6f}")

roots = {"base": Path(sys.argv[1]), "extension": Path(sys.argv[2])}
for label, root in roots.items():
    if not root.exists():
        print(f"{label:34s} ROOT_MISSING {root}")
        continue
    found = sorted(root.rglob("*.idx"))
    if not found:
        print(f"{label:34s} NO_IDX_UNDER {root}")
        continue
    for idx in found:
        report(f"{label}:{idx.stem}", str(idx)[:-4])
print("PACKING_COMPARE=DONE")
PY
echo "PY_EXIT=$?"
exit 0
