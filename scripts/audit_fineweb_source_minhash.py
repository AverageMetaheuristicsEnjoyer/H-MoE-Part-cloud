#!/usr/bin/env python3

import argparse
import json
import tempfile
from collections import Counter
from functools import partial
from pathlib import Path

from benchmark_fineweb_edu_minhash import (
    FINEWEB_EDU_REPO,
    FINEWEB_EDU_REVISION,
    load_cluster_metadata,
    load_signature_keys,
    make_minhash_config,
    setup_datatrove,
)


def limit_documents(data, rank, world_size, *, limit):
    for _, document in zip(range(limit), data):
        yield document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datatrove-root", required=True, type=Path)
    parser.add_argument("--shard-start", required=True, type=int)
    parser.add_argument("--shard-end", required=True, type=int)
    parser.add_argument("--documents-per-shard", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    setup_datatrove(args.datatrove_root)

    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import (
        MinhashDedupBuckets,
        MinhashDedupCluster,
    )
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.utils.typeshelper import Languages

    shards = list(range(args.shard_start, args.shard_end))
    if not shards or args.documents_per_shard <= 0:
        raise ValueError("empty shard range or non-positive document count")
    with tempfile.TemporaryDirectory(prefix="fineweb-source-minhash-") as tmp:
        scratch = Path(tmp)
        paths = scratch / "source-paths.txt"
        paths.write_text(
            "".join(f"train-{shard:05d}-of-00100.parquet\n" for shard in shards),
            encoding="utf-8",
        )
        signatures = scratch / "signatures"
        buckets = scratch / "buckets"
        clusters = scratch / "clusters"
        config = make_minhash_config()
        source = (
            f"hf://datasets/{FINEWEB_EDU_REPO}"
            f"@{FINEWEB_EDU_REVISION}/data"
        )
        LocalPipelineExecutor(
            pipeline=[
                ParquetReader(
                    data_folder=source,
                    paths_file=str(paths),
                    limit=-1,
                    read_metadata=False,
                    recursive=False,
                    shuffle_files=False,
                ),
                partial(limit_documents, limit=args.documents_per_shard),
                MinhashDedupSignature(
                    output_folder=str(signatures),
                    config=config,
                    language=Languages.english,
                ),
            ],
            tasks=len(shards),
            workers=min(4, len(shards)),
            logging_dir=str(scratch / "logs-signatures"),
            skip_completed=False,
        ).run()
        LocalPipelineExecutor(
            pipeline=[
                MinhashDedupBuckets(
                    input_folder=str(signatures),
                    output_folder=str(buckets),
                    config=config,
                    only_dedup_in_index=False,
                )
            ],
            tasks=14,
            workers=4,
            logging_dir=str(scratch / "logs-buckets"),
            skip_completed=False,
        ).run()
        LocalPipelineExecutor(
            pipeline=[
                MinhashDedupCluster(
                    input_folder=str(buckets),
                    output_folder=str(clusters),
                    config=config,
                    save_cluster_id=True,
                    save_cluster_size=True,
                )
            ],
            tasks=1,
            workers=1,
            logging_dir=str(scratch / "logs-clusters"),
            skip_completed=False,
        ).run()
        signatures_keys = load_signature_keys(signatures, len(shards))
        candidate_clusters = load_cluster_metadata(clusters)

    candidate_keys = {
        key for keys in candidate_clusters.values() for key in keys
    }
    by_shard = Counter(shards[file_id] for file_id, _ in candidate_keys)
    cross_shard_clusters = sum(
        len({file_id for file_id, _ in keys}) > 1
        for keys in candidate_clusters.values()
    )
    report = {
        "schema_version": 1,
        "source": {"repo": FINEWEB_EDU_REPO, "revision": FINEWEB_EDU_REVISION},
        "shards": shards,
        "documents_per_shard": args.documents_per_shard,
        "signature_documents": len(signatures_keys),
        "signature_coverage": len(signatures_keys)
        / (len(shards) * args.documents_per_shard),
        "candidate_clusters": len(candidate_clusters),
        "candidate_documents": len(candidate_keys),
        "candidate_document_rate": len(candidate_keys) / len(signatures_keys),
        "cross_shard_candidate_clusters": cross_shard_clusters,
        "per_shard_candidate_documents": {
            str(shard): by_shard[shard] for shard in shards
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "SOURCE_MINHASH"
        f" shards={args.shard_start}:{args.shard_end}"
        f" signatures={len(signatures_keys)}"
        f" clusters={len(candidate_clusters)}"
        f" candidate_documents={len(candidate_keys)}"
        f" candidate_rate={report['candidate_document_rate']:.8f}"
        f" cross_shard_clusters={cross_shard_clusters}"
    )
    print(f"REPORT={args.output}")
    print("SOURCE_MINHASH_AUDIT=pass")


if __name__ == "__main__":
    main()
