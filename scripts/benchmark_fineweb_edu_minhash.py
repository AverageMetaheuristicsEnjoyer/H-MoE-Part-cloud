#!/usr/bin/env python3

import argparse
import hashlib
import importlib.metadata
import json
import platform
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path


FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb_edu_100BT-shuffled"
FINEWEB_EDU_REVISION = "be6b2a50d3a9c60d330c45384e80c7863cd3a25d"
DATATROVE_COMMIT = "87f7bad5c4a56ec648265fbf0b91d7d226bad428"
N_GRAMS = 5
NUM_BUCKETS = 14
HASHES_PER_BUCKET = 8
SEED = 1
STAGE_WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datatrove-root", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--train-selection", type=Path)
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--final-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-per-source", default=15_000, type=int)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path):
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )


def setup_datatrove(datatrove_root):
    datatrove_root = datatrove_root.resolve()
    revision = git_revision(datatrove_root)
    if revision != DATATROVE_COMMIT:
        raise RuntimeError(
            f"DataTrove commit is {revision}, expected {DATATROVE_COMMIT}"
        )
    sys.path.insert(0, str(datatrove_root / "src"))


def load_sources(
    plan_path,
    selection_path,
    development_manifest_path,
    final_manifest_path,
    limit_per_source,
):
    from audit_fineweb_edu_exact import load_index_lengths

    plan_bytes = plan_path.read_bytes()
    selection_bytes = selection_path.read_bytes()
    plan = json.loads(plan_bytes)
    selection = json.loads(selection_bytes)

    if plan["schema_version"] != 2:
        raise RuntimeError("unsupported source-plan schema")
    if plan["dataset"] != {
        "repo": FINEWEB_EDU_REPO,
        "revision": FINEWEB_EDU_REVISION,
    }:
        raise RuntimeError("source plan does not contain the pinned dataset")
    if selection["schema_version"] != 1:
        raise RuntimeError("unsupported train-selection schema")
    if selection["dataset"] != plan["dataset"]:
        raise RuntimeError("train selection and source plan use different datasets")

    training = plan["training"]
    train_sources = []
    previous_shard = None
    for record in selection["selected_shards"]:
        shard = record["shard"]
        source_path = record["source_path"]
        expected_path = training["path_template"].format(shard=shard)
        if source_path != expected_path:
            raise RuntimeError(f"unexpected source for train shard {shard}")
        if previous_shard is None:
            if shard != training["shard_start_inclusive"]:
                raise RuntimeError(
                    "selected training shards do not start at the plan boundary"
                )
        elif shard != previous_shard + 1:
            raise RuntimeError("selected training shards are not contiguous")
        if not (
            training["shard_start_inclusive"]
            <= shard
            < training["shard_end_exclusive"]
        ):
            raise RuntimeError(f"train shard is outside candidate range: {shard}")
        train_sources.append(
            {
                "available_documents": record["indexed_documents"],
                "available_indexed_tokens": record["indexed_tokens"],
                "index_path": str(
                    selection_path.parent / f"{record['output_prefix']}.idx"
                ),
                "source_path": source_path,
                "split": "train",
                "train_shard": shard,
            }
        )
        previous_shard = shard
    if not train_sources:
        raise RuntimeError("train selection contains no source files")

    sources = train_sources
    heldout_inputs = {}
    for split, field, manifest_path in (
        ("development", "development", development_manifest_path),
        ("final", "final", final_manifest_path),
    ):
        split_paths = plan[field]
        if len(split_paths) != 1:
            raise RuntimeError(f"source plan must contain one {split} path")
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        selection_record = manifest["selection"]
        prefix_name = Path(manifest["conversion"]["output_prefix"]).name
        bin_record = manifest["artifacts"][f"{prefix_name}.bin"]
        if (
            manifest["schema_version"] != 2
            or manifest["source"]["repo"] != FINEWEB_EDU_REPO
            or manifest["source"]["revision"] != FINEWEB_EDU_REVISION
            or manifest["source"]["path"] != split_paths[0]
            or selection_record["minimum_indexed_tokens"]
            != plan[f"{field}_minimum_indexed_tokens"]
            or selection_record["indexed_documents"]
            != bin_record["indexed_documents"]
            or selection_record["indexed_tokens"]
            != bin_record["indexed_tokens"]
        ):
            raise RuntimeError(f"unexpected {split} manifest")
        prefix = manifest_path.parent / manifest["conversion"]["output_prefix"]
        sources.append(
            {
                "available_documents": selection_record["indexed_documents"],
                "available_indexed_tokens": selection_record["indexed_tokens"],
                "index_path": f"{prefix}.idx",
                "source_path": split_paths[0],
                "split": split,
            }
        )
        heldout_inputs[split] = {
            "artifact_path": manifest_path.name,
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }

    paths = [source["source_path"] for source in sources]
    if len(paths) != len(set(paths)):
        raise RuntimeError("audit source paths are not disjoint")
    for rank, source in enumerate(sources):
        path = Path(source["source_path"])
        if path.parent.as_posix() != "data":
            raise RuntimeError(f"source is not directly under data/: {path}")
        source["file_id"] = rank
        source["document_limit"] = (
            source["available_documents"]
            if limit_per_source == -1
            else min(limit_per_source, source["available_documents"])
        )

    indexed_lengths = {}
    for source in sources:
        path = Path(source.pop("index_path"))
        lengths = load_index_lengths(path)
        if (
            len(lengths) != source["available_documents"]
            or int(lengths.sum()) != source["available_indexed_tokens"]
        ):
            raise RuntimeError(f"indexed counts differ for {source['source_path']}")
        lengths = lengths[: source["document_limit"]]
        source["audited_indexed_tokens"] = int(lengths.sum())
        indexed_lengths[source["file_id"]] = lengths

    return sources, indexed_lengths, {
        **heldout_inputs,
        "source_plan": {
            "artifact_path": plan_path.name,
            "sha256": hashlib.sha256(plan_bytes).hexdigest(),
        },
        "train_selection": {
            "artifact_path": selection_path.name,
            "sha256": hashlib.sha256(selection_bytes).hexdigest(),
        },
    }


def limit_rank_documents(data, rank, world_size, *, sources):
    if world_size != len(sources):
        raise RuntimeError("stage-1 rank count does not match source count")
    for _, document in zip(range(sources[rank]["document_limit"]), data):
        yield document


def record_document_ledger(
    data,
    rank,
    world_size,
    *,
    sources,
    ledger_dir,
):
    if world_size != len(sources):
        raise RuntimeError("stage-1 rank count does not match source count")
    source = sources[rank]
    path = Path(ledger_dir) / f"{rank:05d}.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for document_index, document in enumerate(data):
            text_bytes = document.text.encode("utf-8")
            record = {
                "document_id": document.id,
                "document_index": document_index,
                "file_id": rank,
                "source_path": source["source_path"],
                "source_utf8_bytes": len(text_bytes),
                "split": source["split"],
            }
            stream.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            yield document


def read_fixed_records(path, format_string):
    record_size = struct.calcsize(format_string)
    with path.open("rb") as stream:
        while chunk := stream.read(record_size):
            if len(chunk) != record_size:
                raise RuntimeError(f"truncated record in {path}")
            yield struct.unpack(format_string, chunk)


def load_document_ledger(ledger_dir, sources, indexed_lengths):
    source_count = len(sources)
    paths = sorted(ledger_dir.glob("*.jsonl"))
    expected_paths = [
        ledger_dir / f"{file_id:05d}.jsonl"
        for file_id in range(source_count)
    ]
    if paths != expected_paths:
        raise RuntimeError("document ledger does not contain one file per source rank")

    documents = {}
    for file_id, path in enumerate(paths):
        next_document_index = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                key = (record["file_id"], record["document_index"])
                if record["file_id"] != file_id:
                    raise RuntimeError(f"wrong file_id in {path}")
                if record["document_index"] != next_document_index:
                    raise RuntimeError(f"non-contiguous document indices in {path}")
                if (
                    record["source_path"] != sources[file_id]["source_path"]
                    or record["split"] != sources[file_id]["split"]
                ):
                    raise RuntimeError(f"wrong source provenance in {path}")
                record["indexed_tokens"] = int(
                    indexed_lengths[file_id][next_document_index]
                )
                documents[key] = record
                next_document_index += 1
        if next_document_index != sources[file_id]["document_limit"]:
            raise RuntimeError(
                f"wrong document count for {sources[file_id]['source_path']}: "
                f"{next_document_index}"
            )
    return documents


def load_signature_keys(signatures_dir, source_count):
    signature_format = f"<{HASHES_PER_BUCKET}QI"
    reference_keys = None
    for bucket in range(NUM_BUCKETS):
        bucket_dir = signatures_dir / f"bucket_{bucket:03d}"
        paths = sorted(bucket_dir.glob("*.minhash.sig"))
        expected_paths = [
            bucket_dir / f"{file_id:05d}.minhash.sig"
            for file_id in range(source_count)
        ]
        if paths != expected_paths:
            raise RuntimeError(
                f"signature bucket {bucket} does not contain one file "
                "per source rank"
            )

        bucket_keys = set()
        for file_id, path in enumerate(paths):
            for record in read_fixed_records(path, signature_format):
                key = (file_id, record[-1])
                if key in bucket_keys:
                    raise RuntimeError(
                        f"duplicate signature key in bucket {bucket}: {key}"
                    )
                bucket_keys.add(key)
        if reference_keys is None:
            reference_keys = bucket_keys
        elif bucket_keys != reference_keys:
            raise RuntimeError(
                f"signature document keys differ in bucket {bucket}"
            )
    return reference_keys


def load_cluster_metadata(clusters_dir):
    cluster_ids = {}
    cluster_sizes = {}
    for suffix, output in (("clusters", cluster_ids), ("sizes", cluster_sizes)):
        for path in sorted(clusters_dir.glob(f"*.{suffix}")):
            stem = path.name.removesuffix(f".{suffix}")
            if len(stem) != 6 or not stem.isdigit():
                raise RuntimeError(f"unexpected cluster metadata filename: {path}")
            file_id = int(stem)
            for document_index, value in read_fixed_records(path, "<II"):
                key = (file_id, document_index)
                if key in output:
                    raise RuntimeError(f"duplicate cluster metadata key: {key}")
                output[key] = value
    if cluster_ids.keys() != cluster_sizes.keys():
        raise RuntimeError("cluster ID and size records cover different documents")

    grouped = defaultdict(list)
    for key, cluster_id in cluster_ids.items():
        grouped[cluster_id].append(key)
    for cluster_id, keys in grouped.items():
        expected_size = len(keys)
        if expected_size < 2:
            raise RuntimeError(f"singleton MinHash cluster: {cluster_id}")
        if any(cluster_sizes[key] != expected_size for key in keys):
            raise RuntimeError(f"inconsistent size for MinHash cluster: {cluster_id}")
    return grouped


def cross_split_candidates(clusters, documents):
    candidates = []
    for cluster_id in sorted(clusters):
        keys = sorted(clusters[cluster_id])
        members = [documents[key] for key in keys]
        split_counts = Counter(member["split"] for member in members)
        if "train" not in split_counts or not (
            "development" in split_counts or "final" in split_counts
        ):
            continue
        candidates.append(
            {
                "candidate_cluster_id": cluster_id,
                "cluster_size": len(members),
                "members": members,
                "split_counts": dict(sorted(split_counts.items())),
            }
        )
    return candidates


def coverage_summary(documents, signature_keys):
    groups = {
        "all": list(documents),
    }
    for split in sorted({record["split"] for record in documents.values()}):
        groups[f"split:{split}"] = [
            key for key, record in documents.items() if record["split"] == split
        ]
    for source_path in sorted(
        {record["source_path"] for record in documents.values()}
    ):
        groups[f"source:{source_path}"] = [
            key
            for key, record in documents.items()
            if record["source_path"] == source_path
        ]

    summary = {}
    for name, keys in groups.items():
        signed_keys = [key for key in keys if key in signature_keys]
        unsigned_keys = [key for key in keys if key not in signature_keys]

        def totals(selected_keys):
            return {
                "documents": len(selected_keys),
                "indexed_tokens": sum(
                    documents[key]["indexed_tokens"] for key in selected_keys
                ),
                "source_utf8_bytes": sum(
                    documents[key]["source_utf8_bytes"] for key in selected_keys
                ),
            }

        all_totals = totals(keys)
        signed_totals = totals(signed_keys)
        summary[name] = {
            "all": all_totals,
            "with_5gram_signature": signed_totals,
            "without_5gram_signature": totals(unsigned_keys),
            "signature_coverage_fraction": {
                unit: (
                    signed_totals[unit] / all_totals[unit]
                    if all_totals[unit]
                    else None
                )
                for unit in (
                    "documents",
                    "indexed_tokens",
                    "source_utf8_bytes",
                )
            },
        }
    return summary


def recursive_artifact_hashes(output_dir, roots):
    records = []
    for relative_root in roots:
        root = output_dir / relative_root
        paths = [root] if root.is_file() else sorted(
            path for path in root.rglob("*") if path.is_file()
        )
        for path in paths:
            records.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    records.sort(key=lambda record: record["path"])
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "files": records,
        "roots": roots,
        "tree_sha256": digest.hexdigest(),
    }


def make_minhash_config():
    from datatrove.pipeline.dedup.minhash import MinhashConfig
    from datatrove.utils.hashing import HashConfig

    return MinhashConfig(
        hash_config=HashConfig(hash_fc="sha1", precision=64),
        n_grams=N_GRAMS,
        num_buckets=NUM_BUCKETS,
        hashes_per_bucket=HASHES_PER_BUCKET,
        seed=SEED,
    )


def run_self_test():
    from datatrove.data import Document
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import (
        MinhashDedupBuckets,
        MinhashDedupCluster,
    )
    from datatrove.utils.typeshelper import Languages

    with tempfile.TemporaryDirectory(prefix="fineweb-edu-minhash-fixture-") as tmp:
        root = Path(tmp)
        config = make_minhash_config()
        repeated = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        )
        heldout_repeated = (
            "redwood saffron cobalt marble willow comet harbor thistle"
        )
        documents_by_rank = [
            [
                Document(text=repeated, id="train-common"),
                Document(
                    text="quartz violin meadow lantern copper orbit velvet prism",
                    id="train-unique",
                ),
                Document(text="tiny", id="train-tiny"),
            ],
            [
                Document(text=repeated, id="development-common"),
                Document(text=heldout_repeated, id="development-heldout-only"),
            ],
            [
                Document(text=repeated, id="final-common"),
                Document(text=heldout_repeated, id="final-heldout-only"),
                Document(text="small", id="final-tiny"),
            ],
        ]
        signatures_dir = root / "signatures"
        buckets_dir = root / "buckets"
        clusters_dir = root / "clusters"
        signature = MinhashDedupSignature(
            output_folder=str(signatures_dir),
            config=config,
            language=Languages.english,
        )
        for rank, documents in enumerate(documents_by_rank):
            signature.run(
                iter(documents),
                rank=rank,
                world_size=len(documents_by_rank),
            )
        buckets = MinhashDedupBuckets(
            input_folder=str(signatures_dir),
            output_folder=str(buckets_dir),
            config=config,
            only_dedup_in_index=False,
        )
        for rank in range(NUM_BUCKETS):
            buckets.run(None, rank=rank, world_size=NUM_BUCKETS)
        cluster = MinhashDedupCluster(
            input_folder=str(buckets_dir),
            output_folder=str(clusters_dir),
            config=config,
            save_cluster_id=True,
            save_cluster_size=True,
        )
        cluster.run(None, world_size=1)

        splits = ("train", "development", "final")
        ledger = {
            (file_id, document_index): {
                "split": splits[file_id],
                "document_id": document.id,
            }
            for file_id, documents in enumerate(documents_by_rank)
            for document_index, document in enumerate(documents)
        }
        signature_keys = load_signature_keys(
            signatures_dir, len(documents_by_rank)
        )
        clusters = load_cluster_metadata(clusters_dir)
        candidates = cross_split_candidates(clusters, ledger)
        if (0, 2) in signature_keys or (2, 2) in signature_keys:
            raise RuntimeError(
                "fixture tiny document unexpectedly received a signature"
            )
        candidate_ids = [
            {member["document_id"] for member in candidate["members"]}
            for candidate in candidates
        ]
        if candidate_ids != [
            {"train-common", "development-common", "final-common"}
        ]:
            raise RuntimeError("fixture train-to-heldout cluster report differs")
        if not any(
            set(keys) == {(1, 1), (2, 1)}
            for keys in clusters.values()
        ):
            raise RuntimeError("fixture heldout-only duplicate was not clustered")
        if (0, 1) in {
            key for keys in clusters.values() for key in keys
        }:
            raise RuntimeError("fixture unique document entered a cluster")
    print("self_test=pass")


def main():
    args = parse_args()
    setup_datatrove(args.datatrove_root)
    if args.self_test:
        run_self_test()
        return
    if any(
        value is None
        for value in (
            args.plan,
            args.train_selection,
            args.development_manifest,
            args.final_manifest,
            args.output_dir,
        )
    ):
        raise ValueError(
            "--plan, --train-selection, --development-manifest, "
            "--final-manifest, and --output-dir are required unless "
            "--self-test is used"
        )
    if args.limit_per_source == 0 or args.limit_per_source < -1:
        raise ValueError("--limit-per-source must be -1 or positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import (
        MinhashDedupBuckets,
        MinhashDedupCluster,
    )
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.utils.typeshelper import Languages

    total_start = time.perf_counter()
    sources, indexed_lengths, inputs = load_sources(
        args.plan,
        args.train_selection,
        args.development_manifest,
        args.final_manifest,
        args.limit_per_source,
    )
    source_paths_path = args.output_dir / "source-paths.txt"
    source_paths_path.write_text(
        "".join(f"{Path(source['source_path']).name}\n" for source in sources),
        encoding="utf-8",
    )

    signatures_dir = args.output_dir / "signatures"
    buckets_dir = args.output_dir / "buckets"
    clusters_dir = args.output_dir / "clusters"
    ledger_dir = args.output_dir / "document-ledger"
    ledger_dir.mkdir()
    config = make_minhash_config()
    signature = MinhashDedupSignature(
        output_folder=str(signatures_dir),
        config=config,
        language=Languages.english,
    )
    if (
        signature.word_tokenizer.__class__.__name__ != "SpaCyTokenizer"
        or signature.word_tokenizer.language != "en"
    ):
        raise RuntimeError("DataTrove English did not resolve to spaCy English")

    source_folder = (
        f"hf://datasets/{FINEWEB_EDU_REPO}"
        f"@{FINEWEB_EDU_REVISION}/data"
    )
    signature_start = time.perf_counter()
    LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=source_folder,
                paths_file=str(source_paths_path.resolve()),
                limit=-1,
                read_metadata=False,
                recursive=False,
                shuffle_files=False,
            ),
            partial(limit_rank_documents, sources=sources),
            partial(
                record_document_ledger,
                sources=sources,
                ledger_dir=str(ledger_dir),
            ),
            signature,
        ],
        tasks=len(sources),
        workers=min(STAGE_WORKERS, len(sources)),
        logging_dir=str(args.output_dir / "logs-signatures"),
        skip_completed=False,
    ).run()
    signature_seconds = time.perf_counter() - signature_start

    buckets_start = time.perf_counter()
    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(signatures_dir),
                output_folder=str(buckets_dir),
                config=config,
                only_dedup_in_index=False,
            )
        ],
        tasks=NUM_BUCKETS,
        workers=min(STAGE_WORKERS, NUM_BUCKETS),
        logging_dir=str(args.output_dir / "logs-buckets"),
        skip_completed=False,
    ).run()
    buckets_seconds = time.perf_counter() - buckets_start

    cluster_start = time.perf_counter()
    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=str(buckets_dir),
                output_folder=str(clusters_dir),
                config=config,
                save_cluster_id=True,
                save_cluster_size=True,
            )
        ],
        tasks=1,
        workers=1,
        logging_dir=str(args.output_dir / "logs-clusters"),
        skip_completed=False,
    ).run()
    cluster_seconds = time.perf_counter() - cluster_start

    report_start = time.perf_counter()
    documents = load_document_ledger(
        ledger_dir,
        sources,
        indexed_lengths,
    )
    for source in sources:
        source_documents = [
            record
            for (file_id, _), record in documents.items()
            if file_id == source["file_id"]
        ]
        audited_indexed_tokens = sum(
            record["indexed_tokens"] for record in source_documents
        )
        if audited_indexed_tokens != source["audited_indexed_tokens"]:
            raise RuntimeError(
                f"audited token total differs for {source['source_path']}"
            )
        source["audited_source_utf8_bytes"] = sum(
            record["source_utf8_bytes"] for record in source_documents
        )
    signature_keys = load_signature_keys(signatures_dir, len(sources))
    if not signature_keys.issubset(documents):
        raise RuntimeError("signature output references an unknown document")
    clusters = load_cluster_metadata(clusters_dir)
    if any(key not in documents for keys in clusters.values() for key in keys):
        raise RuntimeError("cluster output references an unknown document")

    short_documents = []
    for key in sorted(documents):
        if key not in signature_keys:
            short_documents.append(
                documents[key]
                | {
                    "coverage_reason": (
                        "no 5-gram signature after the pinned FineWeb "
                        "normalization and spaCy English tokenization"
                    )
                }
            )
    short_path = args.output_dir / "short-documents.jsonl"
    write_jsonl(short_path, short_documents)

    candidates = cross_split_candidates(clusters, documents)
    candidates_path = args.output_dir / "cross-split-candidate-clusters.jsonl"
    write_jsonl(candidates_path, candidates)
    coverage = coverage_summary(documents, signature_keys)

    stable_roots = [
        "source-paths.txt",
        "document-ledger",
        "signatures",
        "buckets",
        "clusters",
        "short-documents.jsonl",
        "cross-split-candidate-clusters.jsonl",
    ]
    artifact_hashes = recursive_artifact_hashes(args.output_dir, stable_roots)
    hashes_path = args.output_dir / "artifact-hashes.json"
    write_json(hashes_path, artifact_hashes)
    report_seconds = time.perf_counter() - report_start

    report = {
        "artifact_hashes": {
            "path": hashes_path.name,
            "sha256": sha256_file(hashes_path),
            "tree_sha256": artifact_hashes["tree_sha256"],
        },
        "candidate_cross_split_clusters": {
            "clusters": len(candidates),
            "documents": sum(candidate["cluster_size"] for candidate in candidates),
            "path": candidates_path.name,
        },
        "configuration": {
            "audit_scope": (
                "full" if args.limit_per_source == -1 else "bounded"
            ),
            "hash": "sha1",
            "hash_precision": 64,
            "hashes_per_bucket": HASHES_PER_BUCKET,
            "language": "English via SpaCyTokenizer(en)",
            "n_grams": N_GRAMS,
            "num_buckets": NUM_BUCKETS,
            "requested_document_limit_per_source": args.limit_per_source,
            "seed": SEED,
            "source_document_limits": {
                source["source_path"]: source["document_limit"]
                for source in sources
            },
            "stage1_tasks": len(sources),
            "stage_workers": STAGE_WORKERS,
        },
        "coverage": coverage,
        "environment": {
            "datatrove_commit": DATATROVE_COMMIT,
            "python": platform.python_version(),
            "spacy": importlib.metadata.version("spacy"),
        },
        "hash_scope_excludes": [
            "artifact-hashes.json",
            "benchmark-report.json",
            "logs-buckets",
            "logs-clusters",
            "logs-signatures",
        ],
        "inputs": inputs,
        "policy": {
            "candidate_only": True,
            "data_filtered": False,
            "data_mutated": False,
        },
        "schema_version": 1,
        "short_document_coverage": {
            "documents_without_signature": len(short_documents),
            "indexed_tokens_without_signature": sum(
                record["indexed_tokens"] for record in short_documents
            ),
            "path": short_path.name,
            "source_utf8_bytes_without_signature": sum(
                record["source_utf8_bytes"] for record in short_documents
            ),
        },
        "sources": sources,
        "timings_seconds": {
            "buckets": buckets_seconds,
            "cluster": cluster_seconds,
            "report_and_hashes": report_seconds,
            "signatures_and_source_read": signature_seconds,
            "total": time.perf_counter() - total_start,
        },
    }
    report_path = args.output_dir / "benchmark-report.json"
    write_json(report_path, report)
    print(f"candidate_cross_split_clusters={len(candidates)}")
    print(f"documents_without_signature={len(short_documents)}")
    print(f"artifact_tree_sha256={artifact_hashes['tree_sha256']}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
