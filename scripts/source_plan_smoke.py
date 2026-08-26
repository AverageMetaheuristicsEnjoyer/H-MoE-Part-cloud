#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    dataset = plan["dataset"]
    training = plan["training"]
    training_paths = [
        training["path_template"].format(shard=shard)
        for shard in range(
            training["shard_start_inclusive"],
            training["shard_end_exclusive"],
        )
    ]
    previous_training_paths = plan.get("previous_training", [])
    development_paths = plan["development"]
    final_paths = plan["final"]
    development_target = plan["development_minimum_indexed_tokens"]
    final_target = plan["final_minimum_indexed_tokens"]
    assert development_target > 0
    assert final_target > development_target

    previous_training_set = set(previous_training_paths)
    training_set = set(training_paths)
    development_set = set(development_paths)
    final_set = set(final_paths)
    assert len(training_set) == len(training_paths)
    assert len(previous_training_set) == len(previous_training_paths)
    assert previous_training_set.isdisjoint(training_set)
    assert previous_training_set.isdisjoint(development_set)
    assert previous_training_set.isdisjoint(final_set)
    assert training_set.isdisjoint(development_set)
    assert training_set.isdisjoint(final_set)
    assert development_set.isdisjoint(final_set)
    all_paths = previous_training_paths + training_paths + development_paths + final_paths
    assert len(all_paths) == 100

    api = HfApi()
    resolved_revision = api.dataset_info(
        dataset["repo"], revision=dataset["revision"]
    ).sha
    assert resolved_revision == dataset["revision"], resolved_revision
    entries = api.get_paths_info(
        dataset["repo"],
        paths=all_paths,
        repo_type="dataset",
        revision=dataset["revision"],
    )
    resolved = {entry.path: entry for entry in entries}
    assert set(resolved) == set(all_paths)
    assert all(resolved[path].size > 0 and resolved[path].lfs for path in all_paths)

    print(
        f"source_plan=pass previous_training={len(previous_training_paths)}"
        f" train_candidates={len(training_paths)}"
        f" development={len(development_paths)} final={len(final_paths)}"
        f" development_target={development_target} final_target={final_target}"
        f" revision={resolved_revision}"
        f" sha256={hashlib.sha256(raw).hexdigest()}"
    )


if __name__ == "__main__":
    main()
