import hashlib
import json
from numbers import Real
from pathlib import Path


def _reported_metrics(results, higher_is_better):
    reported = {}
    for key, value in sorted(results.items()):
        if "," not in key or not isinstance(value, Real) or isinstance(value, bool):
            continue
        metric = key.split(",", 1)[0]
        direction = higher_is_better.get(metric)
        if not isinstance(direction, bool):
            continue
        reported[metric] = (float(value), direction)
    return reported


def _descendant_leaves(group, group_subtasks):
    leaves = []
    for child in group_subtasks.get(group, []):
        descendants = group_subtasks.get(child, [])
        if descendants:
            leaves.extend(_descendant_leaves(child, group_subtasks))
        else:
            leaves.append(child)
    return leaves


def _descendants(group, group_subtasks):
    descendants = set(group_subtasks.get(group, []))
    for child in list(descendants):
        descendants.update(_descendants(child, group_subtasks))
    return descendants


def _write_artifact(path, samples, metrics, *, prefix_doc_id=False):
    with path.open("w") as handle:
        for task, sample in samples:
            values = {metric: sample[metric] for metric in metrics if metric in sample}
            if not values:
                continue
            doc_id = sample.get("doc_id")
            if prefix_doc_id:
                doc_id = f"{task}:{doc_id}"
            handle.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "target": sample.get("target"),
                        "resps": sample.get("filtered_resps", sample.get("resps")),
                        "metrics": values,
                    },
                    sort_keys=True,
                    default=lambda value: value.item(),
                )
                + "\n"
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_downstream(output, artifact_dir):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results = output["results"]
    samples = output.get("samples", {})
    directions = output["higher_is_better"]
    groups = output.get("groups", {})
    group_subtasks = output.get("group_subtasks", {})

    scored_groups = {
        group
        for group, metrics in groups.items()
        if _reported_metrics(metrics, directions.get(group, {}))
    }
    selected_groups = {
        group
        for group in scored_groups
        if not any(
            group in _descendants(parent, group_subtasks)
            for parent in scored_groups
            if parent != group
        )
    }
    suppressed_leaves = {
        leaf for group in selected_groups for leaf in _descendant_leaves(group, group_subtasks)
    }

    downstream = []
    for task in sorted(selected_groups):
        reported = _reported_metrics(groups[task], directions.get(task, {}))
        leaf_samples = [
            (leaf, sample)
            for leaf in _descendant_leaves(task, group_subtasks)
            for sample in samples.get(leaf, [])
        ]
        artifact = artifact_dir / f"{task}.jsonl"
        digest = _write_artifact(
            artifact, leaf_samples, reported, prefix_doc_id=True
        )
        for metric, (value, higher) in sorted(reported.items()):
            downstream.append(
                {
                    "task": task,
                    "metric": metric,
                    "higher_is_better": higher,
                    "value": value,
                    "per_example_artifact_sha256": digest,
                }
            )

    for task in sorted(samples):
        if task in suppressed_leaves:
            continue
        reported = _reported_metrics(results.get(task, {}), directions.get(task, {}))
        if not reported:
            continue
        artifact = artifact_dir / f"{task}.jsonl"
        digest = _write_artifact(
            artifact,
            [(task, sample) for sample in samples[task]],
            reported,
        )
        for metric, (value, higher) in sorted(reported.items()):
            downstream.append(
                {
                    "task": task,
                    "metric": metric,
                    "higher_is_better": higher,
                    "value": value,
                    "per_example_artifact_sha256": digest,
                }
            )

    downstream.sort(key=lambda item: (item["task"], item["metric"]))
    (artifact_dir / "downstream.json").write_text(
        json.dumps(downstream, indent=1, sort_keys=True)
    )
    return downstream
