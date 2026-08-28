#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

import fsspec
import numpy as np
import pyarrow.parquet as pq


REPO = "HuggingFaceFW/fineweb_edu_100BT-shuffled"
REVISION = "be6b2a50d3a9c60d330c45384e80c7863cd3a25d"
NUMERIC_COLUMNS = ("token_count", "score", "int_score", "language_score")


def numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {
            str(q): float(value)
            for q, value in zip(
                (0.01, 0.1, 0.5, 0.9, 0.99),
                np.quantile(values, (0.01, 0.1, 0.5, 0.9, 0.99)),
            )
        },
    }


def read_shard(shard, limit):
    path = (
        f"hf://datasets/{REPO}@{REVISION}/data/"
        f"train-{shard:05d}-of-00100.parquet"
    )
    columns = [*NUMERIC_COLUMNS, "dump"]
    values = {name: [] for name in NUMERIC_COLUMNS}
    dumps = []
    with fsspec.open(path, "rb").open() as stream:
        parquet = pq.ParquetFile(stream)
        for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
            take = min(len(batch), limit - len(dumps))
            if take <= 0:
                break
            table = batch.slice(0, take)
            for name in NUMERIC_COLUMNS:
                values[name].append(table.column(name).to_numpy(zero_copy_only=False))
            dumps.extend(table.column("dump").to_pylist())
            if len(dumps) == limit:
                break
    if len(dumps) != limit:
        raise RuntimeError(f"shard {shard} yielded {len(dumps)} rows, expected {limit}")
    values = {name: np.concatenate(parts) for name, parts in values.items()}
    dump_counts = Counter(dumps)
    expected_adjacent_dump_rate = sum(
        (count / len(dumps)) ** 2 for count in dump_counts.values()
    )
    observed_adjacent_dump_rate = float(
        np.mean(np.asarray(dumps[1:], dtype=object) == np.asarray(dumps[:-1], dtype=object))
    )
    return {
        "shard": shard,
        "rows": len(dumps),
        "numeric_values": values,
        "dump_counts": dump_counts,
        "summary": {
            "numeric": {name: numeric_summary(value) for name, value in values.items()},
            "dump_classes": len(dump_counts),
            "adjacent_dump_rate": observed_adjacent_dump_rate,
            "expected_adjacent_dump_rate": expected_adjacent_dump_rate,
            "adjacent_dump_excess": observed_adjacent_dump_rate
            - expected_adjacent_dump_rate,
            "adjacent_token_count_correlation": float(
                np.corrcoef(values["token_count"][:-1], values["token_count"][1:])[0, 1]
            ),
        },
    }


def group_summary(rows):
    numeric = {
        name: np.concatenate([row["numeric_values"][name] for row in rows])
        for name in NUMERIC_COLUMNS
    }
    dump_counts = sum((row["dump_counts"] for row in rows), Counter())
    total = sum(dump_counts.values())
    return {
        "rows": total,
        "shards": [row["shard"] for row in rows],
        "numeric": {name: numeric_summary(value) for name, value in numeric.items()},
        "dump_distribution": {
            name: count / total for name, count in sorted(dump_counts.items())
        },
        "per_shard": [row["summary"] | {"shard": row["shard"]} for row in rows],
    }


def comparison(base, extension):
    numeric = {}
    for name in NUMERIC_COLUMNS:
        base_value = base["numeric"][name]
        extension_value = extension["numeric"][name]
        pooled_std = (
            (base_value["std"] ** 2 + extension_value["std"] ** 2) / 2
        ) ** 0.5
        numeric[name] = {
            "extension_minus_base_mean": extension_value["mean"] - base_value["mean"],
            "standardized_mean_difference": (
                (extension_value["mean"] - base_value["mean"]) / pooled_std
                if pooled_std
                else 0.0
            ),
        }
    dumps = set(base["dump_distribution"]) | set(extension["dump_distribution"])
    dump_total_variation = 0.5 * sum(
        abs(
            base["dump_distribution"].get(name, 0.0)
            - extension["dump_distribution"].get(name, 0.0)
        )
        for name in dumps
    )
    return {"numeric": numeric, "dump_total_variation": dump_total_variation}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for shard in range(12):
        row = read_shard(shard, args.rows_per_shard)
        print(
            f"SOURCE_SHARD shard={shard} rows={row['rows']}"
            f" token_count_mean={row['summary']['numeric']['token_count']['mean']:.6f}"
            f" score_mean={row['summary']['numeric']['score']['mean']:.6f}"
            f" adjacent_token_count_correlation="
            f"{row['summary']['adjacent_token_count_correlation']:.8f}"
            f" adjacent_dump_excess={row['summary']['adjacent_dump_excess']:.8f}",
            flush=True,
        )
        rows.append(row)
    base = group_summary(rows[:8])
    extension = group_summary(rows[8:])
    result = comparison(base, extension)
    report = {
        "schema_version": 1,
        "source": {"repo": REPO, "revision": REVISION},
        "sampling": {
            "policy": "first rows of each already globally shuffled source shard",
            "rows_per_shard": args.rows_per_shard,
        },
        "base": base,
        "extension": extension,
        "comparison": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "SOURCE_GROUP_COMPARISON"
        f" dump_total_variation={result['dump_total_variation']:.8f}"
        + "".join(
            f" {name}_smd={value['standardized_mean_difference']:.8f}"
            for name, value in result["numeric"].items()
        )
    )
    print(f"REPORT={args.output}")
    print("SOURCE_SHARD_AUDIT=pass")


if __name__ == "__main__":
    main()
