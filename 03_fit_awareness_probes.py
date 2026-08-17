#!/usr/bin/env python3
"""Select a conflict-sensitive residual site and fit factorized linear probes.

Layer selection uses only the country-capital validation split.  The held-out
fact, held-out paraphrase, and held-out relation splits are evaluated exactly
once after selection.  The primary awareness label is the preregistered
factorial label (false AND query-relevant), never the model's answer choice.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from common import (
    file_sha256,
    json_sha256,
    load_config,
    read_json,
    read_jsonl,
    unique_by,
    write_json_atomic,
    write_jsonl_atomic,
)


ANALYSIS_VERSION = "1.0.0"
TARGET_FIELDS = {
    "awareness": "effective_query_conflict",
    "falsehood": "claim_conflict_label",
    "relevance": "claim_is_query_relevant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--behavior", type=Path, default=None)
    parser.add_argument("--activation-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-relation", default="country_capital")
    parser.add_argument("--transfer-relation", default="element_symbol")
    parser.add_argument("--train-template", default="development")
    parser.add_argument("--validation-template", default="validation")
    parser.add_argument("--test-template", default="development")
    parser.add_argument("--paraphrase-template", default="heldout_paraphrase")
    parser.add_argument(
        "--selection-position",
        default="assistant_start",
        help="Position whose layer is selected for probing and intervention.",
    )
    parser.add_argument(
        "--c-grid",
        default="0.001,0.01,0.1,1,10",
        help="Comma-separated inverse L2 regularization strengths.",
    )
    parser.add_argument(
        "--scan-shrinkage",
        type=float,
        default=0.10,
        help="Shrinkage toward each layer-position cell's mean diagonal variance.",
    )
    parser.add_argument("--layer-chunk-size", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_paths(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[Path, Path, Path]:
    root = Path(config["paths"]["output_dir"])
    behavior = args.behavior or root / str(config["collection"]["behavior_results_file"])
    activation_manifest = args.activation_manifest or (
        root
        / str(config["collection"]["activations"]["directory"])
        / str(config["collection"]["activations"]["manifest_file"])
    )
    output = args.output_dir or root / "analysis" / "probes"
    return behavior.resolve(), activation_manifest.resolve(), output.resolve()


def parse_c_grid(text: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--c-grid must contain comma-separated numbers") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError("Every --c-grid value must be positive")
    if len(values) != len(set(values)):
        raise ValueError("--c-grid values must be unique")
    return values


def assign_analysis_splits(
    rows: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> list[str]:
    splits: list[str] = []
    for row in rows:
        relation = str(row["relation_id"])
        fact_split = str(row["fact_split"])
        template = str(row["template_bundle_id"])
        if (
            relation == args.train_relation
            and fact_split == "train"
            and template == args.train_template
        ):
            split = "train"
        elif (
            relation == args.train_relation
            and fact_split == "validation"
            and template == args.validation_template
        ):
            split = "validation"
        elif (
            relation == args.train_relation
            and fact_split == "test"
            and template == args.test_template
        ):
            split = "id_test"
        elif (
            relation == args.train_relation
            and fact_split == "test"
            and template == args.paraphrase_template
        ):
            split = "paraphrase_test"
        elif relation == args.transfer_relation and template == args.test_template:
            split = "ood_relation"
        elif relation == args.transfer_relation and template == args.paraphrase_template:
            split = "ood_relation_paraphrase"
        else:
            split = "excluded"
        splits.append(split)
    return splits


def validate_split_design(
    rows: Sequence[Mapping[str, Any]], splits: Sequence[str]
) -> dict[str, Any]:
    import numpy as np

    if len(rows) != len(splits):
        raise ValueError("Rows and split labels differ in length")
    counts = Counter(splits)
    required = {
        "train",
        "validation",
        "id_test",
        "paraphrase_test",
        "ood_relation",
        "ood_relation_paraphrase",
    }
    missing = sorted(name for name in required if counts[name] == 0)
    if missing:
        raise ValueError(f"The predefined analysis splits are empty: {missing}")

    split_array = np.asarray(splits)
    labels = np.asarray([bool(row["effective_query_conflict"]) for row in rows])
    for name in required:
        values = labels[split_array == name]
        if len(np.unique(values)) != 2:
            raise ValueError(f"Split {name} does not contain both awareness classes")

    # The development relation's base partitions must be disjoint even when
    # paragraph facts and false-answer sources are considered.  ID and
    # paraphrase tests intentionally reuse the same held-out facts.
    referenced: dict[str, set[str]] = {}
    for name, members in {
        "train": {"train"},
        "validation": {"validation"},
        "test": {"id_test", "paraphrase_test"},
    }.items():
        referenced[name] = {
            str(row[field])
            for row, split in zip(rows, splits)
            if split in members
            for field in ("fact_id", "claim_fact_id")
        }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = referenced[left].intersection(referenced[right])
        if overlap:
            raise ValueError(
                f"Fact leakage between {left} and {right}: {sorted(overlap)[:5]}"
            )

    return {
        "record_counts": dict(sorted(counts.items())),
        "unique_fact_counts": {
            name: len(
                {
                    str(row["fact_id"])
                    for row, split in zip(rows, splits)
                    if split == name
                }
            )
            for name in sorted(required)
        },
        "class_counts": {
            name: {
                "nonconflict": int(np.sum(labels[split_array == name] == 0)),
                "conflict": int(np.sum(labels[split_array == name] == 1)),
            }
            for name in sorted(required)
        },
    }


def validate_activation_manifest(
    manifest_path: Path,
    behavior_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Activation manifest must be a JSON object")
    expected = len(behavior_rows)
    if not bool(manifest.get("complete")):
        raise RuntimeError("Activation collection is incomplete")
    if int(manifest.get("expected_samples", -1)) != expected:
        raise ValueError("Activation manifest and behavior row counts disagree")
    if int(manifest.get("completed_samples", -1)) != expected:
        raise ValueError("Activation manifest completed_samples is inconsistent")
    ordered_ids = [str(row["sample_id"]) for row in behavior_rows]
    if manifest.get("ordered_sample_ids_sha256") != json_sha256(ordered_ids):
        raise ValueError("Activation and behavior sample ordering disagree")

    layer_indices = manifest.get("layer_indices")
    position_names = manifest.get("position_names")
    if not isinstance(layer_indices, list) or not layer_indices:
        raise ValueError("Activation manifest has no layer indices")
    if not isinstance(position_names, list) or not position_names:
        raise ValueError("Activation manifest has no positions")

    entries: list[dict[str, Any]] = []
    cursor = 0
    for raw in manifest.get("shards", []):
        entry = dict(raw)
        start, end = int(entry["start_index"]), int(entry["end_index"])
        if start != cursor or not start < end <= expected:
            raise ValueError("Activation shards are not contiguous")
        path = (manifest_path.parent / str(entry["file"])).resolve()
        if path.parent != manifest_path.parent.resolve() or not path.is_file():
            raise FileNotFoundError(f"Missing or unsafe activation shard: {path}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Activation shard size changed: {path}")
        expected_ids = ordered_ids[start:end]
        if entry.get("sample_ids_sha256") != json_sha256(expected_ids):
            raise ValueError(f"Activation shard ID signature disagrees: {path}")
        entry["path"] = path
        entries.append(entry)
        cursor = end
    if cursor != expected:
        raise ValueError("Activation shards do not cover every behavior row")
    return manifest, entries


def load_shard(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:  # mmap is absent in older supported PyTorch builds.
        return torch.load(path, map_location="cpu", weights_only=True)


def iter_shards(
    entries: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    behavior_rows: Sequence[Mapping[str, Any]],
) -> Iterable[tuple[int, int, Any]]:
    expected_layers = [int(value) for value in manifest["layer_indices"]]
    expected_positions = [str(value) for value in manifest["position_names"]]
    for entry in entries:
        start, end = int(entry["start_index"]), int(entry["end_index"])
        payload = load_shard(Path(entry["path"]))
        ids = [str(value) for value in payload.get("sample_ids", [])]
        expected_ids = [str(row["sample_id"]) for row in behavior_rows[start:end]]
        if ids != expected_ids:
            raise ValueError(f"Sample IDs inside {entry['path']} are misordered")
        if [int(value) for value in payload.get("layer_indices", [])] != expected_layers:
            raise ValueError(f"Layer indices inside {entry['path']} disagree")
        if [str(value) for value in payload.get("position_names", [])] != expected_positions:
            raise ValueError(f"Position names inside {entry['path']} disagree")
        activations = payload.get("activations")
        if activations is None or activations.ndim != 4:
            raise ValueError(f"Invalid activation tensor inside {entry['path']}")
        if tuple(activations.shape[:3]) != (
            end - start,
            len(expected_layers),
            len(expected_positions),
        ):
            raise ValueError(f"Activation tensor shape disagrees in {entry['path']}")
        yield start, end, activations
        del payload, activations


def estimate_scan_direction(
    entries: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    split_labels: Sequence[str],
    shrinkage: float,
    layer_chunk_size: int,
) -> Any:
    import numpy as np

    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("--scan-shrinkage must lie in [0, 1]")
    split_array = np.asarray(split_labels)
    y = np.asarray([bool(row["effective_query_conflict"]) for row in rows], dtype=np.int8)
    train = split_array == "train"
    counts = np.asarray([np.sum(train & (y == value)) for value in (0, 1)], dtype=np.int64)
    if np.any(counts < 2):
        raise ValueError("At least two training rows per awareness class are required")

    layer_count = len(manifest["layer_indices"])
    position_count = len(manifest["position_names"])
    hidden_size: int | None = None
    sums: Any = None
    sumsq: Any = None

    for start, end, tensor in iter_shards(entries, manifest, rows):
        if hidden_size is None:
            hidden_size = int(tensor.shape[-1])
            sums = np.zeros((2, layer_count, position_count, hidden_size), dtype=np.float64)
            sumsq = np.zeros_like(sums)
        local_train = train[start:end]
        local_y = y[start:end]
        for layer_start in range(0, layer_count, layer_chunk_size):
            layer_end = min(layer_start + layer_chunk_size, layer_count)
            for value in (0, 1):
                mask = local_train & (local_y == value)
                if not np.any(mask):
                    continue
                block = tensor[mask, layer_start:layer_end].float().numpy()
                sums[value, layer_start:layer_end] += np.sum(block, axis=0, dtype=np.float64)
                sumsq[value, layer_start:layer_end] += np.sum(
                    np.square(block, dtype=np.float32), axis=0, dtype=np.float64
                )
                del block

    if hidden_size is None:
        raise RuntimeError("No activation shards were read")
    means = sums / counts[:, None, None, None]
    centered_ss = sumsq - np.square(sums) / counts[:, None, None, None]
    pooled_var = np.sum(centered_ss, axis=0) / float(np.sum(counts) - 2)
    pooled_var = np.maximum(pooled_var, 0.0)
    target_var = np.mean(pooled_var, axis=-1, keepdims=True)
    denominator = (1.0 - shrinkage) * pooled_var + shrinkage * target_var
    denominator = np.maximum(denominator, np.maximum(target_var * 1e-6, 1e-12))
    direction = (means[1] - means[0]) / denominator
    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise RuntimeError("A scan direction is zero or non-finite")
    direction = (direction / norms).astype(np.float32)
    return direction


def score_scan(
    entries: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    direction: Any,
    layer_chunk_size: int,
) -> Any:
    import numpy as np

    scores = np.empty(
        (len(rows), len(manifest["layer_indices"]), len(manifest["position_names"])),
        dtype=np.float32,
    )
    for start, end, tensor in iter_shards(entries, manifest, rows):
        for layer_start in range(0, scores.shape[1], layer_chunk_size):
            layer_end = min(layer_start + layer_chunk_size, scores.shape[1])
            block = tensor[:, layer_start:layer_end].float().numpy()
            scores[start:end, layer_start:layer_end] = np.einsum(
                "nlph,lph->nlp",
                block,
                direction[layer_start:layer_end],
                optimize=True,
            )
            del block
    return scores


def safe_auc(y_true: Any, score: Any) -> float | None:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(score)
    if len(np.unique(y_true[valid])) != 2:
        return None
    return float(roc_auc_score(y_true[valid], score[valid]))


def robust_awareness_metrics(
    rows: Sequence[Mapping[str, Any]], indices: Any, scores: Any
) -> dict[str, float | None]:
    import numpy as np

    indices = np.asarray(indices, dtype=np.int64)
    y = np.asarray([bool(rows[index]["effective_query_conflict"]) for index in indices])
    relevant = np.asarray(
        [bool(rows[index]["claim_is_query_relevant"]) for index in indices]
    )
    false = np.asarray([bool(rows[index]["claim_conflict_label"]) for index in indices])
    policies = np.asarray([str(rows[index]["policy_id"]) for index in indices])
    scores = np.asarray(scores)
    overall = safe_auc(y, scores)
    within_relevant = safe_auc(y[relevant], scores[relevant])
    within_false = safe_auc(y[false], scores[false])
    policy_aucs = {
        f"auc_policy_{policy}": safe_auc(y[policies == policy], scores[policies == policy])
        for policy in ("neutral", "context", "parametric")
    }
    components = [overall, within_relevant, within_false, *policy_aucs.values()]
    robust = min(value for value in components if value is not None) if all(
        value is not None for value in components
    ) else None
    return {
        "auc": overall,
        "auc_within_relevant": within_relevant,
        "auc_within_false": within_false,
        **policy_aucs,
        "robust_auc": robust,
    }


def build_scan_table(
    rows: Sequence[Mapping[str, Any]],
    split_labels: Sequence[str],
    scores: Any,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np

    split_array = np.asarray(split_labels)
    train_indices = np.flatnonzero(split_array == "train")
    validation_indices = np.flatnonzero(split_array == "validation")
    table: list[dict[str, Any]] = []
    for layer_offset, layer_index in enumerate(manifest["layer_indices"]):
        for position_offset, position in enumerate(manifest["position_names"]):
            train_metrics = robust_awareness_metrics(
                rows, train_indices, scores[train_indices, layer_offset, position_offset]
            )
            validation_metrics = robust_awareness_metrics(
                rows,
                validation_indices,
                scores[validation_indices, layer_offset, position_offset],
            )
            table.append(
                {
                    "layer_index": int(layer_index),
                    "layer_offset": layer_offset,
                    "position": str(position),
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{
                        f"validation_{key}": value
                        for key, value in validation_metrics.items()
                    },
                }
            )
    return table


def select_cell(
    table: Sequence[Mapping[str, Any]], position: str
) -> dict[str, Any]:
    candidates = [row for row in table if row["position"] == position]
    if not candidates:
        raise ValueError(f"Selection position {position!r} was not collected")
    if any(row["validation_robust_auc"] is None for row in candidates):
        raise ValueError("A validation robust AUC is undefined")
    # Deterministic tie-breaking: robust AUC, overall AUC, then earlier layer.
    return dict(
        max(
            candidates,
            key=lambda row: (
                float(row["validation_robust_auc"]),
                float(row["validation_auc"]),
                -int(row["layer_index"]),
            ),
        )
    )


def extract_cell(
    entries: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    layer_offset: int,
    position_offset: int,
) -> Any:
    import numpy as np

    matrix: Any = None
    for start, end, tensor in iter_shards(entries, manifest, rows):
        block = tensor[:, layer_offset, position_offset, :].float().numpy()
        if matrix is None:
            matrix = np.empty((len(rows), block.shape[1]), dtype=np.float32)
        matrix[start:end] = block
    if matrix is None:
        raise RuntimeError("No selected-cell activations were loaded")
    return matrix


def fit_target(
    name: str,
    x_scaled: Any,
    y: Any,
    train_indices: Any,
    validation_indices: Any,
    rows: Sequence[Mapping[str, Any]],
    c_grid: Sequence[float],
    max_iter: int,
    seed: int,
) -> tuple[Any, float, list[dict[str, Any]]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    dual = len(train_indices) < x_scaled.shape[1]
    trials: list[dict[str, Any]] = []
    best: tuple[float, float, Any] | None = None
    for c_value in c_grid:
        model = LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            dual=dual,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=seed,
        )
        model.fit(x_scaled[train_indices], y[train_indices])
        decision = model.decision_function(x_scaled[validation_indices])
        if name == "awareness":
            components = robust_awareness_metrics(rows, validation_indices, decision)
            criterion = components["robust_auc"]
        else:
            components = {"auc": safe_auc(y[validation_indices], decision)}
            criterion = components["auc"]
        if criterion is None:
            raise RuntimeError(f"Validation metric is undefined for {name}")
        trials.append({"C": float(c_value), **components})
        candidate = (float(criterion), -float(c_value), model)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return best[2], float(-best[1]), trials


def classification_metrics(y_true: Any, probability: Any) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import average_precision_score, balanced_accuracy_score

    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    auc = safe_auc(y_true, probability)
    if len(np.unique(y_true)) < 2:
        average_precision = None
    else:
        average_precision = float(average_precision_score(y_true, probability))
    return {
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "roc_auc": auc,
        "average_precision": average_precision,
        "balanced_accuracy_at_0.5": float(
            balanced_accuracy_score(y_true, probability >= 0.5)
        ),
    }


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fieldnames = list(rows[0].keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    import numpy as np
    from scipy.special import expit
    from sklearn.preprocessing import StandardScaler

    started = time.time()
    args = parse_args()
    if args.layer_chunk_size < 1 or args.max_iter < 1:
        raise ValueError("--layer-chunk-size and --max-iter must be positive")
    c_grid = parse_c_grid(args.c_grid)
    config = load_config(args.config)
    behavior_path, activation_manifest_path, output_dir = resolve_paths(args, config)
    for path in (behavior_path, activation_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_paths = {
        "scan": output_dir / "probe_scan.csv",
        "model": output_dir / "probe_model.npz",
        "metadata": output_dir / "probe_metadata.json",
        "scores": output_dir / "probe_scores.jsonl",
        "evaluations": output_dir / "probe_evaluations.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Probe outputs already exist; pass --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(behavior_path)
    unique_by(rows, "sample_id", str(behavior_path))
    for row in rows:
        if row.get("record_type") != "gemma4_conflict_behavior":
            raise ValueError(f"Unexpected behavior record type: {row.get('sample_id')}")
        if bool(row["effective_query_conflict"]) != (
            bool(row["claim_conflict_label"])
            and bool(row["claim_is_query_relevant"])
        ):
            raise ValueError(f"Inconsistent awareness label: {row['sample_id']}")

    split_labels = assign_analysis_splits(rows, args)
    split_summary = validate_split_design(rows, split_labels)
    manifest, entries = validate_activation_manifest(
        activation_manifest_path, rows
    )
    if args.selection_position not in manifest["position_names"]:
        raise ValueError(
            f"{args.selection_position!r} not in activation positions "
            f"{manifest['position_names']}"
        )

    print("Estimating training-only diagonal scan directions...")
    scan_direction = estimate_scan_direction(
        entries,
        manifest,
        rows,
        split_labels,
        args.scan_shrinkage,
        args.layer_chunk_size,
    )
    print("Scoring every collected layer and position...")
    scan_scores = score_scan(
        entries, manifest, rows, scan_direction, args.layer_chunk_size
    )
    scan_table = build_scan_table(rows, split_labels, scan_scores, manifest)
    selected = select_cell(scan_table, args.selection_position)
    print(
        "Selected layer "
        f"{selected['layer_index']} at {selected['position']} "
        f"(validation robust AUC={selected['validation_robust_auc']:.4f})"
    )

    layer_offset = int(selected["layer_offset"])
    position_offset = list(manifest["position_names"]).index(args.selection_position)
    x = extract_cell(entries, manifest, rows, layer_offset, position_offset)
    split_array = np.asarray(split_labels)
    train_indices = np.flatnonzero(split_array == "train")
    validation_indices = np.flatnonzero(split_array == "validation")

    scaler = StandardScaler(copy=True)
    scaler.fit(x[train_indices])
    x_scaled = scaler.transform(x).astype(np.float32, copy=False)
    target_labels = {
        name: np.asarray([bool(row[field]) for row in rows], dtype=np.int8)
        for name, field in TARGET_FIELDS.items()
    }

    probabilities: dict[str, Any] = {}
    logits: dict[str, Any] = {}
    model_metadata: dict[str, Any] = {}
    model_arrays: dict[str, Any] = {
        "format_version": np.asarray([1], dtype=np.int32),
        "analysis_version": np.asarray([ANALYSIS_VERSION]),
        "selected_layer_index": np.asarray([int(selected["layer_index"])], dtype=np.int32),
        "selected_layer_offset": np.asarray([layer_offset], dtype=np.int32),
        "selected_position_index": np.asarray([position_offset], dtype=np.int32),
        "selected_position_name": np.asarray([args.selection_position]),
        "layer_indices": np.asarray(manifest["layer_indices"], dtype=np.int32),
        "position_names": np.asarray(manifest["position_names"]),
        "feature_mean": scaler.mean_.astype(np.float32),
        "feature_scale": scaler.scale_.astype(np.float32),
        "scan_direction_selected": scan_direction[layer_offset, position_offset],
    }
    for target_index, (name, y) in enumerate(target_labels.items()):
        model, selected_c, trials = fit_target(
            name,
            x_scaled,
            y,
            train_indices,
            validation_indices,
            rows,
            c_grid,
            args.max_iter,
            int(config["project"]["seed"]) + target_index,
        )
        decision = np.asarray(model.decision_function(x_scaled), dtype=np.float64)
        probability = expit(decision)
        coef_scaled = np.asarray(model.coef_[0], dtype=np.float64)
        intercept_scaled = float(model.intercept_[0])
        coef_raw = coef_scaled / scaler.scale_
        intercept_raw = intercept_scaled - float(np.dot(coef_raw, scaler.mean_))
        logits[name] = decision
        probabilities[name] = probability
        model_arrays.update(
            {
                f"{name}_coef_scaled": coef_scaled.astype(np.float32),
                f"{name}_intercept_scaled": np.asarray([intercept_scaled], dtype=np.float64),
                f"{name}_coef_raw": coef_raw.astype(np.float32),
                f"{name}_intercept_raw": np.asarray([intercept_raw], dtype=np.float64),
                f"{name}_C": np.asarray([selected_c], dtype=np.float64),
            }
        )
        converged = int(model.n_iter_[0]) < args.max_iter
        if not converged:
            print(
                f"WARNING: {name} probe reached max_iter={args.max_iter}; "
                "rerun with a larger --max-iter before interpreting it."
            )
        model_metadata[name] = {
            "label_field": TARGET_FIELDS[name],
            "selected_C": selected_c,
            "validation_trials": trials,
            "n_iter": int(model.n_iter_[0]),
            "converged_before_max_iter": converged,
            "class_weight": "balanced",
            "penalty": "l2",
            "solver": "liblinear",
            "dual": bool(len(train_indices) < x_scaled.shape[1]),
        }

    factorized_probability = probabilities["falsehood"] * probabilities["relevance"]
    evaluations: dict[str, Any] = {}
    for split_name in (
        "train",
        "validation",
        "id_test",
        "paraphrase_test",
        "ood_relation",
        "ood_relation_paraphrase",
    ):
        indices = np.flatnonzero(split_array == split_name)
        target_metrics = {
            name: classification_metrics(y[indices], probabilities[name][indices])
            for name, y in target_labels.items()
        }
        awareness_robust = robust_awareness_metrics(
            rows, indices, probabilities["awareness"][indices]
        )
        factorized_metrics = classification_metrics(
            target_labels["awareness"][indices], factorized_probability[indices]
        )
        evaluations[split_name] = {
            "selection_role": (
                "fit" if split_name == "train" else
                "model_selection" if split_name == "validation" else
                "held_out_evaluation"
            ),
            "targets": target_metrics,
            "awareness_robust": awareness_robust,
            "factorized_baseline": factorized_metrics,
            "awareness_minus_factorized_roc_auc": (
                target_metrics["awareness"]["roc_auc"]
                - factorized_metrics["roc_auc"]
                if target_metrics["awareness"]["roc_auc"] is not None
                and factorized_metrics["roc_auc"] is not None
                else None
            ),
        }

    score_rows: list[dict[str, Any]] = []
    for index, (row, split_name) in enumerate(zip(rows, split_labels)):
        score_rows.append(
            {
                "sample_id": row["sample_id"],
                "fact_id": row["fact_id"],
                "claim_fact_id": row["claim_fact_id"],
                "matched_factorial_group_id": row["matched_factorial_group_id"],
                "content_pair_id": row["content_pair_id"],
                "relation_id": row["relation_id"],
                "fact_split": row["fact_split"],
                "template_bundle_id": row["template_bundle_id"],
                "condition_id": row["condition_id"],
                "policy_id": row["policy_id"],
                "analysis_split": split_name,
                "awareness_label": bool(target_labels["awareness"][index]),
                "falsehood_label": bool(target_labels["falsehood"][index]),
                "relevance_label": bool(target_labels["relevance"][index]),
                "awareness_probe_logit": float(logits["awareness"][index]),
                "awareness_probe_probability": float(probabilities["awareness"][index]),
                "falsehood_probe_logit": float(logits["falsehood"][index]),
                "falsehood_probe_probability": float(probabilities["falsehood"][index]),
                "relevance_probe_logit": float(logits["relevance"][index]),
                "relevance_probe_probability": float(probabilities["relevance"][index]),
                "factorized_awareness_probability": float(factorized_probability[index]),
            }
        )

    metadata = {
        "analysis_version": ANALYSIS_VERSION,
        "script": Path(__file__).name,
        "elapsed_seconds": time.time() - started,
        "inputs": {
            "behavior": str(behavior_path),
            "behavior_sha256": file_sha256(behavior_path),
            "activation_manifest": str(activation_manifest_path),
            "activation_manifest_sha256": file_sha256(activation_manifest_path),
            "activation_input_signature": manifest.get("input_signature"),
        },
        "split_definition": {
            "train": f"{args.train_relation}/train/{args.train_template}",
            "validation": f"{args.train_relation}/validation/{args.validation_template}",
            "id_test": f"{args.train_relation}/test/{args.test_template}",
            "paraphrase_test": f"{args.train_relation}/test/{args.paraphrase_template}",
            "ood_relation": f"{args.transfer_relation}/*/{args.test_template}",
            "ood_relation_paraphrase": (
                f"{args.transfer_relation}/*/{args.paraphrase_template}"
            ),
        },
        "split_summary": split_summary,
        "selection": {
            "rule": (
                "maximize on validation the minimum of overall awareness AUC, "
                "AUC within relevant claims, AUC within false claims, and AUC "
                "within each answer-source policy; "
                "selection restricted to the preregistered position"
            ),
            "position": args.selection_position,
            "selected_cell": selected,
            "scan_estimator": "training-only diagonal shrinkage LDA direction",
            "scan_shrinkage": args.scan_shrinkage,
            "test_metrics_used_for_selection": False,
        },
        "activation": {
            "definition": manifest.get("activation_definition"),
            "hidden_size": int(x.shape[1]),
            "storage_dtype": manifest.get("storage_dtype"),
        },
        "models": model_metadata,
        "factorized_baseline": (
            "falsehood_probe_probability * relevance_probe_probability"
        ),
        "packages": {
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "scikit-learn": package_version("scikit-learn"),
            "torch": package_version("torch"),
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }

    write_csv_atomic(output_paths["scan"], scan_table)
    save_npz_atomic(output_paths["model"], **model_arrays)
    write_jsonl_atomic(output_paths["scores"], score_rows)
    write_json_atomic(output_paths["evaluations"], evaluations)
    write_json_atomic(output_paths["metadata"], metadata)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
