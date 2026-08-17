#!/usr/bin/env python3
"""Test whether conflict awareness and answer resolution can be dissociated.

The intervention is added to the selected decoder-block output at the exact
``assistant_start`` token: the final token of Gemma's canonical generation
prefix, whose residual predicts the first answer token.  Exact candidate
sequence log-probabilities (EOS excluded) are the primary causal outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import (
    _prepare_continuation,
    _to_model_device,
    build_position_ids,
    file_sha256,
    get_decoder_layers,
    load_config,
    load_model_bundle,
    read_json,
    read_jsonl,
    render_dataset_record,
    runtime_fingerprint,
    seed_everything,
    unique_by,
    validate_manifest_file,
    write_json_atomic,
    write_jsonl_atomic,
)


ANALYSIS_VERSION = "1.0.1"
DEFAULT_SPLITS = ("id_test", "paraphrase_test", "ood_relation")
DEFAULT_DIRECTIONS = (
    "awareness_specific",
    "resolution_awareness_orthogonal",
    "random_orthogonal_01",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--behavior", type=Path, default=None)
    parser.add_argument("--experiment", type=Path, default=None)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--activation-manifest", type=Path, default=None)
    parser.add_argument("--probe-model", type=Path, default=None)
    parser.add_argument("--probe-scores", type=Path, default=None)
    parser.add_argument("--probe-metadata", type=Path, default=None)
    parser.add_argument("--collection-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--evaluation-splits", default=",".join(DEFAULT_SPLITS)
    )
    parser.add_argument(
        "--directions", default=",".join(DEFAULT_DIRECTIONS),
        help="Comma-separated direction IDs saved by this script.",
    )
    parser.add_argument(
        "--strengths",
        default="-2,-1,1,2",
        help="Signed intervention strengths in training projection SDs.",
    )
    parser.add_argument("--max-examples-per-split", type=int, default=24)
    parser.add_argument("--scoring-batch-size", type=int, default=None)
    parser.add_argument("--random-controls", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--baseline-tolerance",
        type=float,
        default=0.10,
        help="Maximum allowed absolute mismatch from step 02's cached margin.",
    )
    parser.add_argument(
        "--allow-baseline-mismatch",
        action="store_true",
        help="Warn rather than stop if causal baseline scoring is not reproduced.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Path]:
    root = Path(config["paths"]["output_dir"])
    dataset = Path(config["paths"]["dataset_dir"])
    activation_root = root / str(config["collection"]["activations"]["directory"])
    probe_root = root / "analysis" / "probes"
    return {
        "behavior": (args.behavior or root / str(config["collection"]["behavior_results_file"])).resolve(),
        "experiment": (args.experiment or dataset / "experiment.jsonl").resolve(),
        "dataset_manifest": (args.dataset_manifest or dataset / "manifest.json").resolve(),
        "activation_manifest": (
            args.activation_manifest
            or activation_root / str(config["collection"]["activations"]["manifest_file"])
        ).resolve(),
        "probe_model": (args.probe_model or probe_root / "probe_model.npz").resolve(),
        "probe_scores": (args.probe_scores or probe_root / "probe_scores.jsonl").resolve(),
        "probe_metadata": (args.probe_metadata or probe_root / "probe_metadata.json").resolve(),
        "collection_metadata": (
            args.collection_metadata or root / str(config["collection"]["run_metadata_file"])
        ).resolve(),
        "output_dir": (args.output_dir or root / "analysis" / "causal").resolve(),
    }


def parse_unique_list(text: str, argument: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{argument} must be a non-empty unique list")
    return values


def parse_strengths(text: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--strengths must contain comma-separated numbers") from exc
    if not values or any(value == 0 for value in values):
        raise ValueError("--strengths must be non-empty and exclude zero (baseline is automatic)")
    if len(values) != len(set(values)):
        raise ValueError("--strengths must be unique")
    if not any(value < 0 for value in values) or not any(value > 0 for value in values):
        raise ValueError("--strengths must contain both negative and positive values")
    return sorted(values)


def scalar(array: Any, key: str) -> Any:
    value = array[key]
    if value.size != 1:
        raise ValueError(f"Probe model field {key} is not scalar")
    return value.reshape(-1)[0].item()


def merge_behavior_and_scores(
    behavior: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    unique_by(behavior, "sample_id", "behavior results")
    unique_by(scores, "sample_id", "probe scores")
    score_by_id = {str(row["sample_id"]): row for row in scores}
    if {str(row["sample_id"]) for row in behavior} != set(score_by_id):
        raise ValueError("Behavior and probe-score sample sets differ")
    rows: list[dict[str, Any]] = []
    for behavior_row in behavior:
        score = score_by_id[str(behavior_row["sample_id"])]
        for field in (
            "fact_id",
            "matched_factorial_group_id",
            "content_pair_id",
            "condition_id",
            "policy_id",
        ):
            if str(behavior_row[field]) != str(score[field]):
                raise ValueError(f"Behavior/probe mismatch for {behavior_row['sample_id']}: {field}")
        row = dict(behavior_row)
        row.update(
            {
                key: value
                for key, value in score.items()
                if key == "analysis_split" or key.endswith(("_logit", "_probability"))
            }
        )
        rows.append(row)
    return rows


def validate_activation_manifest(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from common import json_sha256

    manifest = read_json(path)
    ids = [str(row["sample_id"]) for row in rows]
    if not manifest.get("complete") or int(manifest.get("completed_samples", -1)) != len(rows):
        raise RuntimeError("Activation manifest is not complete")
    if manifest.get("ordered_sample_ids_sha256") != json_sha256(ids):
        raise ValueError("Activation and behavior ordering disagree")
    entries: list[dict[str, Any]] = []
    cursor = 0
    for raw in manifest.get("shards", []):
        entry = dict(raw)
        start, end = int(entry["start_index"]), int(entry["end_index"])
        if start != cursor or not start < end <= len(rows):
            raise ValueError("Activation shards are not contiguous")
        shard_path = (path.parent / str(entry["file"])).resolve()
        if shard_path.parent != path.parent.resolve() or not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        if shard_path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Activation shard size changed: {shard_path}")
        entry["path"] = shard_path
        entries.append(entry)
        cursor = end
    if cursor != len(rows):
        raise ValueError("Activation shards do not cover all behavior rows")
    return manifest, entries


def load_shard(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def extract_selected_activations(
    entries: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    layer_offset: int,
    position_offset: int,
) -> Any:
    import numpy as np

    matrix: Any = None
    expected_layers = [int(value) for value in manifest["layer_indices"]]
    expected_positions = [str(value) for value in manifest["position_names"]]
    for entry in entries:
        start, end = int(entry["start_index"]), int(entry["end_index"])
        payload = load_shard(Path(entry["path"]))
        expected_ids = [str(row["sample_id"]) for row in rows[start:end]]
        if [str(value) for value in payload.get("sample_ids", [])] != expected_ids:
            raise ValueError(f"Misordered IDs in {entry['path']}")
        if [int(value) for value in payload.get("layer_indices", [])] != expected_layers:
            raise ValueError(f"Layer mismatch in {entry['path']}")
        if [str(value) for value in payload.get("position_names", [])] != expected_positions:
            raise ValueError(f"Position mismatch in {entry['path']}")
        tensor = payload["activations"]
        if tensor.ndim != 4 or tuple(tensor.shape[:3]) != (
            end - start,
            len(expected_layers),
            len(expected_positions),
        ):
            raise ValueError(f"Invalid activation shape in {entry['path']}")
        block = tensor[:, layer_offset, position_offset, :].float().numpy()
        if matrix is None:
            matrix = np.empty((len(rows), block.shape[1]), dtype=np.float32)
        matrix[start:end] = block
        del payload, tensor, block
    if matrix is None:
        raise RuntimeError("No activations were loaded")
    return matrix


def unit(vector: Any, name: str) -> Any:
    import numpy as np

    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"Direction {name} is zero or non-finite")
    return vector / norm


def orthonormal_basis(vectors: Sequence[Any]) -> list[Any]:
    import numpy as np

    basis: list[Any] = []
    for raw in vectors:
        value = np.asarray(raw, dtype=np.float64).copy()
        for existing in basis:
            value -= float(np.dot(value, existing)) * existing
        norm = float(np.linalg.norm(value))
        if norm > 1e-10:
            basis.append(value / norm)
    return basis


def residualize(vector: Any, nuisance: Sequence[Any], name: str) -> Any:
    import numpy as np

    value = np.asarray(vector, dtype=np.float64).copy()
    original_norm = float(np.linalg.norm(value))
    for basis in orthonormal_basis(nuisance):
        value -= float(np.dot(value, basis)) * basis
    if float(np.linalg.norm(value)) <= max(1e-12, original_norm * 1e-6):
        raise ValueError(f"Residualization removed essentially all of {name}")
    return unit(value, name)


def factorial_interaction_direction(
    x: Any, rows: Sequence[Mapping[str, Any]]
) -> tuple[Any, int]:
    import numpy as np

    groups: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        if row["analysis_split"] == "train" and row["policy_id"] == "neutral":
            group = str(row["matched_factorial_group_id"])
            condition = str(row["condition_id"])
            if condition in groups[group]:
                raise ValueError(f"Duplicate condition in training group {group}")
            groups[group][condition] = index
    required = {"false_relevant", "true_relevant", "false_irrelevant", "true_irrelevant"}
    contrasts = []
    for group, indices in groups.items():
        if set(indices) != required:
            raise ValueError(f"Incomplete neutral training factorial group: {group}")
        contrasts.append(
            x[indices["false_relevant"]]
            - x[indices["true_relevant"]]
            - x[indices["false_irrelevant"]]
            + x[indices["true_irrelevant"]]
        )
    if not contrasts:
        raise ValueError("No neutral training factorial groups are available")
    return np.mean(np.stack(contrasts), axis=0), len(contrasts)


def policy_resolution_direction(
    x: Any, rows: Sequence[Mapping[str, Any]]
) -> tuple[Any, int]:
    import numpy as np

    pairs: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        if row["analysis_split"] == "train" and row["condition_id"] == "false_relevant":
            content = str(row["content_pair_id"])
            policy = str(row["policy_id"])
            if policy in pairs[content]:
                raise ValueError(f"Duplicate training policy in content pair {content}")
            pairs[content][policy] = index
    differences = []
    for policies in pairs.values():
        if "context" in policies and "parametric" in policies:
            differences.append(x[policies["context"]] - x[policies["parametric"]])
    if not differences:
        raise ValueError("No context/parametric training policy pairs are available")
    return np.mean(np.stack(differences), axis=0), len(differences)


def construct_directions(
    x: Any,
    rows: Sequence[Mapping[str, Any]],
    probe: Any,
    random_controls: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    awareness_coef = unit(probe["awareness_coef_raw"], "awareness_probe")
    falsehood_coef = unit(probe["falsehood_coef_raw"], "falsehood_probe")
    relevance_coef = unit(probe["relevance_coef_raw"], "relevance_probe")
    awareness_mean, factorial_groups = factorial_interaction_direction(x, rows)
    resolution_mean, policy_pairs = policy_resolution_direction(x, rows)
    awareness_raw = unit(awareness_mean, "awareness_raw")
    resolution_raw = unit(resolution_mean, "resolution_raw")

    awareness_specific = residualize(
        awareness_raw,
        [resolution_raw, falsehood_coef, relevance_coef],
        "awareness_specific",
    )
    if float(np.dot(awareness_specific, awareness_coef)) < 0:
        awareness_specific = -awareness_specific
    resolution_orthogonal = residualize(
        resolution_raw,
        [awareness_coef],
        "resolution_awareness_orthogonal",
    )
    if float(np.dot(resolution_orthogonal, resolution_raw)) < 0:
        resolution_orthogonal = -resolution_orthogonal

    directions = {
        "awareness_raw": awareness_raw,
        "resolution_raw": resolution_raw,
        "awareness_specific": awareness_specific,
        "resolution_awareness_orthogonal": resolution_orthogonal,
    }
    random_basis = orthonormal_basis(
        [awareness_raw, resolution_raw, awareness_coef, falsehood_coef, relevance_coef]
    )
    rng = np.random.default_rng(seed)
    for index in range(1, random_controls + 1):
        random = rng.standard_normal(x.shape[1])
        random = residualize(
            random,
            random_basis + list(directions.values()),
            f"random_orthogonal_{index:02d}",
        )
        directions[f"random_orthogonal_{index:02d}"] = random
        random_basis.append(random)

    train_mask = np.asarray([row["analysis_split"] == "train" for row in rows])
    scales: dict[str, float] = {}
    for name, direction in directions.items():
        scale = float(np.std(x[train_mask] @ direction, ddof=1))
        if not np.isfinite(scale) or scale <= 1e-8:
            raise ValueError(f"Training projection scale is degenerate for {name}")
        scales[name] = scale

    def cosine(left: str, right: Any) -> float:
        return float(np.dot(directions[left], unit(right, "cosine_reference")))

    metadata = {
        "training_factorial_groups": factorial_groups,
        "training_policy_pairs": policy_pairs,
        "scale_definition": "sample SD of training activations projected on the unit direction",
        "directions": {
            name: {
                "unit_norm": float(np.linalg.norm(direction)),
                "training_projection_sd": scales[name],
                "cosine_with_awareness_probe": cosine(name, awareness_coef),
                "cosine_with_falsehood_probe": cosine(name, falsehood_coef),
                "cosine_with_relevance_probe": cosine(name, relevance_coef),
                "cosine_with_raw_resolution": cosine(name, resolution_raw),
                "cosine_with_raw_awareness_interaction": cosine(name, awareness_raw),
            }
            for name, direction in directions.items()
        },
        "definitions": {
            "awareness_raw": "mean training neutral-policy FR - TR - FI + TI residual contrast",
            "resolution_raw": "mean training false-relevant context-policy minus parametric-policy residual contrast",
            "awareness_specific": "awareness_raw residualized against resolution_raw and the falsehood/relevance probe directions",
            "resolution_awareness_orthogonal": "resolution_raw residualized against the awareness-probe direction",
            "random_orthogonal_*": "seeded Gaussian control residualized against the experimental direction/readout subspace",
        },
    }
    return directions, {"scales": scales, **metadata}


def stable_subset(
    rows: Sequence[Mapping[str, Any]],
    splits: Sequence[str],
    maximum: int,
    seed: int,
) -> list[int]:
    selected: list[int] = []
    for split in splits:
        candidates = [
            index
            for index, row in enumerate(rows)
            if row["analysis_split"] == split
            and row["condition_id"] == "false_relevant"
            and row["policy_id"] == "neutral"
            and row.get("context_candidate_answer") is not None
            and bool(row.get("claim_and_parametric_answers_are_distinct"))
        ]
        candidates.sort(
            key=lambda index: hashlib.sha256(
                f"{seed}|{rows[index]['sample_id']}".encode("utf-8")
            ).hexdigest()
        )
        chosen = candidates[:maximum]
        if not chosen:
            raise ValueError(f"No eligible false-relevant neutral examples in {split}")
        selected.extend(chosen)
    return selected


def replace_first_tensor(output: Any, replacement: Any) -> Any:
    import torch

    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        values = list(output)
        for index, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                values[index] = replacement
                return type(output)(*values) if hasattr(output, "_fields") else tuple(values)
    if isinstance(output, list):
        values = list(output)
        for index, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                values[index] = replacement
                return values
    raise TypeError(f"Unsupported decoder-block output: {type(output).__name__}")


class ResidualIntervention:
    """Add one vector per batch row at one post-block token position."""

    def __init__(self, layer: Any) -> None:
        self.positions: Any = None
        self.deltas: Any = None
        self.handle = layer.register_forward_hook(self._hook)

    def set_batch(self, positions: Any, deltas: Any) -> None:
        self.positions = positions
        self.deltas = deltas

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        if self.positions is None or self.deltas is None:
            raise RuntimeError("Intervention hook fired without batch state")
        hidden = output if isinstance(output, torch.Tensor) else next(
            value for value in output if isinstance(value, torch.Tensor)
        )
        if hidden.shape[0] != len(self.positions):
            raise ValueError("Intervention batch size does not match hidden states")
        positions = self.positions.to(hidden.device)
        deltas = self.deltas.to(device=hidden.device, dtype=hidden.dtype)
        modified = hidden.clone()
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        modified[batch, positions, :] = modified[batch, positions, :] + deltas
        return replace_first_tensor(output, modified)

    def close(self) -> None:
        self.handle.remove()
        self.positions = None
        self.deltas = None


def score_intervened_continuations(
    model: Any,
    tokenizer: Any,
    selected_layer: Any,
    requests: Sequence[Mapping[str, Any]],
    delta_lookup: Mapping[str, Any],
    batch_size: int,
    max_input_tokens: int,
) -> dict[str, dict[str, Any]]:
    import numpy as np
    import torch

    prepared: list[dict[str, Any]] = []
    for request in requests:
        prompt_ids, candidate_ids = _prepare_continuation(
            tokenizer,
            str(request["rendered_text"]),
            str(request["continuation"]),
        )
        if len(prompt_ids) + len(candidate_ids) > max_input_tokens:
            raise ValueError(f"Scored request exceeds max_input_tokens: {request['key']}")
        prepared.append(
            {
                **request,
                "prompt_ids": prompt_ids,
                "candidate_ids": candidate_ids,
                "combined_ids": prompt_ids + candidate_ids,
            }
        )

    scores: dict[str, dict[str, Any]] = {}
    intervention = ResidualIntervention(selected_layer)
    try:
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            width = max(len(item["combined_ids"]) for item in batch)
            input_ids = torch.full(
                (len(batch), width), int(tokenizer.pad_token_id), dtype=torch.long
            )
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
            candidate_starts: list[int] = []
            injection_positions: list[int] = []
            delta_rows: list[Any] = []
            for row_index, item in enumerate(batch):
                combined = torch.tensor(item["combined_ids"], dtype=torch.long)
                left_pad = width - len(combined)
                input_ids[row_index, left_pad:] = combined
                attention_mask[row_index, left_pad:] = 1
                candidate_starts.append(left_pad + len(item["prompt_ids"]))
                injection_positions.append(left_pad + len(item["prompt_ids"]) - 1)
                delta_rows.append(delta_lookup[str(item["delta_key"])])
            deltas = torch.from_numpy(np.stack(delta_rows).astype(np.float32, copy=False))
            intervention.set_batch(
                torch.tensor(injection_positions, dtype=torch.long), deltas
            )
            keep = max(len(item["candidate_ids"]) + 1 for item in batch)
            model_inputs = _to_model_device(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": build_position_ids(attention_mask),
                },
                model,
            )
            with torch.inference_mode():
                outputs = model(
                    **model_inputs,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=keep,
                )
            logits = outputs.logits
            logits_start = width - int(logits.shape[1])
            for row_index, item in enumerate(batch):
                token_logprobs: list[float] = []
                for offset, token_id in enumerate(item["candidate_ids"]):
                    absolute = candidate_starts[row_index] - 1 + offset
                    local = absolute - logits_start
                    if not 0 <= local < logits.shape[1]:
                        raise AssertionError("logits_to_keep omitted a candidate prediction")
                    token_logits = logits[row_index, local].float()
                    logprob = token_logits[int(token_id)] - torch.logsumexp(token_logits, dim=-1)
                    token_logprobs.append(float(logprob.item()))
                scores[str(item["key"])] = {
                    "sum_logprob": float(sum(token_logprobs)),
                    "num_tokens": len(token_logprobs),
                    "token_logprobs": token_logprobs,
                }
            del outputs, logits, model_inputs
    finally:
        intervention.close()
    return scores


def verify_runtime(
    collection: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    paths = (
        ("model", "id"),
        ("model", "requested_revision"),
        ("model", "loaded_commit_hash"),
        ("model", "class"),
        ("chat", "chat_template_sha256"),
        ("chat", "add_generation_prompt"),
        ("chat", "enable_thinking"),
        ("chat", "pad_token_id"),
        ("packages", "transformers"),
    )
    differences: list[str] = []
    for path in paths:
        old: Any = collection
        new: Any = current
        for key in path:
            old = old.get(key) if isinstance(old, Mapping) else None
            new = new.get(key) if isinstance(new, Mapping) else None
        if old != new:
            differences.append(f"{'.'.join(path)}: step02={old!r}, current={new!r}")
    if differences:
        raise RuntimeError("Runtime differs from step 02:\n  " + "\n  ".join(differences))


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    import numpy as np

    lower, upper = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(lower), float(upper)


def bootstrap_mean(
    values: Any, facts: Sequence[str], replicates: int, rng: Any
) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    facts_array = np.asarray(facts).astype(str)
    unique = np.asarray(sorted(set(facts_array)))
    indices = {fact: np.flatnonzero(facts_array == fact) for fact in unique}
    draws = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        chosen = np.concatenate([indices[str(fact)] for fact in sampled])
        draws.append(float(np.mean(values[chosen])))
    lower, upper = percentile_interval(draws)
    return {
        "estimate": float(np.mean(values)),
        "ci95_low": lower,
        "ci95_high": upper,
        "n": int(len(values)),
        "unique_facts": int(len(unique)),
    }


def summarize_causal_results(
    results: Sequence[Mapping[str, Any]],
    splits: Sequence[str],
    directions: Sequence[str],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    strength_rows: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []
    interventions = [row for row in results if row["direction_id"] != "baseline"]
    for split in splits:
        for direction in directions:
            group = [
                row for row in interventions
                if row["analysis_split"] == split and row["direction_id"] == direction
            ]
            for strength in sorted({float(row["strength_sd"]) for row in group}):
                subset = [row for row in group if float(row["strength_sd"]) == strength]
                facts = [str(row["fact_id"]) for row in subset]
                for outcome in (
                    "resolution_margin_delta",
                    "awareness_probe_logit_delta",
                ):
                    stats = bootstrap_mean(
                        [float(row[outcome]) for row in subset],
                        facts,
                        replicates,
                        rng,
                    )
                    strength_rows.append(
                        {
                            "analysis_split": split,
                            "direction_id": direction,
                            "strength_sd": strength,
                            "outcome": outcome,
                            **stats,
                        }
                    )

            by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in group:
                by_sample[str(row["sample_id"])].append(row)
            facts, margin_slopes, awareness_slopes = [], [], []
            for sample_rows in by_sample.values():
                strengths = np.asarray([float(row["strength_sd"]) for row in sample_rows])
                denominator = float(np.dot(strengths, strengths))
                if denominator <= 0:
                    raise ValueError("Cannot estimate a dose-response slope")
                margin_slopes.append(
                    float(
                        np.dot(
                            strengths,
                            [float(row["resolution_margin_delta"]) for row in sample_rows],
                        )
                        / denominator
                    )
                )
                awareness_slopes.append(
                    float(
                        np.dot(
                            strengths,
                            [float(row["awareness_probe_logit_delta"]) for row in sample_rows],
                        )
                        / denominator
                    )
                )
                facts.append(str(sample_rows[0]["fact_id"]))
            for outcome, values in (
                ("resolution_margin_per_1sd", margin_slopes),
                ("awareness_logit_per_1sd", awareness_slopes),
            ):
                slope_rows.append(
                    {
                        "analysis_split": split,
                        "direction_id": direction,
                        "outcome": outcome,
                        **bootstrap_mean(values, facts, replicates, rng),
                    }
                )
    return strength_rows, slope_rows


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Causal dissociation of conflict awareness and answer resolution",
        "",
        "Interventions were applied to the selected post-block residual at `assistant_start`. "
        "The resolution outcome is the change in `log P(context answer) − log P(parametric answer)`; "
        "positive values favor the contextual answer.",
        "",
        "## Baseline fidelity",
        "",
        f"Maximum absolute difference from step 02's cached margin: "
        f"**{fmt(summary['baseline_fidelity']['max_absolute_margin_difference'])}** "
        f"(tolerance {fmt(summary['baseline_fidelity']['tolerance'])}).",
        "",
        "## Dose-response slopes",
        "",
        "Each estimate is the mean per-example slope per one training projection SD. "
        "Intervals are query-fact cluster-bootstrap 95% intervals.",
        "",
        "| Split | Direction | Outcome | Slope (95% CI) | N |",
        "|---|---|---|---:|---:|",
    ]
    for row in summary["dose_response_slopes"]:
        lines.append(
            f"| {row['analysis_split']} | {row['direction_id']} | {row['outcome']} | "
            f"{fmt(row['estimate'])} [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {row['n']} |"
        )

    lines.extend(
        [
            "",
            "## How to interpret the pattern",
            "",
            "- First verify baseline fidelity and a roughly monotonic, sign-symmetric dose response. "
            "A one-sided jump at only the largest strength is more consistent with disruption.",
            "- The strongest dissociation is: `awareness_specific` reliably changes the awareness "
            "readout with little resolution-margin movement, while "
            "`resolution_awareness_orthogonal` reliably changes the margin. The latter direction "
            "is algebraically orthogonal to the fitted awareness readout, so its near-zero readout "
            "effect is a construction check—not independent evidence that every possible awareness "
            "representation is unchanged.",
            "- If both experimental directions move the answer margin similarly, awareness and "
            "resolution are entangled at this site (or the residualization basis is incomplete).",
            "- If neither direction moves the margin, the probe may be correlational, the selected "
            "site may be downstream or bypassed, or the tested strengths may be too weak.",
            "- Random-control effects comparable to the experimental effects indicate nonspecific "
            "residual-stream disruption. Do not interpret such a run mechanistically.",
            "- Replication on held-out facts, paraphrases, and the element-symbol relation matters "
            "more than a large effect on one split.",
            "",
            "## Scope of the claim",
            "",
            "This experiment can support a claim about a linearly readable conflict variable and its "
            "causal relationship to answer-source resolution. It cannot establish phenomenal awareness, "
            "and it cannot prove that the fitted direction is the model's only conflict representation.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    import numpy as np

    started = time.time()
    args = parse_args()
    if args.max_examples_per_split < 1 or args.random_controls < 1:
        raise ValueError("Example and random-control counts must be positive")
    if args.bootstrap_replicates < 100:
        raise ValueError("--bootstrap-replicates must be at least 100")
    if args.baseline_tolerance < 0:
        raise ValueError("--baseline-tolerance cannot be negative")
    splits = parse_unique_list(args.evaluation_splits, "--evaluation-splits")
    requested_directions = parse_unique_list(args.directions, "--directions")
    strengths = parse_strengths(args.strengths)

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["project"]["seed"])
    seed_everything(seed)
    paths = resolve_paths(args, config)
    for name, path in paths.items():
        if name != "output_dir" and not path.is_file():
            raise FileNotFoundError(path)
    output_dir = paths["output_dir"]
    output_paths = {
        "directions": output_dir / "causal_directions.npz",
        "results": output_dir / "causal_results.jsonl",
        "summary": output_dir / "causal_summary.json",
        "report": output_dir / "causal_report.md",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Causal outputs already exist; pass --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    behavior = read_jsonl(paths["behavior"])
    probe_scores = read_jsonl(paths["probe_scores"])
    rows = merge_behavior_and_scores(behavior, probe_scores)
    probe_metadata = read_json(paths["probe_metadata"])
    collection_metadata = read_json(paths["collection_metadata"])
    experiment = read_jsonl(paths["experiment"])
    dataset_manifest = read_json(paths["dataset_manifest"])
    validate_manifest_file(
        dataset_manifest, "experiment.jsonl", paths["experiment"], len(experiment)
    )
    experiment_by_id = {str(row["sample_id"]): row for row in experiment}
    if len(experiment_by_id) != len(experiment):
        raise ValueError("Experiment sample IDs are not unique")

    with np.load(paths["probe_model"], allow_pickle=False) as loaded:
        probe = {key: loaded[key].copy() for key in loaded.files}
    selected_position = str(scalar(probe, "selected_position_name"))
    if selected_position != "assistant_start":
        raise ValueError(
            "Causal intervention requires a probe selected at assistant_start; "
            f"found {selected_position!r}"
        )
    selected_layer_index = int(scalar(probe, "selected_layer_index"))
    selected_layer_offset = int(scalar(probe, "selected_layer_offset"))
    selected_position_offset = int(scalar(probe, "selected_position_index"))

    manifest, entries = validate_activation_manifest(paths["activation_manifest"], rows)
    if int(manifest["layer_indices"][selected_layer_offset]) != selected_layer_index:
        raise ValueError("Probe layer offset disagrees with activation manifest")
    if str(manifest["position_names"][selected_position_offset]) != selected_position:
        raise ValueError("Probe position offset disagrees with activation manifest")
    if probe_metadata.get("inputs", {}).get("behavior_sha256") != file_sha256(paths["behavior"]):
        raise ValueError("Probe metadata does not match the behavior file")
    if probe_metadata.get("inputs", {}).get("activation_manifest_sha256") != file_sha256(
        paths["activation_manifest"]
    ):
        raise ValueError("Probe metadata does not match the activation manifest")

    print("Loading the selected activation cell and constructing training-only directions...")
    x = extract_selected_activations(
        entries,
        manifest,
        rows,
        selected_layer_offset,
        selected_position_offset,
    )
    if x.shape[1] != len(probe["awareness_coef_raw"]):
        raise ValueError("Probe coefficient and activation hidden sizes disagree")
    directions, direction_metadata = construct_directions(
        x, rows, probe, args.random_controls, seed
    )
    unavailable = sorted(set(requested_directions).difference(directions))
    if unavailable:
        raise ValueError(
            f"Requested directions are unavailable: {unavailable}; available={sorted(directions)}"
        )

    available_splits = {str(row["analysis_split"]) for row in rows}
    unknown_splits = sorted(set(splits).difference(available_splits))
    if unknown_splits:
        raise ValueError(f"Evaluation splits are unavailable: {unknown_splits}")
    selected_indices = stable_subset(
        rows, splits, args.max_examples_per_split, seed
    )
    selected_rows = [rows[index] for index in selected_indices]
    if any(str(row["sample_id"]) not in experiment_by_id for row in selected_rows):
        raise ValueError("A selected behavior row is absent from experiment.jsonl")

    # Reproduce the exact model/chat runtime before doing any intervention.
    print(f"Loading {config['model']['id']} at revision {config['model']['revision']}")
    bundle = load_model_bundle(config)
    current_runtime = runtime_fingerprint(
        config,
        bundle,
        {"experiment.jsonl": paths["experiment"], "behavior_results.jsonl": paths["behavior"]},
    )
    verify_runtime(collection_metadata, current_runtime)
    layers = get_decoder_layers(bundle.model)
    if not 0 <= selected_layer_index < len(layers):
        raise ValueError("Selected layer does not exist in the loaded model")

    rendered_by_id = {
        str(row["sample_id"]): render_dataset_record(
            bundle.processor, experiment_by_id[str(row["sample_id"])], config
        ).rendered_text
        for row in selected_rows
    }
    delta_lookup: dict[str, Any] = {"baseline": np.zeros(x.shape[1], dtype=np.float32)}
    scenarios: list[dict[str, Any]] = [
        {
            "scenario_id": "baseline",
            "direction_id": "baseline",
            "strength_sd": 0.0,
            "coefficient": 0.0,
            "delta_key": "baseline",
        }
    ]
    for direction_name in requested_directions:
        scale = float(direction_metadata["scales"][direction_name])
        for strength in strengths:
            delta_key = f"{direction_name}|{strength:+g}"
            coefficient = float(strength * scale)
            delta_lookup[delta_key] = (
                directions[direction_name] * coefficient
            ).astype(np.float32)
            scenarios.append(
                {
                    "scenario_id": delta_key,
                    "direction_id": direction_name,
                    "strength_sd": float(strength),
                    "coefficient": coefficient,
                    "delta_key": delta_key,
                }
            )

    # requests: list[dict[str, Any]] = []
    # for row in selected_rows:
    #     sample_id = str(row["sample_id"])
    #     context_answer = str(row["context_candidate_answer"])
    #     parametric_answer = str(row["best_parametric_answer"])
    #     if context_answer.casefold() == parametric_answer.casefold():
    #         raise ValueError(f"Causal candidates are not distinct for {sample_id}")
    #     for scenario in scenarios:
    #         for role, continuation in (
    #             ("context", context_answer),
    #             ("parametric", parametric_answer),
    #         ):
    #             requests.append(
    #                 {
    #                     "key": f"{sample_id}|{scenario['scenario_id']}|{role}",
    #                     "sample_id": sample_id,
    #                     "scenario_id": scenario["scenario_id"],
    #                     "role": role,
    #                     "rendered_text": rendered_by_id[sample_id],
    #                     "continuation": continuation,
    #                     "delta_key": scenario["delta_key"],
    #                 }
    #             )

    requests: list[dict[str, Any]] = []
    for row in selected_rows:
        sample_id = str(row["sample_id"])
        context_answer = str(row["context_candidate_answer"])
        parametric_answer = str(row["best_parametric_answer"])

        if context_answer.casefold() == parametric_answer.casefold():
            raise ValueError(f"Causal candidates are not distinct for {sample_id}")

        # Every scenario is a contiguous four-row group:
        #
        #   zero-delta context
        #   zero-delta parametric
        #   intervened context
        #   intervened parametric
        #
        # This ensures the intervention is compared with a baseline evaluated
        # using exactly the same batch shape and padding width.
        for scenario in scenarios:
            for scoring_mode, delta_key in (
                ("paired_baseline", "baseline"),
                ("intervention", scenario["delta_key"]),
            ):
                for role, continuation in (
                    ("context", context_answer),
                    ("parametric", parametric_answer),
                ):
                    requests.append(
                        {
                            "key": (
                                f"{sample_id}|{scenario['scenario_id']}|"
                                f"{scoring_mode}|{role}"
                            ),
                            "sample_id": sample_id,
                            "scenario_id": scenario["scenario_id"],
                            "scoring_mode": scoring_mode,
                            "role": role,
                            "rendered_text": rendered_by_id[sample_id],
                            "continuation": continuation,
                            "delta_key": delta_key,
                        }
                    )


    scoring_batch_size = int(
        args.scoring_batch_size
        if args.scoring_batch_size is not None
        else config["collection"]["scoring_batch_size"]
    )
    # if scoring_batch_size < 1:
    #     raise ValueError("--scoring-batch-size must be positive")
    if scoring_batch_size != 4:
        raise ValueError(
            "--scoring-batch-size must be exactly 4 so that every paired "
            "baseline/intervention group is evaluated in one isolated batch"
        )
    # print(
    #     f"Scoring {len(requests)} candidate continuations "
    #     f"({len(selected_rows)} prompts × {len(scenarios)} scenarios × 2 answers)..."
    # )
    print(
        f"Scoring {len(requests)} candidate continuations "
        f"({len(selected_rows)} prompts × {len(scenarios)} scenarios "
        "× 2 conditions × 2 answers)..."
    )
    scores = score_intervened_continuations(
        bundle.model,
        bundle.tokenizer,
        layers[selected_layer_index],
        requests,
        delta_lookup,
        scoring_batch_size,
        int(config["chat"]["max_input_tokens"]),
    )

    awareness_coef = np.asarray(probe["awareness_coef_raw"], dtype=np.float64)
    awareness_intercept = float(scalar(probe, "awareness_intercept_raw"))
    row_index_by_id = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    
    
    # result_rows: list[dict[str, Any]] = []
    # mismatch_values: list[float] = []
    # for row in selected_rows:
    #     sample_id = str(row["sample_id"])
    #     activation = x[row_index_by_id[sample_id]].astype(np.float64)
    #     recomputed_probe_logit = float(np.dot(activation, awareness_coef) + awareness_intercept)
    #     cached_probe_logit = float(row["awareness_probe_logit"])
    #     baseline_context = scores[f"{sample_id}|baseline|context"]["sum_logprob"]
    #     baseline_parametric = scores[f"{sample_id}|baseline|parametric"]["sum_logprob"]
    #     baseline_margin = float(baseline_context - baseline_parametric)
    #     cached_margin = float(row["context_minus_parametric_logprob_margin"])
    #     mismatch = baseline_margin - cached_margin
    #     mismatch_values.append(abs(mismatch))
    #     for scenario in scenarios:
    #         scenario_id = str(scenario["scenario_id"])
    #         context_score = scores[f"{sample_id}|{scenario_id}|context"]
    #         parametric_score = scores[f"{sample_id}|{scenario_id}|parametric"]
    #         margin = float(context_score["sum_logprob"] - parametric_score["sum_logprob"])
    #         delta = np.asarray(delta_lookup[str(scenario["delta_key"])], dtype=np.float64)
    #         probe_delta = float(np.dot(awareness_coef, delta))
    #         result_rows.append(
    #             {
    #                 "sample_id": sample_id,
    #                 "fact_id": row["fact_id"],
    #                 "analysis_split": row["analysis_split"],
    #                 "condition_id": row["condition_id"],
    #                 "policy_id": row["policy_id"],
    #                 "direction_id": scenario["direction_id"],
    #                 "strength_sd": float(scenario["strength_sd"]),
    #                 "intervention_coefficient": float(scenario["coefficient"]),
    #                 "context_answer": row["context_candidate_answer"],
    #                 "parametric_answer": row["best_parametric_answer"],
    #                 "context_answer_sequence_logprob": float(context_score["sum_logprob"]),
    #                 "parametric_answer_sequence_logprob": float(parametric_score["sum_logprob"]),
    #                 "context_answer_num_tokens": int(context_score["num_tokens"]),
    #                 "parametric_answer_num_tokens": int(parametric_score["num_tokens"]),
    #                 "resolution_margin": margin,
    #                 "baseline_resolution_margin": baseline_margin,
    #                 "resolution_margin_delta": margin - baseline_margin,
    #                 "step02_cached_resolution_margin": cached_margin,
    #                 "baseline_minus_step02_margin": mismatch,
    #                 "awareness_probe_logit_before": recomputed_probe_logit,
    #                 "awareness_probe_logit_after": recomputed_probe_logit + probe_delta,
    #                 "awareness_probe_logit_delta": probe_delta,
    #                 "step03_cached_awareness_probe_logit": cached_probe_logit,
    #                 "recomputed_minus_step03_awareness_logit": (
    #                     recomputed_probe_logit - cached_probe_logit
    #                 ),
    #             }
    #         )

    # max_mismatch = max(mismatch_values)
    # baseline_passed = max_mismatch <= args.baseline_tolerance
    # if not baseline_passed and not args.allow_baseline_mismatch:
    #     raise RuntimeError(
    #         f"Baseline scoring mismatch {max_mismatch:.4f} exceeds tolerance "
    #         f"{args.baseline_tolerance:.4f}. Use --allow-baseline-mismatch only after "
    #         "checking model, chat template, padding, and candidate strings."
    #     )
    # if not baseline_passed:
    #     print("WARNING: baseline fidelity tolerance was exceeded")


    result_rows: list[dict[str, Any]] = []
    mismatch_values: list[float] = []
    paired_null_values: list[float] = []

    for row in selected_rows:
        sample_id = str(row["sample_id"])
        activation = x[row_index_by_id[sample_id]].astype(np.float64)

        recomputed_probe_logit = float(
            np.dot(activation, awareness_coef) + awareness_intercept
        )
        cached_probe_logit = float(row["awareness_probe_logit"])
        cached_margin = float(row["context_minus_parametric_logprob_margin"])

        for scenario in scenarios:
            scenario_id = str(scenario["scenario_id"])

            baseline_context_score = scores[
                f"{sample_id}|{scenario_id}|paired_baseline|context"
            ]
            baseline_parametric_score = scores[
                f"{sample_id}|{scenario_id}|paired_baseline|parametric"
            ]
            context_score = scores[
                f"{sample_id}|{scenario_id}|intervention|context"
            ]
            parametric_score = scores[
                f"{sample_id}|{scenario_id}|intervention|parametric"
            ]

            paired_baseline_margin = float(
                baseline_context_score["sum_logprob"]
                - baseline_parametric_score["sum_logprob"]
            )
            margin = float(
                context_score["sum_logprob"]
                - parametric_score["sum_logprob"]
            )
            resolution_delta = margin - paired_baseline_margin
            mismatch = paired_baseline_margin - cached_margin

            # This remains useful diagnostically, but it is not the causal
            # baseline because step 02 used different batch shapes.
            mismatch_values.append(abs(mismatch))

            # In the baseline scenario, both copies have exactly zero
            # intervention. Their difference tests the paired scorer itself.
            if scenario_id == "baseline":
                paired_null_values.append(abs(resolution_delta))

            delta = np.asarray(
                delta_lookup[str(scenario["delta_key"])],
                dtype=np.float64,
            )
            probe_delta = float(np.dot(awareness_coef, delta))

            result_rows.append(
                {
                    "sample_id": sample_id,
                    "fact_id": row["fact_id"],
                    "analysis_split": row["analysis_split"],
                    "condition_id": row["condition_id"],
                    "policy_id": row["policy_id"],
                    "direction_id": scenario["direction_id"],
                    "strength_sd": float(scenario["strength_sd"]),
                    "intervention_coefficient": float(
                        scenario["coefficient"]
                    ),
                    "context_answer": row["context_candidate_answer"],
                    "parametric_answer": row["best_parametric_answer"],
                    "context_answer_sequence_logprob": float(
                        context_score["sum_logprob"]
                    ),
                    "parametric_answer_sequence_logprob": float(
                        parametric_score["sum_logprob"]
                    ),
                    "context_answer_num_tokens": int(
                        context_score["num_tokens"]
                    ),
                    "parametric_answer_num_tokens": int(
                        parametric_score["num_tokens"]
                    ),
                    "resolution_margin": margin,
                    "baseline_resolution_margin": paired_baseline_margin,
                    "resolution_margin_delta": resolution_delta,
                    "step02_cached_resolution_margin": cached_margin,
                    "baseline_minus_step02_margin": mismatch,
                    "awareness_probe_logit_before": recomputed_probe_logit,
                    "awareness_probe_logit_after": (
                        recomputed_probe_logit + probe_delta
                    ),
                    "awareness_probe_logit_delta": probe_delta,
                    "step03_cached_awareness_probe_logit": cached_probe_logit,
                    "recomputed_minus_step03_awareness_logit": (
                        recomputed_probe_logit - cached_probe_logit
                    ),
                }
            )

    paired_null_tolerance = 1e-5
    max_paired_null_difference = max(paired_null_values)

    if max_paired_null_difference > paired_null_tolerance:
        raise RuntimeError(
            "Within-batch zero-intervention mismatch "
            f"{max_paired_null_difference:.8f} exceeds tolerance "
            f"{paired_null_tolerance:.8f}. The paired scorer is not stable."
        )

    # Comparing against step 02 is now diagnostic only. Step 02 used different
    # batch shapes, so its BF16 logits need not be exactly reproducible.
    max_mismatch = max(mismatch_values)
    baseline_passed = max_mismatch <= args.baseline_tolerance

    if not baseline_passed:
        print(
            "WARNING: scores differ from the cached step-02 margins, but the "
            "paired causal baseline passed. The step-02 comparison is "
            "diagnostic only."
        )







    strength_summary, slope_summary = summarize_causal_results(
        result_rows,
        splits,
        requested_directions,
        args.bootstrap_replicates,
        seed,
    )
    direction_arrays: dict[str, Any] = {
        "format_version": np.asarray([1], dtype=np.int32),
        "analysis_version": np.asarray([ANALYSIS_VERSION]),
        "selected_layer_index": np.asarray([selected_layer_index], dtype=np.int32),
        "selected_position_name": np.asarray([selected_position]),
        "awareness_probe_coef_raw": awareness_coef.astype(np.float32),
    }
    for name, direction in directions.items():
        direction_arrays[f"direction__{name}"] = np.asarray(direction, dtype=np.float32)
        direction_arrays[f"scale__{name}"] = np.asarray(
            [direction_metadata["scales"][name]], dtype=np.float64
        )

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "script": Path(__file__).name,
        "elapsed_seconds": time.time() - started,
        "intervention_site": {
            "layer_index": selected_layer_index,
            "position": selected_position,
            "activation_definition": manifest.get("activation_definition"),
            "operation": "add vector to decoder-block output residual",
        },
        "scoring": {
            "primary_outcome": "context answer sum logprob minus parametric answer sum logprob",
            "eos_included": False,
            "decoding_used_for_causal_outcome": False,
            "padding": "left",
            "position_ids": "explicit 0..length-1 over non-padding tokens",
            "injection_token": "final non-padding token of canonical generation prefix",
        },
        "evaluation_splits": splits,
        "requested_directions": requested_directions,
        "strengths_sd": strengths,
        "selected_examples": {
            split: sum(row["analysis_split"] == split for row in selected_rows)
            for split in splits
        },
        "direction_construction": direction_metadata,

        "paired_causal_baseline": {
            "passed": max_paired_null_difference <= paired_null_tolerance,
            "tolerance": paired_null_tolerance,
            "max_absolute_zero_intervention_difference": (
                max_paired_null_difference
            ),
            "method": (
                "zero-delta and intervention candidate pairs evaluated "
                "together in the same isolated four-row batch"
            ),
        },
        
        "baseline_fidelity": {
            "passed": baseline_passed,
            "tolerance": args.baseline_tolerance,
            "max_absolute_margin_difference": max_mismatch,
            "allow_mismatch_override": args.allow_baseline_mismatch,
        },
        "uncertainty": {
            "method": "percentile cluster bootstrap over query fact_id",
            "replicates": args.bootstrap_replicates,
            "seed": seed,
        },
        "by_strength": strength_summary,
        "dose_response_slopes": slope_summary,
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
            if name != "output_dir"
        },
        "runtime": current_runtime,
        "interpretive_limits": [
            "The resolution-orthogonal direction is orthogonal only to the fitted linear awareness readout.",
            "A residual-stream intervention can have off-manifold effects; random controls and dose response are essential.",
            "The experiment tests causal influence on candidate likelihoods, not phenomenal awareness.",
        ],
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }

    save_npz_atomic(output_paths["directions"], **direction_arrays)
    write_jsonl_atomic(output_paths["results"], result_rows)
    write_json_atomic(output_paths["summary"], summary)
    write_text_atomic(output_paths["report"], render_report(summary))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
