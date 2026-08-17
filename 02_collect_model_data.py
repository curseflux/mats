#!/usr/bin/env python3
"""Collect Gemma 4 behavior, candidate likelihoods, and residual activations.

Only complete factorial groups whose query and paragraph facts passed the
context-free knowledge screen are retained.  Behavior is checkpointed as
JSONL; activations are stored in resumable PyTorch shards without raw prompts.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import (
    CODE_VERSION,
    LayerActivationCapture,
    answer_matches,
    append_jsonl,
    file_sha256,
    finalize_jsonl,
    generate_batch,
    get_decoder_layers,
    is_one_orthographic_word,
    json_sha256,
    load_config,
    load_model_bundle,
    locate_activation_positions,
    normalize_answer,
    read_json,
    read_jsonl,
    render_dataset_record,
    resolve_layer_indices,
    runtime_fingerprint,
    score_continuations,
    seed_everything,
    unique_by,
    validate_manifest_file,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace behavior, activation, and metadata outputs owned by this script.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and report the screened subset without loading Gemma.",
    )
    parser.add_argument(
        "--behavior-only",
        action="store_true",
        help="Collect generations and likelihoods but skip activation capture.",
    )
    return parser.parse_args()


def expected_factorial_cells(manifest: Mapping[str, Any]) -> set[tuple[str, str]]:
    conditions = {
        str(condition["condition_id"]) for condition in manifest.get("conditions", [])
    }
    policies = {str(policy) for policy in manifest.get("included_policies", [])}
    if not conditions or not policies:
        raise ValueError("Dataset manifest does not define conditions and included_policies")
    return {(condition, policy) for condition in conditions for policy in policies}


def validate_and_select(
    experiment: Sequence[Mapping[str, Any]],
    eligible_facts: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    require_complete_groups: bool,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if not require_complete_groups:
        raise ValueError(
            "This design requires collection.require_complete_factorial_groups=true"
        )
    unique_by(experiment, "sample_id", "experiment.jsonl")
    unique_by(eligible_facts, "fact_id", "eligible_facts.jsonl")
    eligibility = {
        str(record["fact_id"]): bool(record.get("eligible")) for record in eligible_facts
    }
    expected_cells = expected_factorial_cells(manifest)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for record in experiment:
        sample_id = str(record["sample_id"])
        if record.get("record_type") != "conflict_awareness_experiment":
            raise ValueError(f"Unexpected record_type in {sample_id}")
        for field in (
            "fact_id",
            "claim_fact_id",
            "matched_factorial_group_id",
            "condition_id",
            "policy_id",
            "parametric_candidate_answer",
            "claim_answer",
            "raw_prompt",
            "semantic_positions",
        ):
            if field not in record:
                raise ValueError(f"Sample {sample_id} is missing {field}")
        if str(record["fact_id"]) not in eligibility:
            raise ValueError(f"Sample {sample_id} references an unscreened query fact")
        if str(record["claim_fact_id"]) not in eligibility:
            raise ValueError(f"Sample {sample_id} references an unscreened paragraph fact")
        if bool(record["query_conflict_label"]) != (
            (not bool(record["claim_is_world_true"]))
            and bool(record["claim_is_query_relevant"])
        ):
            raise ValueError(f"Sample {sample_id} has an inconsistent query-conflict label")
        messages = record.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or messages[0].get("role") != "user"
            or messages[0].get("content") != record["raw_prompt"]
        ):
            raise ValueError(f"Sample {sample_id} does not preserve raw_prompt in messages")
        positions = record["semantic_positions"]
        if not isinstance(positions, Mapping):
            raise ValueError(f"Sample {sample_id} has invalid semantic_positions")
        groups[str(record["matched_factorial_group_id"])].append(record)

    selected_group_ids: set[str] = set()
    exclusion_reasons: Counter[str] = Counter()
    for group_id, group in groups.items():
        cells = [(str(row["condition_id"]), str(row["policy_id"])) for row in group]
        if len(cells) != len(set(cells)) or set(cells) != expected_cells:
            raise ValueError(f"Factorial group {group_id} is incomplete or has duplicate cells")
        query_fact_ids = {str(row["fact_id"]) for row in group}
        if len(query_fact_ids) != 1:
            raise ValueError(f"Factorial group {group_id} changes query fact")
        required_fact_ids = query_fact_ids | {str(row["claim_fact_id"]) for row in group}
        unknown = sorted(fact_id for fact_id in required_fact_ids if fact_id not in eligibility)
        if unknown:
            raise ValueError(f"Factorial group {group_id} has unscreened facts: {unknown}")
        failed_query = any(not eligibility[fact_id] for fact_id in query_fact_ids)
        failed_claim = any(
            not eligibility[str(row["claim_fact_id"])] for row in group
        )
        if failed_query:
            exclusion_reasons["query_fact_failed_screening"] += 1
        if failed_claim:
            exclusion_reasons["paragraph_fact_failed_screening"] += 1
        if not failed_query and not failed_claim:
            selected_group_ids.add(group_id)

    selected = [
        row
        for row in experiment
        if str(row["matched_factorial_group_id"]) in selected_group_ids
    ]
    if require_complete_groups:
        selected_cells: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for row in selected:
            selected_cells[str(row["matched_factorial_group_id"])].add(
                (str(row["condition_id"]), str(row["policy_id"]))
            )
        if any(cells != expected_cells for cells in selected_cells.values()):
            raise AssertionError("Screening filter broke a factorial group")
    if not selected:
        raise RuntimeError("No complete factorial groups survived knowledge screening")

    summary = {
        "experiment_records": len(experiment),
        "factorial_groups": len(groups),
        "selected_records": len(selected),
        "selected_factorial_groups": len(selected_group_ids),
        "records_per_group": len(expected_cells),
        "expected_cells": [list(cell) for cell in sorted(expected_cells)],
        "excluded_group_reason_counts_nonexclusive": dict(exclusion_reasons),
        "eligible_facts": sum(eligibility.values()),
        "screened_facts": len(eligibility),
    }
    return selected, summary


def candidate_requests_for_batch(
    records: Sequence[Mapping[str, Any]],
    rendered_texts: Sequence[str],
) -> tuple[list[dict[str, str]], dict[str, dict[str, list[str]]]]:
    requests: list[dict[str, str]] = []
    lookup: dict[str, dict[str, list[str]]] = {}
    for record, rendered in zip(records, rendered_texts):
        sample_id = str(record["sample_id"])
        role_keys: dict[str, list[str]] = defaultdict(list)
        key_by_text: dict[str, str] = {}

        def add(role: str, answer: str) -> None:
            if answer not in key_by_text:
                key = f"{sample_id}|candidate|{len(key_by_text)}"
                key_by_text[answer] = key
                requests.append(
                    {"key": key, "rendered_text": rendered, "continuation": answer}
                )
            role_keys[role].append(key_by_text[answer])

        parametric_answers = record.get("acceptable_world_true_answers") or [
            record["parametric_candidate_answer"]
        ]
        for answer in parametric_answers:
            add("parametric", str(answer))
        add("claim", str(record["claim_answer"]))
        context_answer = record.get("context_candidate_answer")
        if context_answer is not None:
            add("context", str(context_answer))
        lookup[sample_id] = dict(role_keys)
    return requests, lookup


def classify_generation(
    record: Mapping[str, Any],
    generated_text: str,
) -> dict[str, Any]:
    one_word = is_one_orthographic_word(generated_text)
    parametric_answers = record.get("acceptable_world_true_answers") or [
        record["parametric_candidate_answer"]
    ]
    matches_parametric = answer_matches(generated_text, list(parametric_answers))
    matches_claim = normalize_answer(generated_text) == normalize_answer(record["claim_answer"])
    context_answer = record.get("context_candidate_answer")
    matches_context = context_answer is not None and (
        normalize_answer(generated_text) == normalize_answer(context_answer)
    )

    if not one_word:
        source = "unparseable"
    elif matches_parametric and matches_claim:
        source = (
            "shared_parametric_and_context"
            if context_answer is not None
            else "shared_parametric_and_claim"
        )
    elif matches_parametric:
        source = "parametric"
    elif matches_context:
        source = "contextual"
    elif matches_claim:
        source = "irrelevant_claim"
    else:
        source = "other"

    if record["policy_id"] == "parametric":
        policy_compliant: bool | None = matches_parametric
    elif record["policy_id"] == "context" and context_answer is not None:
        policy_compliant = matches_context
    else:
        policy_compliant = None
    return {
        "generated_answer_is_one_word": one_word,
        "generated_matches_parametric": matches_parametric,
        "generated_matches_claim": matches_claim,
        "generated_matches_context": matches_context,
        "observed_knowledge_source": source,
        "policy_compliant": policy_compliant,
    }


def evaluate_behavior_batch(
    records: Sequence[Mapping[str, Any]],
    bundle: Any,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rendered = [render_dataset_record(bundle.processor, record, config) for record in records]
    rendered_texts = [prompt.rendered_text for prompt in rendered]
    generations = generate_batch(bundle.model, bundle.tokenizer, rendered_texts, config)
    requests, lookup = candidate_requests_for_batch(records, rendered_texts)
    scores = score_continuations(
        bundle.model,
        bundle.tokenizer,
        requests,
        batch_size=int(config["collection"]["scoring_batch_size"]),
        max_input_tokens=int(config["chat"]["max_input_tokens"]),
    )
    score_by_key = {score.key: score for score in scores}

    output: list[dict[str, Any]] = []
    for record, generation in zip(records, generations):
        sample_id = str(record["sample_id"])
        roles = lookup[sample_id]
        parametric_scores = [score_by_key[key] for key in roles["parametric"]]
        best_parametric = max(parametric_scores, key=lambda score: score.sum_logprob)
        claim_score = score_by_key[roles["claim"][0]]
        context_score = (
            score_by_key[roles["context"][0]] if "context" in roles else None
        )
        claim_distinct = normalize_answer(record["claim_answer"]) not in {
            normalize_answer(answer)
            for answer in (
                record.get("acceptable_world_true_answers")
                or [record["parametric_candidate_answer"]]
            )
        }
        annotations = classify_generation(record, generation["text"])

        result = {
            "code_version": CODE_VERSION,
            "source_schema_version": record["schema_version"],
            "record_type": "gemma4_conflict_behavior",
            "sample_id": sample_id,
            "stimulus_family_id": record["stimulus_family_id"],
            "matched_factorial_group_id": record["matched_factorial_group_id"],
            "content_pair_id": record["content_pair_id"],
            "fact_id": record["fact_id"],
            "claim_fact_id": record["claim_fact_id"],
            "relation_id": record["relation_id"],
            "transfer_role": record["transfer_role"],
            "fact_split": record["fact_split"],
            "cv_fold": record["cv_fold"],
            "counterbalance_round": record["counterbalance_round"],
            "condition_id": record["condition_id"],
            "claim_is_world_true": record["claim_is_world_true"],
            "claim_is_query_relevant": record["claim_is_query_relevant"],
            "claim_conflict_label": record["claim_conflict_label"],
            "query_conflict_label": record["query_conflict_label"],
            "effective_claim_conflict": bool(record["claim_conflict_label"]),
            "effective_query_conflict": bool(record["query_conflict_label"]),
            "policy_id": record["policy_id"],
            "policy_target": record["policy_target"],
            "policy_analysis_role": record["policy_analysis_role"],
            "template_bundle_id": record["template_bundle_id"],
            "template_analysis_role": record["template_analysis_role"],
            "query_subject": record["query_subject"],
            "world_true_answer": record["world_true_answer"],
            "acceptable_world_true_answers": record["acceptable_world_true_answers"],
            "parametric_candidate_answer": record["parametric_candidate_answer"],
            "claim_subject": record["claim_subject"],
            "claim_answer": record["claim_answer"],
            "claim_world_true_answer": record["claim_world_true_answer"],
            "false_answer_source_fact_id": record.get("false_answer_source_fact_id"),
            "context_candidate_answer": record.get("context_candidate_answer"),
            "candidate_answers_are_distinct": record["candidate_answers_are_distinct"],
            "claim_and_parametric_answers_are_distinct": claim_distinct,
            "expected_answer_under_policy": record.get("expected_answer_under_policy"),
            "generated_answer": generation["text"],
            "generated_answer_raw": generation["raw_text"],
            "generated_answer_normalized": generation["normalized_text"],
            "generated_token_ids": generation["token_ids"],
            "generation_finish_reason": generation["finish_reason"],
            "generation_stop_token_id": generation["stop_token_id"],
            **annotations,
            "parametric_answer_scores": [score.as_dict() for score in parametric_scores],
            "best_parametric_answer": best_parametric.continuation,
            "parametric_answer_sequence_logprob": best_parametric.sum_logprob,
            "claim_answer_score": claim_score.as_dict(),
            "claim_answer_sequence_logprob": claim_score.sum_logprob,
            "claim_minus_parametric_logprob_margin": (
                claim_score.sum_logprob - best_parametric.sum_logprob
                if claim_distinct
                else None
            ),
            "context_answer_score": context_score.as_dict() if context_score else None,
            "context_answer_sequence_logprob": (
                context_score.sum_logprob if context_score else None
            ),
            "context_minus_parametric_logprob_margin": (
                context_score.sum_logprob - best_parametric.sum_logprob
                if context_score is not None and claim_distinct
                else None
            ),
        }
        output.append(result)
    return output


def verify_screening_runtime(
    screening_metadata: Mapping[str, Any],
    current_runtime: Mapping[str, Any],
) -> None:
    paths = (
        ("model", "id"),
        ("model", "requested_revision"),
        ("model", "loaded_commit_hash"),
        ("model", "class"),
        ("chat", "add_generation_prompt"),
        ("chat", "enable_thinking"),
        ("chat", "chat_template_sha256"),
        ("chat", "pad_token_id"),
        ("chat", "generation_eos_token_ids"),
        ("packages", "transformers"),
    )
    differences: list[str] = []
    for path in paths:
        old: Any = screening_metadata
        new: Any = current_runtime
        for key in path:
            old = old.get(key) if isinstance(old, Mapping) else None
            new = new.get(key) if isinstance(new, Mapping) else None
        if old != new:
            differences.append(f"{'.'.join(path)}: screening={old!r}, current={new!r}")
    if differences:
        raise RuntimeError(
            "The current runtime does not match the knowledge-screening runtime:\n  "
            + "\n  ".join(differences)
            + "\nRerun 01_screen_knowledge.py with this environment."
        )


class ActivationShardWriter:
    """Append atomic, recoverable activation shards in a fixed sample order."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        directory: Path,
        manifest_path: Path,
        expected_sample_ids: Sequence[str],
        signature: str,
        layer_indices: Sequence[int],
        position_names: Sequence[str],
        storage_dtype: str,
        shard_size: int,
        *,
        overwrite: bool,
    ) -> None:
        if shard_size < 1:
            raise ValueError("Activation shard_size must be positive")
        self.directory = directory
        self.manifest_path = manifest_path
        self.expected_sample_ids = list(expected_sample_ids)
        self.signature = signature
        self.layer_indices = list(layer_indices)
        self.position_names = list(position_names)
        self.storage_dtype = storage_dtype
        self.shard_size = int(shard_size)
        self.buffer_ids: list[str] = []
        self.buffer_activations: list[Any] = []
        self.buffer_positions: list[Any] = []
        self.buffer_token_ids: list[Any] = []
        self.buffer_lengths: list[Any] = []

        self.directory.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for path in self.directory.glob("shard-*.pt"):
                path.unlink()
            for path in self.directory.glob("shard-*.pt.tmp"):
                path.unlink()
            if self.manifest_path.exists():
                self.manifest_path.unlink()
        else:
            for path in self.directory.glob("shard-*.pt.tmp"):
                path.unlink()

        if self.manifest_path.exists():
            self.manifest = read_json(self.manifest_path)
            self._validate_manifest()
        else:
            self.manifest = {
                "format_version": self.FORMAT_VERSION,
                "code_version": CODE_VERSION,
                "input_signature": self.signature,
                "activation_definition": "decoder block output residual; before final model RMSNorm",
                "layer_indices": self.layer_indices,
                "position_names": self.position_names,
                "storage_dtype": self.storage_dtype,
                "shard_size": self.shard_size,
                "expected_samples": len(self.expected_sample_ids),
                "ordered_sample_ids_sha256": json_sha256(self.expected_sample_ids),
                "completed_samples": 0,
                "complete": False,
                "shards": [],
            }
            write_json_atomic(self.manifest_path, self.manifest)
        self._recover_unlisted_shards()

    @property
    def completed_samples(self) -> int:
        return int(self.manifest["completed_samples"])

    def _validate_manifest(self) -> None:
        expected = {
            "format_version": self.FORMAT_VERSION,
            "input_signature": self.signature,
            "layer_indices": self.layer_indices,
            "position_names": self.position_names,
            "storage_dtype": self.storage_dtype,
            "shard_size": self.shard_size,
            "expected_samples": len(self.expected_sample_ids),
            "ordered_sample_ids_sha256": json_sha256(self.expected_sample_ids),
        }
        mismatches = [
            key for key, value in expected.items() if self.manifest.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"Activation manifest is incompatible in {mismatches}; pass --overwrite"
            )
        cursor = 0
        for shard in self.manifest.get("shards", []):
            start, end = int(shard["start_index"]), int(shard["end_index"])
            if start != cursor or not start < end <= len(self.expected_sample_ids):
                raise ValueError("Activation manifest has non-contiguous shard indices")
            expected_ids = self.expected_sample_ids[start:end]
            if shard.get("sample_ids_sha256") != json_sha256(expected_ids):
                raise ValueError("Activation manifest sample ordering does not match this run")
            path = self.directory / str(shard["file"])
            if path.parent != self.directory or not path.is_file():
                raise FileNotFoundError(f"Missing activation shard: {path}")
            if path.stat().st_size != int(shard["bytes"]):
                raise ValueError(f"Activation shard size changed: {path}")
            cursor = end
        if cursor != int(self.manifest.get("completed_samples", -1)):
            raise ValueError("Activation manifest completed_samples is inconsistent")

    def _recover_unlisted_shards(self) -> None:
        import torch

        listed = {str(shard["file"]) for shard in self.manifest.get("shards", [])}
        unlisted = sorted(
            path for path in self.directory.glob("shard-*.pt") if path.name not in listed
        )
        for path in unlisted:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            start = self.completed_samples
            sample_ids = [str(value) for value in payload.get("sample_ids", [])]
            end = start + len(sample_ids)
            if (
                payload.get("input_signature") != self.signature
                or not sample_ids
                or sample_ids != self.expected_sample_ids[start:end]
                or payload.get("layer_indices") != self.layer_indices
                or payload.get("position_names") != self.position_names
            ):
                raise ValueError(f"Cannot safely recover unlisted activation shard: {path}")
            expected_name = self._shard_name(start, end)
            if path.name != expected_name:
                raise ValueError(f"Unexpected unlisted activation shard name: {path.name}")
            self._register_shard(path, start, end)

    @staticmethod
    def _shard_name(start: int, end: int) -> str:
        return f"shard-{start:06d}-{end:06d}.pt"

    def _register_shard(self, path: Path, start: int, end: int) -> None:
        entry = {
            "file": path.name,
            "start_index": start,
            "end_index": end,
            "samples": end - start,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "sample_ids_sha256": json_sha256(self.expected_sample_ids[start:end]),
        }
        self.manifest["shards"].append(entry)
        self.manifest["completed_samples"] = end
        self.manifest["complete"] = end == len(self.expected_sample_ids)
        write_json_atomic(self.manifest_path, self.manifest)

    def add(
        self,
        sample_ids: Sequence[str],
        activations: Any,
        unpadded_position_indices: Any,
        position_token_ids: Any,
        prompt_lengths: Any,
    ) -> None:
        import torch

        sample_ids = [str(value) for value in sample_ids]
        batch_size = len(sample_ids)
        if not (
            activations.shape[0]
            == unpadded_position_indices.shape[0]
            == position_token_ids.shape[0]
            == prompt_lengths.shape[0]
            == batch_size
        ):
            raise ValueError("Activation batch components have inconsistent row counts")
        start = self.completed_samples + len(self.buffer_ids)
        if sample_ids != self.expected_sample_ids[start : start + batch_size]:
            raise ValueError("Activation samples arrived out of the declared order")
        self.buffer_ids.extend(sample_ids)
        self.buffer_activations.append(activations.cpu())
        self.buffer_positions.append(unpadded_position_indices.to(dtype=torch.int32).cpu())
        self.buffer_token_ids.append(position_token_ids.to(dtype=torch.int32).cpu())
        self.buffer_lengths.append(prompt_lengths.to(dtype=torch.int32).cpu())
        while len(self.buffer_ids) >= self.shard_size:
            self._flush(self.shard_size)

    def _flush(self, count: int) -> None:
        import torch

        if not 0 < count <= len(self.buffer_ids):
            raise ValueError("Invalid activation flush size")
        activations = torch.cat(self.buffer_activations, dim=0)
        positions = torch.cat(self.buffer_positions, dim=0)
        token_ids = torch.cat(self.buffer_token_ids, dim=0)
        lengths = torch.cat(self.buffer_lengths, dim=0)
        sample_ids = self.buffer_ids[:count]
        start, end = self.completed_samples, self.completed_samples + count
        payload = {
            "format_version": self.FORMAT_VERSION,
            "code_version": CODE_VERSION,
            "input_signature": self.signature,
            "sample_ids": sample_ids,
            "layer_indices": self.layer_indices,
            "position_names": self.position_names,
            "activations": activations[:count].contiguous(),
            "position_token_indices_unpadded": positions[:count].contiguous(),
            "position_token_ids": token_ids[:count].contiguous(),
            "prompt_lengths": lengths[:count].contiguous(),
        }
        path = self.directory / self._shard_name(start, end)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self._register_shard(path, start, end)

        self.buffer_ids = self.buffer_ids[count:]
        self.buffer_activations = [activations[count:]] if count < len(activations) else []
        self.buffer_positions = [positions[count:]] if count < len(positions) else []
        self.buffer_token_ids = [token_ids[count:]] if count < len(token_ids) else []
        self.buffer_lengths = [lengths[count:]] if count < len(lengths) else []

    def finish(self) -> None:
        while len(self.buffer_ids) > self.shard_size:
            self._flush(self.shard_size)
        if self.buffer_ids:
            self._flush(len(self.buffer_ids))
        if self.completed_samples != len(self.expected_sample_ids):
            raise RuntimeError("Activation collection ended before all expected samples")
        self.manifest["complete"] = True
        write_json_atomic(self.manifest_path, self.manifest)


def collect_activations(
    records: Sequence[Mapping[str, Any]],
    bundle: Any,
    config: Mapping[str, Any],
    behavior_signature: str,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    import torch

    activation_cfg = config["collection"]["activations"]
    position_names = [str(value) for value in activation_cfg["positions"]]
    layers = get_decoder_layers(bundle.model)
    layer_indices = resolve_layer_indices(activation_cfg["layers"], len(layers))
    sample_ids = [str(record["sample_id"]) for record in records]
    activation_signature = json_sha256(
        {
            "code_version": CODE_VERSION,
            "behavior_signature": behavior_signature,
            "ordered_sample_ids": sample_ids,
            "activation_config": activation_cfg,
            "hidden_size": int(
                getattr(bundle.model.config, "text_config", bundle.model.config).hidden_size
            ),
        }
    )
    directory = Path(config["paths"]["output_dir"]) / str(activation_cfg["directory"])
    manifest_path = directory / str(activation_cfg["manifest_file"])
    writer = ActivationShardWriter(
        directory=directory,
        manifest_path=manifest_path,
        expected_sample_ids=sample_ids,
        signature=activation_signature,
        layer_indices=layer_indices,
        position_names=position_names,
        storage_dtype=str(activation_cfg["storage_dtype"]),
        shard_size=int(activation_cfg["shard_size"]),
        overwrite=overwrite,
    )
    hidden_size = int(
        getattr(bundle.model.config, "text_config", bundle.model.config).hidden_size
    )
    bytes_per_value = 2 if activation_cfg["storage_dtype"] == "float16" else 4
    estimated_gib = (
        len(records)
        * len(layer_indices)
        * len(position_names)
        * hidden_size
        * bytes_per_value
        / (1024**3)
    )
    print(
        f"activations: {len(layer_indices)} layers x {len(position_names)} positions; "
        f"estimated tensor payload {estimated_gib:.2f} GiB"
    )
    if writer.completed_samples:
        print(f"activations: resuming at {writer.completed_samples}/{len(records)} samples")
    if writer.manifest.get("complete"):
        return {
            "manifest": str(manifest_path),
            "input_signature": activation_signature,
            "estimated_tensor_gib": estimated_gib,
            "layers": layer_indices,
            "positions": position_names,
            "completed_samples": writer.completed_samples,
            "reused_complete_output": True,
        }

    pending = records[writer.completed_samples :]
    batch_size = int(config["collection"]["activation_batch_size"])
    report_every = int(config["runtime"].get("report_every_batches", 10))
    with LayerActivationCapture(
        bundle.model,
        layer_indices,
        str(activation_cfg["storage_dtype"]),
    ) as capture:
        for batch_index, start in enumerate(range(0, len(pending), batch_size), start=1):
            batch = pending[start : start + batch_size]
            prompts = [render_dataset_record(bundle.processor, row, config) for row in batch]
            encoded, position_indices, position_token_ids, prompt_lengths = (
                locate_activation_positions(
                    bundle.tokenizer,
                    prompts,
                    position_names,
                    int(config["chat"]["max_input_tokens"]),
                )
            )
            activations = capture.capture(encoded, position_indices)
            width = int(encoded["input_ids"].shape[1])
            left_padding = width - prompt_lengths.to(dtype=torch.long)
            unpadded_indices = position_indices - left_padding[:, None]
            if int(unpadded_indices.min().item()) < 0:
                raise AssertionError("An activation position resolved inside left padding")
            writer.add(
                [str(row["sample_id"]) for row in batch],
                activations,
                unpadded_indices,
                position_token_ids,
                prompt_lengths,
            )
            if batch_index % report_every == 0 or start + len(batch) == len(pending):
                buffered = len(writer.buffer_ids)
                done = writer.completed_samples + buffered
                print(f"activations: {done}/{len(records)} samples")
            del encoded, position_indices, position_token_ids, activations
    writer.finish()
    return {
        "manifest": str(manifest_path),
        "input_signature": activation_signature,
        "estimated_tensor_gib": estimated_gib,
        "layers": layer_indices,
        "positions": position_names,
        "completed_samples": writer.completed_samples,
        "reused_complete_output": False,
    }


def behavior_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "observed_knowledge_source": dict(
            Counter(str(row["observed_knowledge_source"]) for row in records)
        ),
        "by_condition": dict(Counter(str(row["condition_id"]) for row in records)),
        "by_policy": dict(Counter(str(row["policy_id"]) for row in records)),
        "by_template_bundle": dict(
            Counter(str(row["template_bundle_id"]) for row in records)
        ),
    }


def main() -> None:
    started = time.time()
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["project"]["seed"]))
    dataset_dir = Path(config["paths"]["dataset_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    experiment_path = dataset_dir / "experiment.jsonl"
    dataset_manifest_path = dataset_dir / "manifest.json"
    eligible_path = output_dir / str(config["screening"]["eligible_facts_file"])
    screening_metadata_path = output_dir / str(config["screening"]["run_metadata_file"])
    for path in (
        experiment_path,
        dataset_manifest_path,
        eligible_path,
        screening_metadata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    experiment = read_jsonl(experiment_path)
    eligible_facts = read_jsonl(eligible_path)
    dataset_manifest = read_json(dataset_manifest_path)
    validate_manifest_file(
        dataset_manifest,
        "experiment.jsonl",
        experiment_path,
        len(experiment),
    )
    screening_metadata = read_json(screening_metadata_path)
    selected, selection_summary = validate_and_select(
        experiment,
        eligible_facts,
        dataset_manifest,
        require_complete_groups=bool(
            config["collection"]["require_complete_factorial_groups"]
        ),
    )
    print("Screened subset:", selection_summary)
    if args.validate_only:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = output_dir / str(config["collection"]["behavior_results_file"])
    partial_path = behavior_path.with_name(behavior_path.name + ".partial")
    partial_metadata_path = partial_path.with_name(partial_path.name + ".meta.json")
    behavior_metadata_path = behavior_path.with_name(behavior_path.name + ".meta.json")
    run_metadata_path = output_dir / str(config["collection"]["run_metadata_file"])
    activation_cfg = config["collection"]["activations"]
    activation_dir = output_dir / str(activation_cfg["directory"])

    if args.overwrite:
        for path in (
            behavior_path,
            partial_path,
            partial_metadata_path,
            behavior_metadata_path,
            run_metadata_path,
        ):
            if path.exists():
                path.unlink()
    elif behavior_path.exists() and partial_path.exists():
        raise FileExistsError("Both final and partial behavior results exist; pass --overwrite")

    selected_ids = [str(row["sample_id"]) for row in selected]
    signature_payload = {
        "code_version": CODE_VERSION,
        "model": config["model"],
        "chat": config["chat"],
        "generation": config["generation"],
        "collection": {
            "generation_batch_size": config["collection"]["generation_batch_size"],
            "scoring_batch_size": config["collection"]["scoring_batch_size"],
            "require_complete_factorial_groups": config["collection"][
                "require_complete_factorial_groups"
            ],
        },
        "inputs": {
            "experiment.jsonl": file_sha256(experiment_path),
            "manifest.json": file_sha256(dataset_manifest_path),
            "eligible_facts.jsonl": file_sha256(eligible_path),
            "screening_run.json": file_sha256(screening_metadata_path),
        },
        "selected_records": len(selected_ids),
        "selected_sample_ids_sha256": json_sha256(selected_ids),
    }
    behavior_signature = json_sha256(signature_payload)

    if behavior_path.exists():
        if not behavior_metadata_path.exists():
            if (
                not partial_metadata_path.exists()
                or read_json(partial_metadata_path).get("input_signature")
                != behavior_signature
            ):
                raise FileNotFoundError(
                    f"Behavior provenance is missing: {behavior_metadata_path}; pass --overwrite"
                )
            recovered = read_jsonl(behavior_path)
            expected_ids = selected_ids
            if [str(row["sample_id"]) for row in recovered] != expected_ids:
                raise ValueError("Cannot recover provenance for misordered behavior results")
            write_json_atomic(
                behavior_metadata_path,
                {
                    "code_version": CODE_VERSION,
                    "input_signature": behavior_signature,
                    "ordered_sample_ids_sha256": json_sha256(expected_ids),
                    "records": len(expected_ids),
                },
            )
            partial_metadata_path.unlink()
        if read_json(behavior_metadata_path).get("input_signature") != behavior_signature:
            raise ValueError("Existing behavior results are incompatible; pass --overwrite")
        behavior_results = read_jsonl(behavior_path)
        unique_by(behavior_results, "sample_id", str(behavior_path))
        if [str(row["sample_id"]) for row in behavior_results] != selected_ids:
            raise ValueError("Existing behavior results have the wrong sample order")
        print(f"behavior: reusing {len(behavior_results)} completed samples")
    else:
        if behavior_metadata_path.exists():
            raise FileExistsError(
                f"Found behavior provenance without final results: {behavior_metadata_path}"
            )
        if partial_path.exists() and not partial_metadata_path.exists():
            raise FileNotFoundError(
                f"Behavior checkpoint provenance is missing: {partial_metadata_path}"
            )
        if partial_metadata_path.exists():
            if read_json(partial_metadata_path).get("input_signature") != behavior_signature:
                raise ValueError("Partial behavior checkpoint is incompatible; pass --overwrite")
        else:
            write_json_atomic(
                partial_metadata_path,
                {
                    "code_version": CODE_VERSION,
                    "input_signature": behavior_signature,
                    "signature_payload": signature_payload,
                },
            )

    print(f"Loading {config['model']['id']} at revision {config['model']['revision']}")
    bundle = load_model_bundle(config)
    current_runtime = runtime_fingerprint(
        config,
        bundle,
        {
            "experiment.jsonl": experiment_path,
            "manifest.json": dataset_manifest_path,
            "eligible_facts.jsonl": eligible_path,
        },
    )
    verify_screening_runtime(screening_metadata, current_runtime)

    if not behavior_path.exists():
        completed = read_jsonl(partial_path) if partial_path.exists() else []
        unique_by(completed, "sample_id", str(partial_path))
        expected_ids = selected_ids
        completed_ids = [str(row["sample_id"]) for row in completed]
        if completed_ids != expected_ids[: len(completed_ids)]:
            raise ValueError("Partial behavior checkpoint is not an ordered prefix of this run")
        pending = selected[len(completed) :]
        batch_size = int(config["collection"]["generation_batch_size"])
        report_every = int(config["runtime"].get("report_every_batches", 10))
        for batch_index, start in enumerate(range(0, len(pending), batch_size), start=1):
            batch = pending[start : start + batch_size]
            evaluated = evaluate_behavior_batch(batch, bundle, config)
            append_jsonl(partial_path, evaluated)
            if batch_index % report_every == 0 or start + len(batch) == len(pending):
                print(f"behavior: {len(completed) + start + len(batch)}/{len(selected)} samples")
        behavior_results = read_jsonl(partial_path)
        if [str(row["sample_id"]) for row in behavior_results] != expected_ids:
            raise RuntimeError("Behavior checkpoint is incomplete after processing")
        finalize_jsonl(partial_path, behavior_path)
        write_json_atomic(
            behavior_metadata_path,
            {
                "code_version": CODE_VERSION,
                "input_signature": behavior_signature,
                "ordered_sample_ids_sha256": json_sha256(expected_ids),
                "records": len(expected_ids),
            },
        )
        partial_metadata_path.unlink()

    activation_result: dict[str, Any] | None = None
    if bool(activation_cfg["enabled"]) and not args.behavior_only:
        activation_result = collect_activations(
            selected,
            bundle,
            config,
            behavior_signature,
            overwrite=args.overwrite,
        )
    elif args.overwrite and activation_dir.exists():
        # Behavior changed, so stale activation outputs must not appear compatible.
        for path in activation_dir.glob("shard-*.pt"):
            path.unlink()
        for path in activation_dir.glob("shard-*.pt.tmp"):
            path.unlink()
        activation_manifest = activation_dir / str(activation_cfg["manifest_file"])
        if activation_manifest.exists():
            activation_manifest.unlink()

    run_metadata = dict(current_runtime)
    run_metadata.update(
        {
            "script": Path(__file__).name,
            "elapsed_seconds": time.time() - started,
            "selection": selection_summary,
            "behavior": {
                **behavior_summary(behavior_results),
                "results": str(behavior_path),
                "input_signature": behavior_signature,
                "generation": "greedy",
                "sequence_score": "sum of answer-token log-probabilities; EOS excluded",
            },
            "activations": activation_result,
        }
    )
    write_json_atomic(run_metadata_path, run_metadata)
    print(f"Wrote {behavior_path}")
    if activation_result:
        print(f"Wrote {activation_result['manifest']}")


if __name__ == "__main__":
    main()
