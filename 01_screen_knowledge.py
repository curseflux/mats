#!/usr/bin/env python3
"""Establish which candidate facts Gemma 4 demonstrably knows.

A fact is eligible only when every required context-free paraphrase produces an
acceptable answer and the best acceptable true-answer sequence has greater
summed log-probability than every false answer assigned to that fact.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import (
    CODE_VERSION,
    answer_matches,
    append_jsonl,
    file_sha256,
    finalize_jsonl,
    generate_batch,
    is_one_orthographic_word,
    json_sha256,
    load_config,
    load_model_bundle,
    read_json,
    read_jsonl,
    render_dataset_record,
    runtime_fingerprint,
    score_continuations,
    seed_everything,
    unique_by,
    validate_manifest_file,
    write_json_atomic,
    write_jsonl_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this script's existing results and checkpoint.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and dataset files without loading the model.",
    )
    return parser.parse_args()


def validate_inputs(
    facts: Sequence[Mapping[str, Any]],
    screening: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    unique_by(facts, "fact_id", "facts.jsonl")
    unique_by(screening, "sample_id", "screening.jsonl")
    fact_ids = {str(fact["fact_id"]) for fact in facts}
    by_fact: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in screening:
        sample_id = str(record["sample_id"])
        fact_id = str(record.get("fact_id", ""))
        if fact_id not in fact_ids:
            raise ValueError(f"Screening sample {sample_id} references unknown fact {fact_id!r}")
        if record.get("record_type") != "parametric_knowledge_screen":
            raise ValueError(f"Unexpected record_type in {sample_id}")
        messages = record.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or messages[0].get("role") != "user"
            or messages[0].get("content") != record.get("raw_prompt")
        ):
            raise ValueError(f"Screening sample {sample_id} is not exactly one user message")
        if not record.get("acceptable_true_answers"):
            raise ValueError(f"Screening sample {sample_id} has no acceptable true answers")
        if not record.get("contrast_false_answers"):
            raise ValueError(f"Screening sample {sample_id} has no false-answer contrasts")
        by_fact[fact_id].append(record)

    bundle_sets = {
        fact_id: {str(record["template_bundle_id"]) for record in records}
        for fact_id, records in by_fact.items()
    }
    expected_bundles = next(iter(bundle_sets.values()))
    if not expected_bundles or any(value != expected_bundles for value in bundle_sets.values()):
        raise ValueError("Every fact must have exactly the same screening template bundles")
    expected_count = len(facts) * len(expected_bundles)
    if len(screening) != expected_count:
        raise ValueError(f"Expected {expected_count} screening samples; found {len(screening)}")

    return {
        "fact_count": len(facts),
        "screening_count": len(screening),
        "template_bundles": sorted(expected_bundles),
        "by_relation": dict(Counter(str(fact["relation_id"]) for fact in facts)),
    }


def candidate_requests_for_batch(
    records: Sequence[Mapping[str, Any]],
    rendered_texts: Sequence[str],
) -> tuple[list[dict[str, str]], dict[str, list[tuple[str, str, str]]]]:
    requests: list[dict[str, str]] = []
    lookup: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for record, rendered in zip(records, rendered_texts):
        sample_id = str(record["sample_id"])
        seen: set[tuple[str, str]] = set()
        groups = (
            ("true", list(record["acceptable_true_answers"])),
            ("false", list(record["contrast_false_answers"])),
        )
        for kind, answers in groups:
            for index, answer in enumerate(answers):
                answer = str(answer)
                dedup_key = (kind, answer)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                key = f"{sample_id}|{kind}|{index}"
                requests.append(
                    {"key": key, "rendered_text": rendered, "continuation": answer}
                )
                lookup[sample_id].append((key, kind, answer))
    return requests, lookup


def evaluate_batch(
    records: Sequence[Mapping[str, Any]],
    bundle: Any,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rendered = [render_dataset_record(bundle.processor, record, config) for record in records]
    rendered_texts = [prompt.rendered_text for prompt in rendered]
    generations = generate_batch(bundle.model, bundle.tokenizer, rendered_texts, config)

    requests, request_lookup = candidate_requests_for_batch(records, rendered_texts)
    scores = score_continuations(
        bundle.model,
        bundle.tokenizer,
        requests,
        batch_size=int(config["screening"]["scoring_batch_size"]),
        max_input_tokens=int(config["chat"]["max_input_tokens"]),
    )
    score_by_key = {score.key: score for score in scores}
    threshold = float(config["screening"]["min_true_minus_false_logprob"])

    output: list[dict[str, Any]] = []
    for record, generation in zip(records, generations):
        sample_id = str(record["sample_id"])
        true_scores = [
            score_by_key[key]
            for key, kind, _answer in request_lookup[sample_id]
            if kind == "true"
        ]
        false_scores = [
            score_by_key[key]
            for key, kind, _answer in request_lookup[sample_id]
            if kind == "false"
        ]
        best_true = max(true_scores, key=lambda item: item.sum_logprob)
        strongest_false = max(false_scores, key=lambda item: item.sum_logprob)
        margin = best_true.sum_logprob - strongest_false.sum_logprob
        generated_correct = answer_matches(
            generation["text"],
            list(record["acceptable_true_answers"]),
        )
        parseable = is_one_orthographic_word(generation["text"])
        passed = generated_correct and parseable and margin > threshold

        output.append(
            {
                "code_version": CODE_VERSION,
                "sample_id": sample_id,
                "fact_id": record["fact_id"],
                "relation_id": record["relation_id"],
                "transfer_role": record["transfer_role"],
                "fact_split": record["fact_split"],
                "cv_fold": record["cv_fold"],
                "template_bundle_id": record["template_bundle_id"],
                "world_true_answer": record["world_true_answer"],
                "acceptable_true_answers": record["acceptable_true_answers"],
                "contrast_false_answers": record["contrast_false_answers"],
                "generated_answer": generation["text"],
                "generated_answer_raw": generation["raw_text"],
                "generated_answer_normalized": generation["normalized_text"],
                "generated_token_ids": generation["token_ids"],
                "generation_finish_reason": generation["finish_reason"],
                "generation_stop_token_id": generation["stop_token_id"],
                "generated_answer_is_one_word": parseable,
                "generated_answer_is_correct": generated_correct,
                "true_answer_scores": [score.as_dict() for score in true_scores],
                "false_answer_scores": [score.as_dict() for score in false_scores],
                "best_true_answer": best_true.continuation,
                "strongest_false_answer": strongest_false.continuation,
                "true_answer_sequence_logprob": best_true.sum_logprob,
                "strongest_false_answer_sequence_logprob": strongest_false.sum_logprob,
                "true_minus_false_logprob_margin": margin,
                "screening_pass": passed,
            }
        )
    return output


def aggregate_facts(
    facts: Sequence[Mapping[str, Any]],
    screening_results: Sequence[Mapping[str, Any]],
    require_all: bool,
) -> list[dict[str, Any]]:
    by_fact: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in screening_results:
        by_fact[str(result["fact_id"])].append(result)

    output: list[dict[str, Any]] = []
    for fact in facts:
        fact_id = str(fact["fact_id"])
        results = sorted(by_fact[fact_id], key=lambda item: str(item["template_bundle_id"]))
        if not results:
            raise ValueError(f"No screening results for {fact_id}")
        passes = [bool(result["screening_pass"]) for result in results]
        eligible = all(passes) if require_all else any(passes)
        output.append(
            {
                "code_version": CODE_VERSION,
                "fact_id": fact_id,
                "relation_id": fact["relation_id"],
                "transfer_role": fact["transfer_role"],
                "fact_split": fact["fact_split"],
                "cv_fold": fact["cv_fold"],
                "subject": fact["subject"],
                "world_true_answer": fact["world_true_answer"],
                "acceptable_true_answers": fact["acceptable_true_answers"],
                "model_parametric_knowledge_status": (
                    "screened_known" if eligible else "screened_not_reliably_known"
                ),
                "eligible": eligible,
                "required_template_bundles": [
                    result["template_bundle_id"] for result in results
                ],
                "passed_template_bundles": [
                    result["template_bundle_id"]
                    for result in results
                    if result["screening_pass"]
                ],
                "screening_sample_ids": [result["sample_id"] for result in results],
                "minimum_true_minus_false_logprob_margin": min(
                    float(result["true_minus_false_logprob_margin"])
                    for result in results
                ),
                "generated_answers": {
                    str(result["template_bundle_id"]): result["generated_answer"]
                    for result in results
                },
            }
        )
    return output


def run_preflight(
    screening: Sequence[Mapping[str, Any]],
    bundle: Any,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    france = next(
        (
            record
            for record in screening
            if record["fact_id"] == "country_capital:france"
            and record["template_bundle_id"] == "development"
        ),
        None,
    )
    if france is None:
        return None
    result = evaluate_batch([france], bundle, config)[0]
    if not result["generated_answer_is_correct"]:
        raise RuntimeError(
            "Gemma runtime preflight failed on the France/Paris screen: "
            f"generated {result['generated_answer']!r}. Verify Transformers version, "
            "model revision, chat template, and checkpoint integrity before running the study."
        )
    return {
        "sample_id": result["sample_id"],
        "generated_answer": result["generated_answer"],
        "true_minus_false_logprob_margin": result["true_minus_false_logprob_margin"],
    }


def main() -> None:
    started = time.time()
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["project"]["seed"]))
    dataset_dir = Path(config["paths"]["dataset_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    facts_path = dataset_dir / "facts.jsonl"
    screening_path = dataset_dir / "screening.jsonl"
    manifest_path = dataset_dir / "manifest.json"
    for path in (facts_path, screening_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    facts = read_jsonl(facts_path)
    screening = read_jsonl(screening_path)
    dataset_manifest = read_json(manifest_path)
    validate_manifest_file(dataset_manifest, "facts.jsonl", facts_path, len(facts))
    validate_manifest_file(
        dataset_manifest,
        "screening.jsonl",
        screening_path,
        len(screening),
    )
    summary = validate_inputs(facts, screening)
    if args.validate_only:
        print("Dataset validation passed:")
        print(summary)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / str(config["screening"]["results_file"])
    partial_path = results_path.with_name(results_path.name + ".partial")
    partial_metadata_path = partial_path.with_name(partial_path.name + ".meta.json")
    eligible_path = output_dir / str(config["screening"]["eligible_facts_file"])
    metadata_path = output_dir / str(config["screening"]["run_metadata_file"])
    owned_paths = (
        results_path,
        partial_path,
        partial_metadata_path,
        eligible_path,
        metadata_path,
    )
    if args.overwrite:
        for path in owned_paths:
            if path.exists():
                path.unlink()
    elif results_path.exists():
        raise FileExistsError(f"Results already exist: {results_path}; pass --overwrite to replace")
    elif eligible_path.exists() or metadata_path.exists():
        raise FileExistsError(
            "Found final screening outputs without the final results file; "
            "pass --overwrite to start a clean run"
        )

    signature_payload = {
        "code_version": CODE_VERSION,
        "model": config["model"],
        "chat": config["chat"],
        "generation": config["generation"],
        "screening": config["screening"],
        "datasets": {
            "facts.jsonl": file_sha256(facts_path),
            "screening.jsonl": file_sha256(screening_path),
            "manifest.json": file_sha256(manifest_path),
        },
    }
    checkpoint_signature = json_sha256(signature_payload)
    if partial_path.exists():
        if not partial_metadata_path.exists():
            raise FileNotFoundError(
                f"Checkpoint metadata is missing: {partial_metadata_path}; "
                "pass --overwrite rather than guessing compatibility"
            )
        checkpoint_metadata = read_json(partial_metadata_path)
        if checkpoint_metadata.get("input_signature") != checkpoint_signature:
            raise ValueError(
                "Partial screening checkpoint was produced from different inputs or settings; "
                "pass --overwrite to start a clean run"
            )
    elif partial_metadata_path.exists():
        checkpoint_metadata = read_json(partial_metadata_path)
        if checkpoint_metadata.get("input_signature") != checkpoint_signature:
            raise ValueError(
                "Empty screening checkpoint was produced from different inputs or settings; "
                "pass --overwrite to start a clean run"
            )
    else:
        write_json_atomic(
            partial_metadata_path,
            {
                "code_version": CODE_VERSION,
                "input_signature": checkpoint_signature,
                "signature_payload": signature_payload,
            },
        )

    completed = read_jsonl(partial_path) if partial_path.exists() else []
    unique_by(completed, "sample_id", str(partial_path))
    completed_ids = {str(record["sample_id"]) for record in completed}
    expected_ids = {str(record["sample_id"]) for record in screening}
    if not completed_ids.issubset(expected_ids):
        raise ValueError("Partial screening checkpoint does not match the current dataset")
    pending = [record for record in screening if str(record["sample_id"]) not in completed_ids]

    print(f"Loading {config['model']['id']} at revision {config['model']['revision']}")
    bundle = load_model_bundle(config)
    preflight = run_preflight(screening, bundle, config)
    if preflight:
        print(
            "Runtime preflight passed: "
            f"{preflight['generated_answer']!r}, margin={preflight['true_minus_false_logprob_margin']:.3f}"
        )

    batch_size = int(config["screening"]["batch_size"])
    report_every = int(config["runtime"].get("report_every_batches", 10))
    for batch_index, start in enumerate(range(0, len(pending), batch_size), start=1):
        batch = pending[start : start + batch_size]
        evaluated = evaluate_batch(batch, bundle, config)
        append_jsonl(partial_path, evaluated)
        if batch_index % report_every == 0 or start + len(batch) == len(pending):
            done = len(completed_ids) + start + len(batch)
            print(f"screening: {done}/{len(screening)} samples")

    all_results = read_jsonl(partial_path)
    unique_by(all_results, "sample_id", str(partial_path))
    if {str(record["sample_id"]) for record in all_results} != expected_ids:
        raise RuntimeError("Screening checkpoint is incomplete after processing")
    finalize_jsonl(partial_path, results_path)
    partial_metadata_path.unlink()

    eligible_facts = aggregate_facts(
        facts,
        all_results,
        require_all=bool(config["screening"]["require_all_template_bundles"]),
    )
    write_jsonl_atomic(eligible_path, eligible_facts)
    eligible_counts = Counter(
        str(record["relation_id"])
        for record in eligible_facts
        if record["eligible"]
    )
    run_metadata = runtime_fingerprint(
        config,
        bundle,
        {
            "facts.jsonl": facts_path,
            "screening.jsonl": screening_path,
            "manifest.json": manifest_path,
        },
    )
    run_metadata.update(
        {
            "script": Path(__file__).name,
            "elapsed_seconds": time.time() - started,
            "criteria": {
                "require_all_template_bundles": config["screening"][
                    "require_all_template_bundles"
                ],
                "min_true_minus_false_logprob": config["screening"][
                    "min_true_minus_false_logprob"
                ],
                "generation": "greedy",
                "sequence_score": "sum of answer-token log-probabilities; EOS excluded",
            },
            "preflight": preflight,
            "counts": {
                **summary,
                "eligible_total": sum(bool(record["eligible"]) for record in eligible_facts),
                "eligible_by_relation": dict(eligible_counts),
            },
            "outputs": {
                "screening_results": str(results_path),
                "eligible_facts": str(eligible_path),
            },
        }
    )
    write_json_atomic(metadata_path, run_metadata)
    print(
        f"Eligible facts: {run_metadata['counts']['eligible_total']}/{len(facts)} "
        f"({dict(eligible_counts)})"
    )
    print(f"Wrote {results_path}")
    print(f"Wrote {eligible_path}")


if __name__ == "__main__":
    main()
