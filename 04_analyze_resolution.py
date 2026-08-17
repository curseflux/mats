#!/usr/bin/env python3
"""Analyze how Gemma resolves conflict, without reloading the model.

All inferential resampling is clustered by query fact.  Policy effects are
paired within identical content, and truth-by-relevance interactions are
paired within complete factorial groups.  This script is observational: it
describes resolution and its association with probe scores, but does not turn
an association into a causal claim.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from common import (
    file_sha256,
    load_config,
    read_json,
    read_jsonl,
    unique_by,
    write_json_atomic,
)


ANALYSIS_VERSION = "1.0.0"
DEFAULT_SPLITS = (
    "validation",
    "id_test",
    "paraphrase_test",
    "ood_relation",
    "ood_relation_paraphrase",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--behavior", type=Path, default=None)
    parser.add_argument("--probe-scores", type=Path, default=None)
    parser.add_argument("--probe-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--analysis-splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated splits to report. Training is excluded by default.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_paths(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[Path, Path, Path, Path]:
    root = Path(config["paths"]["output_dir"])
    probe_root = root / "analysis" / "probes"
    behavior = args.behavior or root / str(config["collection"]["behavior_results_file"])
    scores = args.probe_scores or probe_root / "probe_scores.jsonl"
    metadata = args.probe_metadata or probe_root / "probe_metadata.json"
    output = args.output_dir or root / "analysis" / "resolution"
    return behavior.resolve(), scores.resolve(), metadata.resolve(), output.resolve()


def parse_split_list(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--analysis-splits must be a non-empty unique list")
    return values


def merge_rows(
    behavior: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    unique_by(behavior, "sample_id", "behavior results")
    unique_by(scores, "sample_id", "probe scores")
    score_by_id = {str(row["sample_id"]): row for row in scores}
    behavior_ids = [str(row["sample_id"]) for row in behavior]
    if set(behavior_ids) != set(score_by_id):
        missing = sorted(set(behavior_ids).difference(score_by_id))[:5]
        extra = sorted(set(score_by_id).difference(behavior_ids))[:5]
        raise ValueError(f"Behavior/probe IDs differ; missing={missing}, extra={extra}")

    merged: list[dict[str, Any]] = []
    consistency_fields = (
        "fact_id",
        "claim_fact_id",
        "matched_factorial_group_id",
        "content_pair_id",
        "relation_id",
        "fact_split",
        "template_bundle_id",
        "condition_id",
        "policy_id",
    )
    for behavior_row in behavior:
        score_row = score_by_id[str(behavior_row["sample_id"])]
        disagreements = [
            field
            for field in consistency_fields
            if str(behavior_row.get(field)) != str(score_row.get(field))
        ]
        if disagreements:
            raise ValueError(
                f"Metadata disagreement for {behavior_row['sample_id']}: {disagreements}"
            )
        row = dict(behavior_row)
        for key, value in score_row.items():
            if key not in row or key.endswith(("_logit", "_probability")) or key == "analysis_split":
                row[key] = value
        merged.append(row)
    return merged


def finite_values(values: Sequence[Any]) -> Any:
    import numpy as np

    output = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            output.append(number)
    return np.asarray(output, dtype=np.float64)


def mean_or_none(values: Sequence[Any]) -> float | None:
    import numpy as np

    valid = finite_values(values)
    return float(np.mean(valid)) if len(valid) else None


def rate_or_none(values: Sequence[Any]) -> float | None:
    valid = [bool(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def safe_auc(y_true: Any, score: Any) -> float | None:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(score)
    if len(np.unique(y_true[valid])) != 2:
        return None
    return float(roc_auc_score(y_true[valid], score[valid]))


def percentile_interval(values: Sequence[float]) -> tuple[float | None, float | None]:
    import numpy as np

    valid = finite_values(values)
    if not len(valid):
        return None, None
    lower, upper = np.percentile(valid, [2.5, 97.5])
    return float(lower), float(upper)


def bootstrap_statistic(
    clusters: Sequence[str],
    statistic: Callable[[Any], float | None],
    replicates: int,
    rng: Any,
) -> dict[str, Any]:
    import numpy as np

    clusters = np.asarray(clusters)
    unique = np.asarray(sorted(set(str(value) for value in clusters)))
    if len(unique) < 2:
        raise ValueError("Cluster bootstrap requires at least two query facts")
    indices_by_cluster = {
        cluster: np.flatnonzero(clusters.astype(str) == cluster) for cluster in unique
    }
    observed = statistic(np.arange(len(clusters), dtype=np.int64))
    draws: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([indices_by_cluster[str(cluster)] for cluster in sampled])
        value = statistic(indices)
        if value is not None and np.isfinite(value):
            draws.append(float(value))
    lower, upper = percentile_interval(draws)
    return {
        "estimate": float(observed) if observed is not None else None,
        "ci95_low": lower,
        "ci95_high": upper,
        "bootstrap_valid_replicates": len(draws),
        "cluster_count": int(len(unique)),
    }


def behavior_table(rows: Sequence[Mapping[str, Any]], splits: set[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row["analysis_split"])
        if split in splits:
            groups[(split, str(row["condition_id"]), str(row["policy_id"]))].append(row)

    output: list[dict[str, Any]] = []
    for (split, condition, policy), group in sorted(groups.items()):
        distinct = [row for row in group if bool(row["claim_and_parametric_answers_are_distinct"])]
        sources = Counter(str(row["observed_knowledge_source"]) for row in group)
        output.append(
            {
                "analysis_split": split,
                "condition_id": condition,
                "policy_id": policy,
                "n": len(group),
                "unique_facts": len({str(row["fact_id"]) for row in group}),
                "generated_parametric_rate": rate_or_none(
                    [row["generated_matches_parametric"] for row in group]
                ),
                "generated_context_rate": rate_or_none(
                    [row["generated_matches_context"] for row in group]
                ),
                "distinct_candidate_n": len(distinct),
                "distinct_generated_parametric_rate": rate_or_none(
                    [row["generated_matches_parametric"] for row in distinct]
                ),
                "distinct_generated_context_rate": rate_or_none(
                    [row["generated_matches_context"] for row in distinct]
                ),
                "other_rate": sources["other"] / len(group),
                "unparseable_rate": sources["unparseable"] / len(group),
                "irrelevant_claim_rate": sources["irrelevant_claim"] / len(group),
                "policy_compliance_rate": rate_or_none(
                    [row.get("policy_compliant") for row in group]
                ),
                "mean_context_minus_parametric_margin": mean_or_none(
                    [row.get("context_minus_parametric_logprob_margin") for row in group]
                ),
                "mean_parametric_answer_logprob": mean_or_none(
                    [row.get("parametric_answer_sequence_logprob") for row in group]
                ),
                "mean_awareness_probe_logit": mean_or_none(
                    [row.get("awareness_probe_logit") for row in group]
                ),
            }
        )
    return output


def effect_summary(
    values: Sequence[float],
    facts: Sequence[str],
    replicates: int,
    rng: Any,
) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if len(array) != len(facts) or not len(array):
        raise ValueError("Effect vector is empty or misaligned")
    result = bootstrap_statistic(
        facts,
        lambda indices: float(np.mean(array[indices])),
        replicates,
        rng,
    )
    result["n_pairs"] = len(array)
    return result


def paired_policy_effects(
    rows: Sequence[Mapping[str, Any]],
    splits: set[str],
    replicates: int,
    rng: Any,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if (
            str(row["analysis_split"]) in splits
            and str(row["condition_id"]) == "false_relevant"
        ):
            key = (str(row["analysis_split"]), str(row["content_pair_id"]))
            policy = str(row["policy_id"])
            if policy in groups[key]:
                raise ValueError(f"Duplicate policy in content pair {key}")
            groups[key][policy] = row

    comparisons = (
        ("context_minus_neutral", "context", "neutral"),
        ("parametric_minus_neutral", "parametric", "neutral"),
        ("context_minus_parametric", "context", "parametric"),
    )
    outcomes: tuple[tuple[str, Callable[[Mapping[str, Any]], float | None]], ...] = (
        (
            "context_minus_parametric_logprob_margin",
            lambda row: (
                float(row["context_minus_parametric_logprob_margin"])
                if row.get("context_minus_parametric_logprob_margin") is not None
                else None
            ),
        ),
        ("generated_matches_context", lambda row: float(bool(row["generated_matches_context"]))),
        ("generated_matches_parametric", lambda row: float(bool(row["generated_matches_parametric"]))),
        ("awareness_probe_logit", lambda row: float(row["awareness_probe_logit"])),
    )
    output: list[dict[str, Any]] = []
    for split in sorted(splits):
        split_groups = [policies for (name, _), policies in groups.items() if name == split]
        for comparison, high, low in comparisons:
            for outcome, getter in outcomes:
                values: list[float] = []
                facts: list[str] = []
                for policies in split_groups:
                    if high not in policies or low not in policies:
                        continue
                    high_value, low_value = getter(policies[high]), getter(policies[low])
                    if high_value is None or low_value is None:
                        continue
                    if str(policies[high]["fact_id"]) != str(policies[low]["fact_id"]):
                        raise ValueError("A content pair changes query fact")
                    values.append(high_value - low_value)
                    facts.append(str(policies[high]["fact_id"]))
                if values:
                    output.append(
                        {
                            "analysis_type": "paired_policy_effect",
                            "analysis_split": split,
                            "condition_id": "false_relevant",
                            "comparison": comparison,
                            "outcome": outcome,
                            **effect_summary(values, facts, replicates, rng),
                        }
                    )
    return output


def factorial_effects(
    rows: Sequence[Mapping[str, Any]],
    splits: set[str],
    replicates: int,
    rng: Any,
) -> list[dict[str, Any]]:
    import numpy as np

    groups: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        split = str(row["analysis_split"])
        if split not in splits:
            continue
        key = (split, str(row["matched_factorial_group_id"]), str(row["policy_id"]))
        condition = str(row["condition_id"])
        if condition in groups[key]:
            raise ValueError(f"Duplicate factorial condition in {key}")
        groups[key][condition] = row

    required = {"false_relevant", "true_relevant", "false_irrelevant", "true_irrelevant"}
    output: list[dict[str, Any]] = []
    for split in sorted(splits):
        for policy in ("neutral", "context", "parametric"):
            members = [
                conditions
                for (name, _, policy_id), conditions in groups.items()
                if name == split and policy_id == policy
            ]
            if any(set(group) != required for group in members):
                raise ValueError(f"Incomplete factorial group in {split}/{policy}")
            if not members:
                continue

            for outcome in (
                "parametric_answer_sequence_logprob",
                "generated_matches_parametric",
            ):
                values, facts = [], []
                for group in members:
                    false_relevant = group["false_relevant"]
                    true_relevant = group["true_relevant"]
                    if outcome == "generated_matches_parametric":
                        value = float(bool(false_relevant[outcome])) - float(
                            bool(true_relevant[outcome])
                        )
                    else:
                        value = float(false_relevant[outcome]) - float(true_relevant[outcome])
                    values.append(value)
                    facts.append(str(false_relevant["fact_id"]))
                output.append(
                    {
                        "analysis_type": "conflict_cost",
                        "analysis_split": split,
                        "condition_id": "false_relevant_minus_true_relevant",
                        "comparison": f"within_{policy}_policy",
                        "outcome": outcome,
                        **effect_summary(values, facts, replicates, rng),
                    }
                )

            # Difference-in-differences isolates the non-additive truth ×
            # relevance component: FR - TR - FI + TI.
            for outcome in (
                "awareness_probe_logit",
                "factorized_awareness_probability",
            ):
                values, facts = [], []
                for group in members:
                    transformed: dict[str, float] = {}
                    for condition, row in group.items():
                        value = float(row[outcome])
                        if outcome.endswith("_probability"):
                            value = float(np.log(np.clip(value, 1e-8, 1 - 1e-8) / np.clip(1 - value, 1e-8, 1)))
                        transformed[condition] = value
                    values.append(
                        transformed["false_relevant"]
                        - transformed["true_relevant"]
                        - transformed["false_irrelevant"]
                        + transformed["true_irrelevant"]
                    )
                    facts.append(str(group["false_relevant"]["fact_id"]))
                output.append(
                    {
                        "analysis_type": "truth_x_relevance_interaction",
                        "analysis_split": split,
                        "condition_id": "FR_minus_TR_minus_FI_plus_TI",
                        "comparison": f"within_{policy}_policy",
                        "outcome": outcome,
                        **effect_summary(values, facts, replicates, rng),
                    }
                )
    return output


def auc_comparisons(
    rows: Sequence[Mapping[str, Any]],
    splits: set[str],
    replicates: int,
    rng: Any,
) -> list[dict[str, Any]]:
    import numpy as np

    output: list[dict[str, Any]] = []
    for split in sorted(splits):
        split_group = [row for row in rows if str(row["analysis_split"]) == split]
        for policy in ("all", "neutral", "context", "parametric"):
            group = (
                split_group
                if policy == "all"
                else [row for row in split_group if str(row["policy_id"]) == policy]
            )
            y = np.asarray([bool(row["awareness_label"]) for row in group], dtype=np.int8)
            awareness = np.asarray(
                [float(row["awareness_probe_probability"]) for row in group]
            )
            factorized = np.asarray(
                [float(row["factorized_awareness_probability"]) for row in group]
            )
            facts = [str(row["fact_id"]) for row in group]

            def statistic(which: str) -> Callable[[Any], float | None]:
                def calculate(indices: Any) -> float | None:
                    direct = safe_auc(y[indices], awareness[indices])
                    baseline = safe_auc(y[indices], factorized[indices])
                    if which == "awareness":
                        return direct
                    if which == "factorized":
                        return baseline
                    return (
                        direct - baseline
                        if direct is not None and baseline is not None
                        else None
                    )
                return calculate

            direct = bootstrap_statistic(facts, statistic("awareness"), replicates, rng)
            factor = bootstrap_statistic(facts, statistic("factorized"), replicates, rng)
            delta = bootstrap_statistic(facts, statistic("delta"), replicates, rng)
            output.append(
                {
                    "analysis_split": split,
                    "policy_id": policy,
                    "n": len(group),
                    "unique_facts": len(set(facts)),
                    "awareness_probe": direct,
                    "factorized_baseline": factor,
                    "awareness_minus_factorized": delta,
                }
            )
    return output


def resolution_associations(
    rows: Sequence[Mapping[str, Any]],
    splits: set[str],
    replicates: int,
    rng: Any,
) -> list[dict[str, Any]]:
    import numpy as np
    import warnings
    from scipy.stats import spearmanr

    output: list[dict[str, Any]] = []
    for split in sorted(splits):
        group = [
            row
            for row in rows
            if str(row["analysis_split"]) == split
            and str(row["condition_id"]) == "false_relevant"
            and str(row["policy_id"]) == "neutral"
            and row.get("context_minus_parametric_logprob_margin") is not None
        ]
        if not group:
            continue
        awareness = np.asarray([float(row["awareness_probe_logit"]) for row in group])
        margin = np.asarray(
            [float(row["context_minus_parametric_logprob_margin"]) for row in group]
        )
        context_choice = np.asarray(
            [bool(row["generated_matches_context"]) for row in group], dtype=np.int8
        )
        parametric_choice = np.asarray(
            [bool(row["generated_matches_parametric"]) for row in group], dtype=np.int8
        )
        facts = [str(row["fact_id"]) for row in group]

        def spearman(indices: Any) -> float | None:
            if len(indices) < 3:
                return None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = spearmanr(awareness[indices], margin[indices])
            value = float(getattr(result, "statistic", result[0]))
            return value if np.isfinite(value) else None

        statistics = {
            "spearman_awareness_logit_vs_resolution_margin": spearman,
            "auc_awareness_predicts_context_generation": lambda indices: safe_auc(
                context_choice[indices], awareness[indices]
            ),
            "auc_awareness_predicts_parametric_generation": lambda indices: safe_auc(
                parametric_choice[indices], awareness[indices]
            ),
        }
        for name, statistic in statistics.items():
            output.append(
                {
                    "analysis_split": split,
                    "subset": "false_relevant/neutral",
                    "statistic": name,
                    "n": len(group),
                    **bootstrap_statistic(facts, statistic, replicates, rng),
                }
            )
    return output


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
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


def fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def ci_text(result: Mapping[str, Any]) -> str:
    return (
        f"{fmt(result.get('estimate'))} "
        f"[{fmt(result.get('ci95_low'))}, {fmt(result.get('ci95_high'))}]"
    )


def render_report(summary: Mapping[str, Any]) -> str:
    splits = summary["analysis_splits"]
    auc_rows = summary["auc_comparisons"]
    effects = summary["paired_effects"]
    associations = summary["resolution_associations"]

    lines = [
        "# Conflict awareness and resolution",
        "",
        "This report keeps two questions separate: **is conflict linearly represented?** "
        "and **which answer does the model select?** Validation was used for model selection; "
        "the ID, paraphrase, and relation-transfer rows are the held-out evidence.",
        "",
        "## Awareness probe versus factorized baseline",
        "",
        "| Split | Policy | Awareness AUC (95% CI) | Falsehood × relevance AUC (95% CI) | AUC difference (95% CI) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in auc_rows:
        lines.append(
            f"| {row['analysis_split']} | {row['policy_id']} | {ci_text(row['awareness_probe'])} | "
            f"{ci_text(row['factorized_baseline'])} | "
            f"{ci_text(row['awareness_minus_factorized'])} |"
        )

    policy_margin = [
        row for row in effects
        if row["analysis_type"] == "paired_policy_effect"
        and row["outcome"] == "context_minus_parametric_logprob_margin"
        and row["comparison"] in {"context_minus_neutral", "parametric_minus_neutral"}
    ]
    lines.extend(
        [
            "",
            "## Policy manipulation check",
            "",
            "Positive values move likelihood toward the contextual answer; negative values move "
            "it toward the parametric answer. Each contrast is paired within identical content.",
            "",
            "| Split | Contrast | Mean margin shift (95% CI) | Pairs |",
            "|---|---|---:|---:|",
        ]
    )
    for row in policy_margin:
        lines.append(
            f"| {row['analysis_split']} | {row['comparison']} | {ci_text(row)} | {row['n_pairs']} |"
        )

    interactions = [
        row for row in effects
        if row["analysis_type"] == "truth_x_relevance_interaction"
        and row["outcome"] == "awareness_probe_logit"
        and row["comparison"] == "within_neutral_policy"
    ]
    lines.extend(
        [
            "",
            "## Factorial conflict interaction",
            "",
            "The contrast is `false-relevant − true-relevant − false-irrelevant + true-irrelevant`. "
            "A positive, transferable contrast is evidence that the readout is not merely a generic "
            "falsehood or relevance detector.",
            "",
            "| Split | Awareness-logit interaction (95% CI) | Groups |",
            "|---|---:|---:|",
        ]
    )
    for row in interactions:
        lines.append(
            f"| {row['analysis_split']} | {ci_text(row)} | {row['n_pairs']} |"
        )

    lines.extend(
        [
            "",
            "## Awareness–resolution association under neutral instructions",
            "",
            "These statistics use only false-relevant neutral prompts. They are descriptive, not "
            "causal: a probe score can track answer selection without controlling it.",
            "",
            "| Split | Statistic | Estimate (95% CI) | N |",
            "|---|---|---:|---:|",
        ]
    )
    for row in associations:
        lines.append(
            f"| {row['analysis_split']} | {row['statistic']} | {ci_text(row)} | {row['n']} |"
        )

    context_good = [
        row for row in policy_margin
        if row["comparison"] == "context_minus_neutral"
        and row["ci95_low"] is not None and row["ci95_low"] > 0
    ]
    parametric_good = [
        row for row in policy_margin
        if row["comparison"] == "parametric_minus_neutral"
        and row["ci95_high"] is not None and row["ci95_high"] < 0
    ]
    interaction_good = [
        row for row in interactions
        if row["ci95_low"] is not None and row["ci95_low"] > 0
    ]
    auc_advantage = [
        row for row in auc_rows
        if row["policy_id"] == "neutral"
        and row["awareness_minus_factorized"]["ci95_low"] is not None
        and row["awareness_minus_factorized"]["ci95_low"] > 0
    ]
    lines.extend(
        [
            "",
            "## What to look for",
            "",
            f"- Context instructions reliably increase the resolution margin in {len(context_good)}/{len(splits)} reported splits; "
            f"parametric instructions reliably decrease it in {len(parametric_good)}/{len(splits)}.",
            f"- The neutral-policy factorial interaction is reliably positive in {len(interaction_good)}/{len(splits)} splits.",
            f"- Under neutral instructions, the direct awareness probe reliably exceeds the factorized baseline in {len(auc_advantage)}/{len(splits)} splits.",
            "- Prioritize consistent held-out and OOD effects over a single high validation number. "
            "A confidence interval crossing zero is inconclusive, not evidence of no effect.",
            "- If the policy anchors do not move the likelihood margin in opposite directions, do "
            "not interpret the neutral condition as meaningful arbitration yet.",
            "- If awareness does not beat the factorized baseline, the representation may still encode "
            "conflict, but the result does not establish a distinct non-additive awareness feature.",
            "- Run `05_causal_dissociation.py` before making mechanistic claims. Probe accuracy alone "
            "establishes decodability, not causal use.",
            "",
            "All 95% intervals are percentile cluster-bootstrap intervals over query facts.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    import numpy as np

    started = time.time()
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("--bootstrap-replicates must be at least 100")
    config = load_config(args.config)
    behavior_path, scores_path, metadata_path, output_dir = resolve_paths(args, config)
    for path in (behavior_path, scores_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    requested_splits = parse_split_list(args.analysis_splits)
    seed = int(args.seed if args.seed is not None else config["project"]["seed"])
    rng = np.random.default_rng(seed)

    output_paths = {
        "behavior": output_dir / "behavior_summary.csv",
        "effects": output_dir / "paired_effects.csv",
        "associations": output_dir / "awareness_resolution_association.csv",
        "summary": output_dir / "resolution_summary.json",
        "report": output_dir / "resolution_report.md",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Analysis outputs already exist; pass --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    behavior = read_jsonl(behavior_path)
    scores = read_jsonl(scores_path)
    probe_metadata = read_json(metadata_path)
    rows = merge_rows(behavior, scores)
    available_splits = {str(row["analysis_split"]) for row in rows}
    unknown = sorted(set(requested_splits).difference(available_splits))
    if unknown:
        raise ValueError(f"Requested analysis splits are unavailable: {unknown}")
    split_set = set(requested_splits)

    behavior_rows = behavior_table(rows, split_set)
    policy_rows = paired_policy_effects(
        rows, split_set, args.bootstrap_replicates, rng
    )
    factorial_rows = factorial_effects(
        rows, split_set, args.bootstrap_replicates, rng
    )
    effect_rows = policy_rows + factorial_rows
    auc_rows = auc_comparisons(rows, split_set, args.bootstrap_replicates, rng)
    association_rows = resolution_associations(
        rows, split_set, args.bootstrap_replicates, rng
    )

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "script": Path(__file__).name,
        "elapsed_seconds": time.time() - started,
        "analysis_splits": requested_splits,
        "selection_warning": (
            "validation was used to select layer, position-restricted layer, and C; "
            "treat validation statistics as model-selection diagnostics"
        ),
        "uncertainty": {
            "method": "percentile cluster bootstrap over query fact_id",
            "replicates": args.bootstrap_replicates,
            "seed": seed,
            "unit_of_analysis": "paired content/group contrast where applicable",
        },
        "inputs": {
            "behavior": str(behavior_path),
            "behavior_sha256": file_sha256(behavior_path),
            "probe_scores": str(scores_path),
            "probe_scores_sha256": file_sha256(scores_path),
            "probe_metadata": str(metadata_path),
            "probe_metadata_sha256": file_sha256(metadata_path),
            "selected_cell": probe_metadata.get("selection", {}).get("selected_cell"),
        },
        "record_counts": dict(Counter(str(row["analysis_split"]) for row in rows)),
        "auc_comparisons": auc_rows,
        "paired_effects": effect_rows,
        "resolution_associations": association_rows,
        "interpretive_limits": [
            "Probe performance establishes linear decodability, not subjective awareness.",
            "Awareness-resolution association is observational and may reflect a common cause.",
            "Generated-answer rates are secondary to exact candidate sequence-log-probability margins.",
            "Rows sharing a query fact are not treated as independent observations.",
        ],
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }

    write_csv_atomic(output_paths["behavior"], behavior_rows)
    write_csv_atomic(output_paths["effects"], effect_rows)
    write_csv_atomic(output_paths["associations"], association_rows)
    write_json_atomic(output_paths["summary"], summary)
    write_text_atomic(output_paths["report"], render_report(summary))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
