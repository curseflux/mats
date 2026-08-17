#!/usr/bin/env python3
"""Q4, redone with fact-disjoint folds and the baselines it has to beat.

The original Q4 in `projection_analysis.py` fitted a probe on template A and
evaluated it on template B.  Its off-diagonal AUCs looked strong (0.75-1.00),
but train and test shared the same facts: a probe that memorised "Hydrogen is a
context-prone item" scores well off-diagonal without encoding anything about
context reliance.  The diagonal was also scored on training rows, so the table
had no honest reference point.

This script asks the same question under conditions that can actually answer it:

    Fit on (template A, facts in the training folds).
    Evaluate on (template B, facts in the held-out fold).

Nothing about a test fact is ever seen during fitting, in any template.  Every
number is an out-of-fold prediction, pooled across folds and scored once.

Three baselines run on exactly the same held-out rows:

  parametric_confidence   -logP(true answer), read off a NON-conflicting prompt
                          for the same fact and template.  A per-fact knowledge
                          property, stable across templates, and therefore the
                          confound most likely to fake a transfer result.
  resolution_projection   projection onto the fixed `resolution_raw` direction
                          from step 05, i.e. no fitting at all.
  shuffled_labels         the same probe fitted after permuting training labels
                          within template.  Should land at 0.5; if it does not,
                          the evaluation is leaking.

Read the result this way:

  probe >> parametric_confidence, holding up off-diagonal
      -> there is a template-invariant context-reliance direction that is not
         just knowledge strength.  Worth characterising.
  probe ~= parametric_confidence
      -> the probe is reading how well the model knows the fact.  The transfer
         result is not about context reliance and should not be reported as if
         it were.
  probe collapses to ~0.5 off-diagonal
      -> the original Q4 was fact memorisation.

No GPU.  Reads cached activations, the fitted probe, and the steering directions.

Usage
-----
python fact_disjoint_transfer.py \
    --behavior   results/gemma4_12b_conflict/behavior_results.jsonl \
    --manifest   /scratch/bbjr/skarmakar/conflict/activations/manifest.json \
    --probe      results/gemma4_12b_conflict/analysis/probes/probe_model.npz \
    --directions results/gemma4_12b_conflict/analysis/causal/causal_directions.npz \
    --out        results/gemma4_12b_conflict/analysis/fact_disjoint
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


BUNDLES = ("development", "validation", "heldout_paraphrase")

# The conflict condition under no answer-source instruction: the only cell where
# the model is actually choosing between the document and its memory.
CONDITION = "false_relevant"
POLICY = "neutral"

# A matched prompt for the same fact and template in which the paragraph makes a
# true claim about an unrelated subject.  The model's confidence in its own
# answer here is uncontaminated by the conflict.
BASELINE_CONDITION = "true_irrelevant"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scalar(array):
    """npz round-trips 0-d values as shape-(1,) arrays; unwrap either form."""
    array = np.asarray(array)
    return array.reshape(-1)[0] if array.size == 1 else array


def load_selected_activations(
    manifest_path: Path,
    rows: list[dict],
    layer_offset: int,
    position_offset: int,
) -> np.ndarray:
    """Return (n_rows, hidden) for one layer/position, in behaviour-file order."""
    import torch

    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise RuntimeError("activation manifest reports an incomplete run")
    if int(manifest["expected_samples"]) != len(rows):
        raise ValueError(
            f"manifest expects {manifest['expected_samples']} samples, "
            f"behaviour file has {len(rows)}"
        )

    out: np.ndarray | None = None
    cursor = 0
    for entry in manifest["shards"]:
        start, end = int(entry["start_index"]), int(entry["end_index"])
        if start != cursor:
            raise ValueError("activation shards are not contiguous")
        path = manifest_path.parent / str(entry["file"])
        try:
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=True)

        ids = [str(v) for v in payload["sample_ids"]]
        if ids != [str(r["sample_id"]) for r in rows[start:end]]:
            raise ValueError(f"sample ids inside {path} are misordered")

        block = payload["activations"][:, layer_offset, position_offset, :]
        block = block.float().numpy()
        if out is None:
            out = np.empty((len(rows), block.shape[1]), dtype=np.float32)
        out[start:end] = block
        cursor = end
        del payload, block

    if cursor != len(rows):
        raise ValueError("shards do not cover every behaviour row")
    return out


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC with explicit tie handling."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cluster_bootstrap_auc(
    fact_ids: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile CI resampling whole facts, never individual rows."""
    by_fact: dict[str, list[int]] = collections.defaultdict(list)
    for index, fact in enumerate(fact_ids):
        by_fact[fact].append(index)
    keys = list(by_fact)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        picked: list[int] = []
        for key in rng.choice(len(keys), size=len(keys), replace=True):
            picked.extend(by_fact[keys[key]])
        index = np.asarray(picked)
        value = auc(labels[index], scores[index])
        if value == value:
            draws.append(value)
    if not draws:
        return float("nan"), float("nan")
    draws.sort()
    return (
        draws[int(0.025 * len(draws))],
        draws[min(int(0.975 * len(draws)), len(draws) - 1)],
    )


def fold_of(fact_id: str, n_folds: int, seed: int) -> int:
    """Deterministic fact -> fold, independent of ordering and of relation."""
    digest = hashlib.sha256(f"{seed}:{fact_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_folds


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def fit_probe(x_train, y_train, c_grid, seed, inner_facts=None):
    """L2 logistic probe.  With >1 C, select on an inner fact-disjoint split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def _fit(x, y, c_value):
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
        model.fit(scaler.transform(x), y)
        return scaler, model

    if len(c_grid) == 1 or inner_facts is None:
        return _fit(x_train, y_train, c_grid[0]) + (float(c_grid[0]),)

    unique = sorted(set(inner_facts))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    holdout = set(unique[: max(1, len(unique) // 5)])
    inner_mask = np.asarray([f in holdout for f in inner_facts])
    if inner_mask.all() or not inner_mask.any():
        return _fit(x_train, y_train, c_grid[0]) + (float(c_grid[0]),)
    if len(np.unique(y_train[~inner_mask])) < 2 or len(np.unique(y_train[inner_mask])) < 2:
        return _fit(x_train, y_train, c_grid[0]) + (float(c_grid[0]),)

    best = (-np.inf, c_grid[0])
    for c_value in c_grid:
        scaler, model = _fit(x_train[~inner_mask], y_train[~inner_mask], c_value)
        score = auc(
            y_train[inner_mask],
            model.decision_function(scaler.transform(x_train[inner_mask])),
        )
        if score == score and score > best[0]:
            best = (score, c_value)
    scaler, model = _fit(x_train, y_train, best[1])
    return scaler, model, float(best[1])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--directions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c-grid", default="0.001,0.01,0.1")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--min-class",
        type=int,
        default=5,
        help="Skip any evaluation cell with fewer than this many of either class.",
    )
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    c_grid = [float(v) for v in args.c_grid.split(",") if v.strip()]
    if not c_grid or any(v <= 0 for v in c_grid):
        raise ValueError("--c-grid must be positive numbers")

    rows = read_jsonl(args.behavior)
    probe = np.load(args.probe)
    layer_offset = int(scalar(probe["selected_layer_offset"]))
    position_offset = int(scalar(probe["selected_position_index"]))
    print(
        f"site: layer {int(scalar(probe['selected_layer_index']))}, "
        f"position {str(scalar(probe['selected_position_name']))}"
    )

    resolution_direction = None
    if args.directions is not None and args.directions.exists():
        directions = np.load(args.directions)
        if "direction__resolution_raw" in directions:
            vector = np.asarray(directions["direction__resolution_raw"], dtype=np.float64)
            resolution_direction = vector / np.linalg.norm(vector)

    # -- the conflict trials, and a matched non-conflicting prompt per (fact, template)
    keep = [
        i
        for i, r in enumerate(rows)
        if r["condition_id"] == CONDITION and r["policy_id"] == POLICY
    ]
    if not keep:
        raise ValueError(f"no {CONDITION}/{POLICY} rows in the behaviour file")
    sub = [rows[i] for i in keep]

    confidence: dict[tuple[str, str], float] = {}
    for r in rows:
        if r["condition_id"] == BASELINE_CONDITION and r["policy_id"] == POLICY:
            value = r.get("parametric_answer_sequence_logprob")
            if value is not None:
                confidence[(str(r["fact_id"]), str(r["template_bundle_id"]))] = float(value)

    print("loading activations (this reads the shards once)...")
    activations = load_selected_activations(
        args.manifest, rows, layer_offset, position_offset
    )
    x = activations[keep].astype(np.float64)
    del activations

    facts = np.asarray([str(r["fact_id"]) for r in sub])
    relation = np.asarray([str(r["relation_id"]) for r in sub])
    template = np.asarray([str(r["template_bundle_id"]) for r in sub])
    followed = np.asarray([bool(r["generated_matches_context"]) for r in sub])
    margin = np.asarray(
        [
            float(r["context_minus_parametric_logprob_margin"])
            if r.get("context_minus_parametric_logprob_margin") is not None
            else np.nan
            for r in sub
        ]
    )
    folds = np.asarray([fold_of(f, args.folds, args.seed) for f in facts])
    print(f"  {x.shape[0]} conflict trials, hidden size {x.shape[1]}\n")

    report: dict = {
        "config": {
            "condition": CONDITION,
            "policy": POLICY,
            "baseline_condition": BASELINE_CONDITION,
            "folds": args.folds,
            "c_grid": c_grid,
            "seed": args.seed,
            "min_class": args.min_class,
        },
        "cells": [],
        "cross_relation": [],
        "direction_similarity": [],
    }

    # -----------------------------------------------------------------
    # main table: fit on (train template, train folds), test on (test
    # template, held-out fold).  Predictions are pooled out-of-fold.
    # -----------------------------------------------------------------
    header = "train -> test"
    print("=" * 108)
    print("FACT-DISJOINT TRANSFER    (out-of-fold; no test fact is ever seen in fitting)")
    print("=" * 108)
    print(
        f"{header:<34}{'probe AUC':>12}{'95% CI':>18}"
        f"{'confid.':>10}{'resol.':>9}{'shuffled':>10}{'pos/neg':>12}"
    )

    for rel in sorted(set(relation)):
        print(f"\n--- {rel}")
        for train_bundle in BUNDLES:
            for test_bundle in BUNDLES:
                test_mask = (relation == rel) & (template == test_bundle)
                if not test_mask.any():
                    continue
                test_index = np.flatnonzero(test_mask)

                scores = np.full(len(test_index), np.nan)
                shuffled_scores = np.full(len(test_index), np.nan)
                chosen_c: list[float] = []

                for fold in range(args.folds):
                    train_mask = (
                        (relation == rel)
                        & (template == train_bundle)
                        & (folds != fold)
                    )
                    local_test = folds[test_index] == fold
                    if not local_test.any() or not train_mask.any():
                        continue
                    y_train = followed[train_mask]
                    if len(np.unique(y_train)) < 2:
                        continue

                    x_train = x[train_mask]
                    scaler, model, c_value = fit_probe(
                        x_train,
                        y_train,
                        c_grid,
                        args.seed + fold,
                        inner_facts=list(facts[train_mask]),
                    )
                    chosen_c.append(c_value)
                    rows_to_score = x[test_index[local_test]]
                    scores[local_test] = model.decision_function(
                        scaler.transform(rows_to_score)
                    )

                    rng = np.random.default_rng(args.seed + 977 * fold)
                    y_shuffled = y_train.copy()
                    rng.shuffle(y_shuffled)
                    if len(np.unique(y_shuffled)) > 1:
                        s_scaler, s_model, _ = fit_probe(
                            x_train, y_shuffled, [c_grid[0]], args.seed + fold
                        )
                        shuffled_scores[local_test] = s_model.decision_function(
                            s_scaler.transform(rows_to_score)
                        )

                valid = np.isfinite(scores)
                y_test = followed[test_index][valid]
                if (
                    int(y_test.sum()) < args.min_class
                    or int((~y_test).sum()) < args.min_class
                ):
                    print(
                        f"{train_bundle[:15] + ' -> ' + test_bundle[:15]:<34}"
                        f"{'skipped (class imbalance)':>52}"
                        f"{f'{int(y_test.sum())}/{int((~y_test).sum())}':>12}"
                    )
                    report["cells"].append(
                        {
                            "relation": rel,
                            "train": train_bundle,
                            "test": test_bundle,
                            "skipped": "class_imbalance",
                            "n_pos": int(y_test.sum()),
                            "n_neg": int((~y_test).sum()),
                        }
                    )
                    continue

                test_facts = list(facts[test_index][valid])
                probe_auc = auc(y_test, scores[valid])
                lo, hi = cluster_bootstrap_auc(
                    test_facts,
                    y_test,
                    scores[valid],
                    args.bootstrap_replicates,
                    args.seed,
                )

                # Baselines on exactly these rows.
                confidence_scores = np.asarray(
                    [
                        -confidence.get((f, test_bundle), np.nan)
                        for f in test_facts
                    ]
                )
                confidence_auc = (
                    auc(y_test, confidence_scores)
                    if np.isfinite(confidence_scores).all()
                    else float("nan")
                )
                resolution_auc = float("nan")
                if resolution_direction is not None:
                    resolution_auc = auc(
                        y_test, x[test_index[valid]] @ resolution_direction
                    )
                shuffled_valid = np.isfinite(shuffled_scores[valid])
                shuffled_auc = (
                    auc(y_test[shuffled_valid], shuffled_scores[valid][shuffled_valid])
                    if shuffled_valid.sum() > 1
                    else float("nan")
                )

                spearman = float("nan")
                item_margin = margin[test_index][valid]
                if np.isfinite(item_margin).all() and len(item_margin) > 2:
                    a = np.argsort(np.argsort(scores[valid]))
                    b = np.argsort(np.argsort(item_margin))
                    spearman = float(np.corrcoef(a, b)[0, 1])

                print(
                    f"{train_bundle[:15] + ' -> ' + test_bundle[:15]:<34}"
                    f"{probe_auc:>12.3f}{f'[{lo:.2f}, {hi:.2f}]':>18}"
                    f"{confidence_auc:>10.3f}{resolution_auc:>9.3f}"
                    f"{shuffled_auc:>10.3f}"
                    f"{f'{int(y_test.sum())}/{int((~y_test).sum())}':>12}"
                )
                report["cells"].append(
                    {
                        "relation": rel,
                        "train": train_bundle,
                        "test": test_bundle,
                        "same_template": train_bundle == test_bundle,
                        "probe_auc": probe_auc,
                        "probe_auc_ci95": [lo, hi],
                        "parametric_confidence_auc": confidence_auc,
                        "resolution_projection_auc": resolution_auc,
                        "shuffled_label_auc": shuffled_auc,
                        "spearman_score_vs_margin": spearman,
                        "n_pos": int(y_test.sum()),
                        "n_neg": int((~y_test).sum()),
                        "selected_C": chosen_c,
                    }
                )

    # -----------------------------------------------------------------
    # cross-relation transfer.  Relations share no facts, so this needs no
    # folding: capitals and element symbols are disjoint by construction.
    # -----------------------------------------------------------------
    print("\n" + "=" * 108)
    print("CROSS-RELATION TRANSFER   (fit on one relation, evaluate on the other)")
    print("=" * 108)
    print(f"{'fit -> eval':<50}{'probe AUC':>12}{'95% CI':>18}{'pos/neg':>12}")
    relations = sorted(set(relation))
    for train_rel in relations:
        for test_rel in relations:
            if train_rel == test_rel:
                continue
            for bundle in BUNDLES:
                train_mask = (relation == train_rel) & (template == bundle)
                test_mask = (relation == test_rel) & (template == bundle)
                if not train_mask.any() or not test_mask.any():
                    continue
                if len(np.unique(followed[train_mask])) < 2:
                    continue
                y_test = followed[test_mask]
                if (
                    int(y_test.sum()) < args.min_class
                    or int((~y_test).sum()) < args.min_class
                ):
                    continue
                scaler, model, _ = fit_probe(
                    x[train_mask],
                    followed[train_mask],
                    c_grid,
                    args.seed,
                    inner_facts=list(facts[train_mask]),
                )
                score = model.decision_function(scaler.transform(x[test_mask]))
                value = auc(y_test, score)
                lo, hi = cluster_bootstrap_auc(
                    list(facts[test_mask]),
                    y_test,
                    score,
                    args.bootstrap_replicates,
                    args.seed,
                )
                label = f"{train_rel[:20]}/{bundle[:12]} -> {test_rel[:20]}"
                print(
                    f"{label:<50}{value:>12.3f}{f'[{lo:.2f}, {hi:.2f}]':>18}"
                    f"{f'{int(y_test.sum())}/{int((~y_test).sum())}':>12}"
                )
                report["cross_relation"].append(
                    {
                        "train_relation": train_rel,
                        "test_relation": test_rel,
                        "template": bundle,
                        "probe_auc": value,
                        "probe_auc_ci95": [lo, hi],
                        "n_pos": int(y_test.sum()),
                        "n_neg": int((~y_test).sum()),
                    }
                )

    # -----------------------------------------------------------------
    # Is it one direction or three?  Cosine between probes fitted per
    # template on all facts.  Diagnostic only: these are not held out.
    # -----------------------------------------------------------------
    print("\n" + "=" * 108)
    print("DIRECTION SIMILARITY      (cosine between per-template probes; diagnostic)")
    print("=" * 108)
    chance = 1.0 / np.sqrt(x.shape[1])
    print(f"random-vector cosine scale ~ 1/sqrt(d) = {chance:.4f}")
    for rel in relations:
        fitted: dict[str, np.ndarray] = {}
        for bundle in BUNDLES:
            mask = (relation == rel) & (template == bundle)
            if not mask.any() or len(np.unique(followed[mask])) < 2:
                continue
            if min(int(followed[mask].sum()), int((~followed[mask]).sum())) < args.min_class:
                continue
            scaler, model, _ = fit_probe(
                x[mask], followed[mask], [c_grid[0]], args.seed
            )
            raw = np.asarray(model.coef_[0], dtype=np.float64) / scaler.scale_
            fitted[bundle] = raw / np.linalg.norm(raw)
        names = list(fitted)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                value = float(np.dot(fitted[a], fitted[b]))
                print(f"  {rel:<17}{a[:16]:<18}{b[:16]:<18}cos = {value:+.4f}")
                report["direction_similarity"].append(
                    {"relation": rel, "a": a, "b": b, "cosine": value}
                )
        if resolution_direction is not None:
            for name, vector in fitted.items():
                value = float(np.dot(vector, resolution_direction))
                print(
                    f"  {rel:<17}{name[:16]:<18}{'resolution_raw':<18}cos = {value:+.4f}"
                )
                report["direction_similarity"].append(
                    {
                        "relation": rel,
                        "a": name,
                        "b": "resolution_raw",
                        "cosine": value,
                    }
                )

    print("\n" + "=" * 108)
    print("HOW TO READ THIS")
    print("=" * 108)
    print(
        """
The off-diagonal cells are the result.  The diagonal is now honest too (it is
also out-of-fold), so it is the ceiling: how well context-following is
predictable at all when the template does not change.

probe >> confid., off-diagonal holds up
    A template-invariant context-reliance direction exists that is not just
    knowledge strength.  Characterise it: cosine against resolution_raw, does
    it survive cross-relation transfer, does steering along it move behaviour.

probe ~= confid.
    The probe is reading how well the model knows the fact.  Report it as that.
    It is still a real finding -- item-level deference is governed by parametric
    confidence -- but it is not a context-reliance representation.

probe ~ 0.5 off-diagonal, high on the diagonal
    Whatever predicts context-following is template-specific.  The original Q4
    was fact memorisation.

shuffled far from 0.5
    Stop.  The evaluation is leaking; nothing else in this table is safe.
"""
    )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "fact_disjoint_transfer.json"
        path.write_text(json.dumps(report, indent=2, default=float))
        print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
