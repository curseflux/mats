#!/usr/bin/env python3
"""E3: Does the template effect on context-following show up in the residual stream?

Behaviourally, three semantically equivalent paraphrases move Gemma's
context-following on identical items from 0.8% to 72% (elements) and reverse
rank order on capitals.  This script asks whether that swing is visible in the
activations, at the site the probes and steering vectors already live.

It answers four questions, cheapest first:

  Q1  Does the projection onto `resolution_raw` differ by template, paired
      within fact?  (Is the knob moving at all?)
  Q2  Within a single template, does the projection predict which individual
      items the model gets wrong?
  Q3  Across templates, paired within fact, does the CHANGE in projection track
      the CHANGE in behaviour?  This is the mediation-flavoured test.
  Q4  Does a behaviour probe fit on one template transfer to another?  If the
      internal state is template-invariant, it should.  If it does not, the
      internal variable is as template-bound as the behaviour is.

No GPU. Reads cached activations, the fitted probe, and the steering directions.

Usage
-----
python E3_projection_analysis.py \
    --behavior   results/gemma4_12b_conflict/behavior_results.jsonl \
    --manifest   /scratch/bbjr/skarmakar/conflict/activations/manifest.json \
    --probe      results/gemma4_12b_conflict/analysis/probes/probe_model.npz \
    --directions results/gemma4_12b_conflict/analysis/causal/causal_directions.npz \
    --out        results/gemma4_12b_conflict/analysis/template_projection
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_selected_activations(manifest_path: Path, rows: list[dict],
                              layer_offset: int, position_offset: int) -> np.ndarray:
    """Return (n_rows, hidden) for one layer/position, in behaviour-file order.

    Mirrors the shard contract in 03_fit_awareness_probes.py: shards are
    contiguous over the behaviour ordering and carry their own sample_ids.
    """
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


# --------------------------------------------------------------------------
# small stats helpers (fact-clustered bootstrap throughout)
# --------------------------------------------------------------------------

def cluster_bootstrap(fact_ids: list[str], statistic, reps: int = 5000, seed: int = 0):
    """Percentile CI resampling whole facts, never individual rows."""
    by_fact: dict[str, list[int]] = collections.defaultdict(list)
    for index, fact in enumerate(fact_ids):
        by_fact[fact].append(index)
    keys = list(by_fact)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        picked: list[int] = []
        for _ in range(len(keys)):
            picked.extend(by_fact[rng.choice(keys)])
        try:
            value = statistic(np.asarray(picked))
        except Exception:
            continue
        if value == value:
            draws.append(value)
    if not draws:
        return float("nan"), float("nan")
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    sorted_scores = scores[order]
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def scalar(array):
    """npz round-trips 0-d values as shape-(1,) arrays; unwrap either form."""
    array = np.asarray(array)
    return array.reshape(-1)[0] if array.size == 1 else array


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()

    rows = read_jsonl(args.behavior)
    probe = np.load(args.probe)
    directions = np.load(args.directions)

    layer_offset = int(scalar(probe["selected_layer_offset"]))
    position_offset = int(scalar(probe["selected_position_index"]))
    layer_index = int(scalar(probe["selected_layer_index"]))
    position_name = str(scalar(probe["selected_position_name"]))
    print(f"site: layer {layer_index}, position {position_name}\n")

    readouts = {
        "resolution_raw": unit(directions["direction__resolution_raw"]),
        "awareness_probe": unit(probe["awareness_coef_raw"]),
    }
    if "direction__awareness_raw" in directions:
        readouts["awareness_interaction"] = unit(directions["direction__awareness_raw"])

    print("loading activations (this reads the shards once)...")
    activations = load_selected_activations(args.manifest, rows, layer_offset, position_offset)
    print(f"  loaded {activations.shape}\n")

    # The behavioural effect lives in unprompted conflict trials only.
    keep = [i for i, r in enumerate(rows)
            if r["condition_id"] == "false_relevant" and r["policy_id"] == "neutral"]
    sub = [rows[i] for i in keep]
    x = activations[keep].astype(np.float64)
    followed = np.array([bool(r["generated_matches_context"]) for r in sub])
    facts = [str(r["fact_id"]) for r in sub]
    relation = np.array([str(r["relation_id"]) for r in sub])
    template = np.array([str(r["template_bundle_id"]) for r in sub])
    projections = {name: x @ vector for name, vector in readouts.items()}
    print(f"analysing {len(sub)} unprompted conflict trials\n")

    report: dict = {"site": {"layer": layer_index, "position": position_name}}
    bundles = ["development", "validation", "heldout_paraphrase"]

    # (fact, template) -> row index, and the fact list belonging to each relation
    index_by = {(f, t): i for i, (f, t) in enumerate(zip(facts, template))}
    facts_by_relation: dict[str, list[str]] = collections.defaultdict(set)
    for f, rel in zip(facts, relation):
        facts_by_relation[rel].add(f)
    facts_by_relation = {k: sorted(v) for k, v in facts_by_relation.items()}

    # ---- Q1: does the knob move with the template? ----------------------
    print("=" * 78)
    print("Q1  projection by template (paired within fact)")
    print("=" * 78)
    q1 = []
    for rel in sorted(set(relation)):
        print(f"\n--- {rel} ---")
        print(f"{'readout':<22}{'template':<21}{'mean proj':>11}{'ctx%':>8}")
        for name, proj in projections.items():
            for bundle in bundles:
                mask = (relation == rel) & (template == bundle)
                if not mask.any():
                    continue
                print(f"{name:<22}{bundle:<21}{proj[mask].mean():>11.3f}"
                      f"{100 * followed[mask].mean():>7.1f}%")
                q1.append({"relation": rel, "readout": name, "template": bundle,
                           "mean_projection": float(proj[mask].mean()),
                           "context_rate": float(followed[mask].mean()),
                           "n": int(mask.sum())})
        # paired contrasts, within fact
        print(f"\n{'readout':<22}{'contrast':<34}{'Δproj':>9}{'95% CI':>20}")
        for name, proj in projections.items():
            shared = facts_by_relation[rel]
            for a, b in (("validation", "heldout_paraphrase"),
                         ("validation", "development"),
                         ("development", "heldout_paraphrase")):
                pairs = [(index_by[(f, a)], index_by[(f, b)])
                         for f in shared if (f, a) in index_by and (f, b) in index_by]
                if not pairs:
                    continue
                deltas = np.array([proj[i] - proj[j] for i, j in pairs])
                pair_facts = [facts[i] for i, _ in pairs]
                lo, hi = cluster_bootstrap(pair_facts,
                                           lambda idx: float(deltas[idx].mean()),
                                           args.bootstrap_replicates)
                star = "***" if (lo > 0 or hi < 0) else "   "
                print(f"{name:<22}{a[:14] + ' - ' + b[:14]:<34}{deltas.mean():>9.3f}"
                      f"{f'[{lo:+.2f}, {hi:+.2f}]':>20} {star}")
                q1.append({"relation": rel, "readout": name,
                           "contrast": f"{a}-{b}", "delta_projection": float(deltas.mean()),
                           "ci95": [lo, hi], "n_pairs": len(pairs)})
    report["q1_projection_by_template"] = q1

    # ---- Q2: within a template, does the projection predict the item? ----
    print("\n" + "=" * 78)
    print("Q2  within-template AUC: projection predicts which items follow context")
    print("=" * 78)
    print(f"{'relation':<17}{'template':<21}{'readout':<22}{'AUC':>7}{'95% CI':>20}")
    q2 = []
    for rel in sorted(set(relation)):
        for bundle in bundles:
            mask = (relation == rel) & (template == bundle)
            if mask.sum() < 8 or len(np.unique(followed[mask])) < 2:
                continue
            sub_facts = [f for f, m in zip(facts, mask) if m]
            for name, proj in projections.items():
                y, s = followed[mask], proj[mask]
                value = auc(y, s)
                lo, hi = cluster_bootstrap(sub_facts,
                                           lambda idx: auc(y[idx], s[idx]),
                                           args.bootstrap_replicates)
                print(f"{rel:<17}{bundle:<21}{name:<22}{value:>7.3f}{f'[{lo:.2f}, {hi:.2f}]':>20}")
                q2.append({"relation": rel, "template": bundle, "readout": name,
                           "auc": value, "ci95": [lo, hi], "n": int(mask.sum())})
    report["q2_within_template_auc"] = q2

    # ---- Q3: does Δprojection track Δbehaviour across templates? --------
    print("\n" + "=" * 78)
    print("Q3  paired within fact: does Δprojection track Δbehaviour?")
    print("    (items that flip between templates vs items that do not)")
    print("=" * 78)
    print(f"{'relation':<17}{'contrast':<30}{'readout':<22}{'Δproj|flip':>11}{'Δproj|same':>12}{'95% CI diff':>20}")
    q3 = []
    for rel in sorted(set(relation)):
        for a, b in (("validation", "heldout_paraphrase"), ("validation", "development")):
            pairs = [(index_by[(f, a)], index_by[(f, b)])
                     for f in facts_by_relation[rel]
                     if (f, a) in index_by and (f, b) in index_by]
            if len(pairs) < 8:
                continue
            flipped = np.array([followed[i] != followed[j] for i, j in pairs])
            pair_facts = [facts[i] for i, _ in pairs]
            if len(np.unique(flipped)) < 2:
                continue
            for name, proj in projections.items():
                deltas = np.array([proj[i] - proj[j] for i, j in pairs])

                def contrast(idx, d=deltas, f=flipped):
                    a_, b_ = d[idx][f[idx]], d[idx][~f[idx]]
                    if len(a_) == 0 or len(b_) == 0:
                        return float("nan")
                    return float(a_.mean() - b_.mean())

                lo, hi = cluster_bootstrap(pair_facts, contrast, args.bootstrap_replicates)
                print(f"{rel:<17}{a[:13] + '-' + b[:13]:<30}{name:<22}"
                      f"{deltas[flipped].mean():>11.3f}{deltas[~flipped].mean():>12.3f}"
                      f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}")
                q3.append({"relation": rel, "contrast": f"{a}-{b}", "readout": name,
                           "delta_when_flipped": float(deltas[flipped].mean()),
                           "delta_when_stable": float(deltas[~flipped].mean()),
                           "difference_ci95": [lo, hi], "n_pairs": len(pairs)})
    report["q3_delta_tracks_flip"] = q3

    # ---- Q4: does a behaviour probe transfer across templates? ----------
    print("\n" + "=" * 78)
    print("Q4  behaviour-probe transfer across templates")
    print("    fit on template A to predict context-following, evaluate on B.")
    print("    diagonal = within-template (optimistic); off-diagonal = the real test")
    print("=" * 78)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  scikit-learn not available; skipping Q4")
        report["q4_transfer"] = None
    else:
        q4 = []
        for rel in sorted(set(relation)):
            print(f"\n--- {rel} ---")
            label = "train \\ test"
            print(f"{label:<22}" + "".join(f"{b[:16]:>18}" for b in bundles))
            for train_bundle in bundles:
                train_mask = (relation == rel) & (template == train_bundle)
                cells = []
                if train_mask.sum() < 12 or len(np.unique(followed[train_mask])) < 2:
                    print(f"{train_bundle:<22}" + "".join(f"{'--':>18}" for _ in bundles))
                    continue
                scaler = StandardScaler().fit(x[train_mask])
                model = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")
                model.fit(scaler.transform(x[train_mask]), followed[train_mask])
                for test_bundle in bundles:
                    test_mask = (relation == rel) & (template == test_bundle)
                    if test_mask.sum() < 8 or len(np.unique(followed[test_mask])) < 2:
                        cells.append("--")
                        continue
                    scores = model.decision_function(scaler.transform(x[test_mask]))
                    value = auc(followed[test_mask], scores)
                    cells.append(f"{value:.3f}")
                    q4.append({"relation": rel, "train": train_bundle,
                               "test": test_bundle, "auc": value,
                               "within_template": train_bundle == test_bundle})
                print(f"{train_bundle:<22}" + "".join(f"{c:>18}" for c in cells))
        report["q4_transfer"] = q4

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("""
Q1 large, reliable Δproj  -> the knob moves with the template. The internal
   variable tracks the behavioural swing; it is a candidate mediator.
Q1 near zero              -> behaviour swings 71 points while this readout sits
   still. Whatever drives the template effect is not this direction.

Q2 high AUC               -> the projection predicts which individual items go
   wrong, i.e. it carries item-level information, not just a condition label.

Q3 is the sharp one. If Δprojection is larger for items that FLIP than for items
   that stay put, the readout tracks behaviour at the item level across
   templates. If the two are indistinguishable, the projection is moving for
   reasons unrelated to what the model actually says.

Q4 off-diagonal >> chance -> the internal state generalises across templates
   even though the behaviour does not. That is the interesting outcome: the
   activations are a more stable measure of context-reliance than the
   behavioural benchmark is.
Q4 off-diagonal ~ chance  -> the internal variable is as template-bound as the
   behaviour. Then 'context-faithfulness' has no template-invariant substrate
   at this site, which is a stronger and more troubling claim.

Caveats: this is observational. A projection that tracks behaviour is a
candidate mediator, not a demonstrated one. Confirming mediation needs the
steering experiment: move the projection to the value the other template
induces and check the behaviour follows.
""")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "template_projection_summary.json"
        path.write_text(json.dumps(report, indent=2, default=float))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
