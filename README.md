# Gemma 4 conflict awareness and resolution

This pipeline asks a sharper question than “can a probe classify contradictory
prompts?”:

> Does Gemma 4 form a conflict-sensitive internal state that generalizes beyond
> facts and surface forms, and can that state be causally dissociated from the
> model's choice to follow context or parametric knowledge?

The key separation is between **awareness** and **resolution**. Awareness is
defined by the factorial stimulus label: a claim is both false and relevant to
the query. Resolution is a separate behavioral outcome: the model's relative
likelihood for the contextual and parametric answers. The model's answer is
never used as the conflict label.

## Pipeline

| File | Purpose | Loads Gemma? | Main outputs |
|---|---|---:|---|
| `01_screen_knowledge.py` | Establish model-specific parametric knowledge without conflicting context. | Yes | `screening_results.jsonl`, `eligible_facts.jsonl` |
| `02_collect_model_data.py` | Retain complete screened factorial groups; collect greedy answers, exact candidate likelihoods, and post-block residuals. | Yes | `behavior_results.jsonl`, activation shards |
| `03_fit_awareness_probes.py` | Select a layer using validation only; fit awareness, falsehood, and relevance probes; compare awareness with a factorized baseline. | No | `probe_scan.csv`, `probe_model.npz`, `probe_scores.jsonl` |
| `04_analyze_resolution.py` | Run paired policy, conflict-cost, factorial-interaction, and awareness–resolution analyses with fact-clustered uncertainty. | No | CSV summaries, `resolution_report.md` |
| `05_causal_dissociation.py` | Steer the selected residual at the exact `assistant_start` token and test awareness/resolution dissociation. | Yes | `causal_results.jsonl`, `causal_report.md` |

`common.py` owns the model-loading, chat-template, padding, generation,
candidate-scoring, and activation-position invariants. `config.yaml` pins the
model revision and experiment settings.

## Setup

Use a fresh environment. Install the CUDA-appropriate PyTorch build for your
machine, then install the remaining packages:

```bash
python -m pip install torch
python -m pip install "transformers==5.15.0" accelerate pyyaml numpy scipy scikit-learn
```

The code deliberately enforces `transformers==5.15.0`, because the checkpoint,
processor, model class, and chat template are version-sensitive. Gemma 4's
[official model card](https://huggingface.co/google/gemma-4-12B-it) uses
`AutoProcessor`, `AutoModelForMultimodalLM`, `add_generation_prompt=True`, and
supports `enable_thinking=False`; the pipeline verifies the render-then-tokenize
path against the processor's direct chat-template path at runtime. Hugging
Face's [chat-template documentation](https://huggingface.co/docs/transformers/main/en/chat_templating)
explains why model-specific control tokens should come from the template rather
than be handwritten.

Before running, edit these two entries in `config.yaml`:

```yaml
paths:
  dataset_dir: /absolute/path/to/your/generated_dataset
  output_dir: /absolute/path/to/results/gemma4_12b_conflict
```

Keep the pinned model ID and revision unchanged across all model-running steps.
The scripts resolve relative YAML paths from the directory containing
`config.yaml`, not from the shell's current directory.

## Commands

Run commands from the directory containing these scripts.

### 1. Screen parametric knowledge

First validate without loading the model:

```bash
python 01_screen_knowledge.py \
  --config config.yaml \
  --validate-only
```

Then run the screen:

```bash
python 01_screen_knowledge.py \
  --config config.yaml
```

The screen requires every prompt bundle to generate an acceptable one-word true
answer and to give the best acceptable true answer a greater summed sequence
log-probability than every assigned false answer. EOS is excluded from all
candidate scores.

### 2. Collect behavior and activations

Validate the screened subset first:

```bash
python 02_collect_model_data.py \
  --config config.yaml \
  --validate-only
```

Optional: inspect behavior before committing disk space to all activations:

```bash
python 02_collect_model_data.py \
  --config config.yaml \
  --behavior-only
```

Then collect or resume the complete activation run:

```bash
python 02_collect_model_data.py \
  --config config.yaml
```

The behavior file is reused if it already exists with matching provenance.
Activation shards are atomic and resumable. Use `--overwrite` only when you
intentionally want to replace this step's outputs.

Your screen retained 9,468 records in 789 complete groups (267 of 271 facts).
The two exclusion counts are nonexclusive: query-fact failures can overlap
paragraph-fact failures. Nothing about that summary suggests a balance problem;
all surviving groups still contain the full 4 conditions × 3 policies.

With 9,468 records, 48 layers, 7 positions, hidden size 3,840, and float16
storage, the activation tensors alone occupy about **22.75 GiB**, before shard
metadata and filesystem overhead.

### 3. Fit awareness and factor probes

```bash
python 03_fit_awareness_probes.py \
  --config config.yaml \
  --selection-position assistant_start \
  --c-grid 0.001,0.01,0.1,1,10 \
  --scan-shrinkage 0.10 \
  --layer-chunk-size 4 \
  --max-iter 2000
```

The script uses the following locked split logic:

| Analysis split | Relation | Fact split | Template | Role |
|---|---|---|---|---|
| `train` | country capital | train | development | Fit only |
| `validation` | country capital | validation | validation | Select layer and regularization |
| `id_test` | country capital | test | development | Held-out facts |
| `paraphrase_test` | country capital | test | heldout paraphrase | Held-out facts and surface form |
| `ood_relation` | element symbol | OOD | development | Relation transfer |
| `ood_relation_paraphrase` | element symbol | OOD | heldout paraphrase | Relation + surface-form transfer |

The all-layer/all-position scan uses a training-only diagonal-shrinkage LDA
direction for efficiency. It writes only train/validation scan metrics, so test
cells cannot influence selection. The selected layer maximizes the minimum of:

1. overall conflict ROC-AUC;
2. ROC-AUC within relevant claims (`false_relevant` versus `true_relevant`);
3. ROC-AUC within false claims (`false_relevant` versus `false_irrelevant`);
4. ROC-AUC within each of the neutral, context, and parametric policies.

Selection is restricted to `assistant_start`, the only cached position used by
the causal intervention. Final L2 logistic probes share a training-fitted
standardization and choose `C` on validation only. The factorized baseline is

```text
P(false claim | h) × P(query-relevant claim | h)
```

If the direct awareness probe does not outperform that baseline on held-out
data, the result supports decodable falsehood and relevance but not a distinct
non-additive conflict feature.

### 4. Analyze resolution

```bash
python 04_analyze_resolution.py \
  --config config.yaml \
  --analysis-splits validation,id_test,paraphrase_test,ood_relation,ood_relation_paraphrase \
  --bootstrap-replicates 2000
```

Open `analysis/resolution/resolution_report.md` first. The script reports:

- policy effects paired by `content_pair_id` within false-relevant prompts;
- conflict costs paired within factorial group and policy;
- the non-additive probe contrast `FR − TR − FI + TI`;
- direct-awareness versus factorized-baseline ROC-AUC;
- the association between neutral-prompt awareness scores and resolution.

All intervals resample `fact_id` clusters. They do not treat the 12 rows in a
factorial group—or the three paraphrase bundles for a fact—as independent.
Validation results are model-selection diagnostics, not confirmatory evidence.

### 5. Run the causal dissociation

A small end-to-end pilot is useful before the full run:

```bash
python 05_causal_dissociation.py \
  --config config.yaml \
  --evaluation-splits id_test,paraphrase_test,ood_relation \
  --directions awareness_specific,resolution_awareness_orthogonal,random_orthogonal_01 \
  --strengths=-1,1 \
  --max-examples-per-split 4 \
  --random-controls 1 \
  --bootstrap-replicates 200
```

Then run the planned evaluation:

```bash
python 05_causal_dissociation.py \
  --config config.yaml \
  --evaluation-splits id_test,paraphrase_test,ood_relation \
  --directions awareness_specific,resolution_awareness_orthogonal,random_orthogonal_01 \
  --strengths=-2,-1,1,2 \
  --max-examples-per-split 24 \
  --scoring-batch-size 8 \
  --random-controls 1 \
  --bootstrap-replicates 1000 \
  --baseline-tolerance 0.10
```

The default full command evaluates 72 prompts × 13 scenarios × 2 candidate
answers = 1,872 teacher-forced sequences. Reduce `--scoring-batch-size` if GPU
memory is tight. Do not use `--allow-baseline-mismatch` merely to get past an
error: first check the model revision, Transformers version, chat template,
candidate strings, and padding. A causal run whose zero-vector baseline does
not reproduce step 02 is not trustworthy.

The directions are training-only:

- `awareness_specific`: the mean neutral-policy factorial interaction direction,
  residualized against the policy-resolution and falsehood/relevance probe
  directions; its sign is oriented to increase the awareness readout.
- `resolution_awareness_orthogonal`: the context-policy minus parametric-policy
  direction, residualized against the fitted awareness probe.
- `random_orthogonal_01`: a seeded random direction orthogonal to the experimental
  direction/readout subspace.

Strength is measured in standard deviations of training activations projected
onto each unit direction, which makes `±1` and `±2` interpretable relative to
the natural activation distribution.

## Reading the final result

A compelling result has all of the following:

1. **Manipulation check:** context instructions shift the answer margin toward
   context, while parametric instructions shift it toward memory.
2. **Generalized decodability:** the awareness probe is above chance within both
   matched-control subsets and transfers to held-out facts, paraphrases, and the
   element-symbol relation.
3. **Non-additivity:** the awareness probe beats the factorized
   falsehood-times-relevance baseline, and `FR − TR − FI + TI` is positive.
4. **Causal specificity:** awareness steering changes the awareness readout with
   little answer-margin movement, whereas awareness-orthogonal resolution
   steering changes the answer margin; the random control remains small.
5. **Dose response:** effects are monotonic across signed strengths and replicate
   across held-out splits rather than appearing only at `+2` or in validation.

Important caveats:

- A probe shows linear decodability, not phenomenal awareness.
- A confidence interval crossing zero is inconclusive; it is not proof of no
  effect. A strong “little effect” claim would require a preregistered
  equivalence bound and more power.
- Orthogonality is relative to the fitted linear readout. It does not prove that
  every possible nonlinear conflict representation is unchanged.
- Residual steering can go off manifold. Random controls, sign symmetry, modest
  strengths, and baseline fidelity are essential.
- Generated answers are useful summaries, but exact summed answer-token
  log-probability margins are the primary outcome because they retain graded
  information and handle multi-token one-word answers correctly.

## Output layout

```text
<output_dir>/
├── screening_results.jsonl
├── eligible_facts.jsonl
├── behavior_results.jsonl
├── activations/
│   ├── manifest.json
│   └── shard-*.pt
└── analysis/
    ├── probes/
    │   ├── probe_scan.csv
    │   ├── probe_model.npz
    │   ├── probe_metadata.json
    │   ├── probe_scores.jsonl
    │   └── probe_evaluations.json
    ├── resolution/
    │   ├── behavior_summary.csv
    │   ├── paired_effects.csv
    │   ├── awareness_resolution_association.csv
    │   ├── resolution_summary.json
    │   └── resolution_report.md
    └── causal/
        ├── causal_directions.npz
        ├── causal_results.jsonl
        ├── causal_summary.json
        └── causal_report.md
```

Every script refuses to overwrite its owned final outputs unless `--overwrite`
is explicit. Steps 01 and 02 additionally checkpoint long model runs.
