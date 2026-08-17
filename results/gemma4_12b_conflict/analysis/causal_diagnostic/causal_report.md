# Causal dissociation of conflict awareness and answer resolution

Interventions were applied to the selected post-block residual at `assistant_start`. The resolution outcome is the change in `log P(context answer) − log P(parametric answer)`; positive values favor the contextual answer.

## Baseline fidelity

Maximum absolute difference from step 02's cached margin: **1.938** (tolerance 0.100).

## Dose-response slopes

Each estimate is the mean per-example slope per one training projection SD. Intervals are query-fact cluster-bootstrap 95% intervals.

| Split | Direction | Outcome | Slope (95% CI) | N |
|---|---|---|---:|---:|
| id_test | random_orthogonal_01 | resolution_margin_per_1sd | -13607.423 [-59523.973, 5094.283] | 24 |
| id_test | random_orthogonal_01 | awareness_logit_per_1sd | -0.000 [-0.000, -0.000] | 24 |
| paraphrase_test | random_orthogonal_01 | resolution_margin_per_1sd | -30598.879 [-83170.414, 0.000] | 24 |
| paraphrase_test | random_orthogonal_01 | awareness_logit_per_1sd | -0.000 [-0.000, -0.000] | 24 |
| ood_relation | random_orthogonal_01 | resolution_margin_per_1sd | -15747.070 [-67342.122, 22136.434] | 24 |
| ood_relation | random_orthogonal_01 | awareness_logit_per_1sd | -0.000 [-0.000, -0.000] | 24 |

## How to interpret the pattern

- First verify baseline fidelity and a roughly monotonic, sign-symmetric dose response. A one-sided jump at only the largest strength is more consistent with disruption.
- The strongest dissociation is: `awareness_specific` reliably changes the awareness readout with little resolution-margin movement, while `resolution_awareness_orthogonal` reliably changes the margin. The latter direction is algebraically orthogonal to the fitted awareness readout, so its near-zero readout effect is a construction check—not independent evidence that every possible awareness representation is unchanged.
- If both experimental directions move the answer margin similarly, awareness and resolution are entangled at this site (or the residualization basis is incomplete).
- If neither direction moves the margin, the probe may be correlational, the selected site may be downstream or bypassed, or the tested strengths may be too weak.
- Random-control effects comparable to the experimental effects indicate nonspecific residual-stream disruption. Do not interpret such a run mechanistically.
- Replication on held-out facts, paraphrases, and the element-symbol relation matters more than a large effect on one split.

## Scope of the claim

This experiment can support a claim about a linearly readable conflict variable and its causal relationship to answer-source resolution. It cannot establish phenomenal awareness, and it cannot prove that the fitted direction is the model's only conflict representation.
