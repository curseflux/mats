# Conflict awareness and resolution

This report keeps two questions separate: **is conflict linearly represented?** and **which answer does the model select?** Validation was used for model selection; the ID, paraphrase, and relation-transfer rows are the held-out evidence.

## Awareness probe versus factorized baseline

| Split | Policy | Awareness AUC (95% CI) | Falsehood × relevance AUC (95% CI) | AUC difference (95% CI) |
|---|---|---:|---:|---:|
| id_test | all | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [-0.000, 0.000] |
| id_test | neutral | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [-0.000, 0.000] |
| id_test | context | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [-0.000, 0.000] |
| id_test | parametric | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [-0.000, 0.000] |
| ood_relation | all | 0.985 [0.975, 0.992] | 0.971 [0.958, 0.982] | 0.014 [0.009, 0.019] |
| ood_relation | neutral | 0.984 [0.974, 0.992] | 0.957 [0.938, 0.973] | 0.028 [0.018, 0.039] |
| ood_relation | context | 0.981 [0.969, 0.990] | 0.993 [0.989, 0.997] | -0.013 [-0.021, -0.007] |
| ood_relation | parametric | 0.997 [0.994, 0.999] | 0.971 [0.956, 0.984] | 0.026 [0.015, 0.039] |
| ood_relation_paraphrase | all | 0.978 [0.967, 0.986] | 0.973 [0.963, 0.981] | 0.005 [-0.001, 0.012] |
| ood_relation_paraphrase | neutral | 0.995 [0.987, 1.000] | 0.991 [0.982, 0.998] | 0.005 [0.001, 0.009] |
| ood_relation_paraphrase | context | 0.959 [0.935, 0.980] | 0.965 [0.949, 0.978] | -0.006 [-0.023, 0.011] |
| ood_relation_paraphrase | parametric | 0.999 [0.996, 1.000] | 0.994 [0.988, 0.998] | 0.005 [0.001, 0.010] |
| paraphrase_test | all | 0.998 [0.993, 1.000] | 0.999 [0.997, 1.000] | -0.001 [-0.005, 0.000] |
| paraphrase_test | neutral | 1.000 [0.997, 1.000] | 1.000 [0.997, 1.000] | 0.000 [-0.000, 0.000] |
| paraphrase_test | context | 0.994 [0.981, 1.000] | 1.000 [1.000, 1.000] | -0.006 [-0.020, 0.000] |
| paraphrase_test | parametric | 0.998 [0.994, 1.000] | 0.997 [0.991, 1.000] | 0.001 [-0.000, 0.004] |
| validation | all | 0.999 [0.996, 1.000] | 0.998 [0.994, 1.000] | 0.001 [-0.002, 0.003] |
| validation | neutral | 1.000 [0.998, 1.000] | 1.000 [1.000, 1.000] | -0.000 [-0.001, 0.000] |
| validation | context | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [-0.000, 0.000] |
| validation | parametric | 1.000 [0.998, 1.000] | 0.999 [0.997, 1.000] | 0.000 [-0.000, 0.002] |

## Policy manipulation check

Positive values move likelihood toward the contextual answer; negative values move it toward the parametric answer. Each contrast is paired within identical content.

| Split | Contrast | Mean margin shift (95% CI) | Pairs |
|---|---|---:|---:|
| id_test | context_minus_neutral | 13.814 [11.559, 16.256] | 28 |
| id_test | parametric_minus_neutral | -29.963 [-34.101, -26.301] | 28 |
| ood_relation | context_minus_neutral | 27.929 [26.007, 29.598] | 118 |
| ood_relation | parametric_minus_neutral | -11.837 [-12.727, -10.989] | 118 |
| ood_relation_paraphrase | context_minus_neutral | 28.664 [26.877, 30.489] | 118 |
| ood_relation_paraphrase | parametric_minus_neutral | -3.561 [-4.066, -3.095] | 118 |
| paraphrase_test | context_minus_neutral | 12.305 [9.230, 15.568] | 28 |
| paraphrase_test | parametric_minus_neutral | -24.337 [-27.986, -20.570] | 28 |
| validation | context_minus_neutral | 21.193 [18.451, 23.971] | 30 |
| validation | parametric_minus_neutral | -14.617 [-17.821, -11.676] | 30 |

## Factorial conflict interaction

The contrast is `false-relevant − true-relevant − false-irrelevant + true-irrelevant`. A positive, transferable contrast is evidence that the readout is not merely a generic falsehood or relevance detector.

| Split | Awareness-logit interaction (95% CI) | Groups |
|---|---:|---:|
| id_test | 10.889 [9.829, 11.900] | 28 |
| ood_relation | 5.189 [4.454, 5.917] | 118 |
| ood_relation_paraphrase | 4.736 [4.371, 5.129] | 118 |
| paraphrase_test | 10.656 [9.304, 11.748] | 28 |
| validation | 7.829 [6.628, 8.975] | 30 |

## Awareness–resolution association under neutral instructions

These statistics use only false-relevant neutral prompts. They are descriptive, not causal: a probe score can track answer selection without controlling it.

| Split | Statistic | Estimate (95% CI) | N |
|---|---|---:|---:|
| id_test | spearman_awareness_logit_vs_resolution_margin | -0.235 [-0.566, 0.137] | 28 |
| id_test | auc_awareness_predicts_context_generation | 0.406 [0.164, 0.646] | 28 |
| id_test | auc_awareness_predicts_parametric_generation | 0.651 [0.422, 0.865] | 28 |
| ood_relation | spearman_awareness_logit_vs_resolution_margin | -0.574 [-0.698, -0.420] | 118 |
| ood_relation | auc_awareness_predicts_context_generation | 0.048 [0.007, 0.113] | 118 |
| ood_relation | auc_awareness_predicts_parametric_generation | 0.952 [0.884, 0.994] | 118 |
| ood_relation_paraphrase | spearman_awareness_logit_vs_resolution_margin | 0.108 [-0.091, 0.309] | 118 |
| ood_relation_paraphrase | auc_awareness_predicts_context_generation | 0.000 [0.000, 0.000] | 118 |
| ood_relation_paraphrase | auc_awareness_predicts_parametric_generation | 0.696 [0.243, 1.000] | 118 |
| paraphrase_test | spearman_awareness_logit_vs_resolution_margin | 0.262 [-0.098, 0.581] | 28 |
| paraphrase_test | auc_awareness_predicts_context_generation | 0.583 [0.348, 0.800] | 28 |
| paraphrase_test | auc_awareness_predicts_parametric_generation | 0.417 [0.194, 0.646] | 28 |
| validation | spearman_awareness_logit_vs_resolution_margin | 0.360 [-0.020, 0.693] | 30 |
| validation | auc_awareness_predicts_context_generation | 0.000 [0.000, 0.000] | 30 |
| validation | auc_awareness_predicts_parametric_generation | 0.643 [0.179, 1.000] | 30 |

## What to look for

- Context instructions reliably increase the resolution margin in 5/5 reported splits; parametric instructions reliably decrease it in 5/5.
- The neutral-policy factorial interaction is reliably positive in 5/5 splits.
- Under neutral instructions, the direct awareness probe reliably exceeds the factorized baseline in 2/5 splits.
- Prioritize consistent held-out and OOD effects over a single high validation number. A confidence interval crossing zero is inconclusive, not evidence of no effect.
- If the policy anchors do not move the likelihood margin in opposite directions, do not interpret the neutral condition as meaningful arbitration yet.
- If awareness does not beat the factorized baseline, the representation may still encode conflict, but the result does not establish a distinct non-additive awareness feature.
- Run `05_causal_dissociation.py` before making mechanistic claims. Probe accuracy alone establishes decodability, not causal use.

All 95% intervals are percentile cluster-bootstrap intervals over query facts.
