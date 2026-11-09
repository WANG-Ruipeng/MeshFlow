# Complete online training evidence: A/B/C

All three arms completed the preregistered 1,000 updates. This analysis reads finalized CSV/diagnostics and teacher summaries only. It does not load checkpoints, execute a model, recompute geometry scores, or inspect output galleries. Final checkpoint and call-ledger audit remains a separate stage after GPU work.

Online losses are observed at different successive weights and different fixed inputs. A final-window average is not a fixed-checkpoint evaluation. Training fits, gradient connectivity and actual parameter changes cannot establish S/I/E, compatibility with C, body quality, diversity, known-side superiority in generation, or model promotion.

| Online quantity | A_FM | B_FREE_ALIGN | C_KNOWN_ALIGN |
|---|---:|---:|---:|
| Mean FM, all 1,000 updates | 0.114649680 | 0.114675129 | 0.114703259 |
| Mean FM, updates 901-1,000 | 0.112801802 | 0.112792877 | 0.112804576 |
| Raw alignment mean, all 4,000 eligible occurrences | NA | 0.025802339 | 0.028914290 |
| Raw alignment mean, final 400 eligible occurrences | NA | 0.001919199 | 0.002328119 |
| Weighted alignment mean, updates 901-1,000 | NA | 0.000191920 | 0.000232812 |
| Cosine mean, updates 901-1,000 | NA | 0.998080801 | 0.997671881 |
| Prediction norm before normalization, updates 901-1,000 | NA | 74.810617142 | 77.232433968 |
| Generator clip trigger count | 15 | 16 | 15 |
| Aligner clip trigger count | NA | 0 | 0 |

A's zero alignment log columns are unused placeholders, not a perfect alignment predictor. B/C use the same four unaugmented occurrences per update, identical initial head tensors, and the same 0.1/4 per-occurrence coefficient. Their token selector is the only prescribed head difference.

## Teacher constant baseline

The same frozen normalized teacher mean, weighted by all 4,000 registered eligible occurrences, has mean raw cosine loss 0.061828205. It uses the same occurrence denominator as each full-run online mean, but the latter is produced by changing checkpoints and includes early optimization. It is not an endpoint generalization test. Final-window occurrence frequencies differ from the full-run frequencies, so do not present the full-run constant baseline as an exactly matched final-window experiment.

The 32 targets are not identical: pairwise cosine mean 0.876316, min 0.736769, max 0.990787. Their global chair semantics do not guarantee sensitivity to local offsets, interfaces or conflicts.

## Sparse gradient observations

Each observation is the first eligible microbatch at the registered step and only the fixed coordinate projection plus fourth attention parameter set. Auxiliary norms already include 0.1/4; FM uses the whole eight-example free-coordinate denominator. Ratios do not describe the full-generator or full-batch gradient balance.

| Arm | Step | FM norm | Weighted auxiliary norm | Auxiliary/FM norm | Cosine |
|---|---:|---:|---:|---:|---:|
| B | 1 | 0.00259221555 | 0.0107783489 | 4.157968 | -0.022729 |
| B | 100 | 0.0108279614 | 0.000721201999 | 0.066606 | -0.034390 |
| B | 1000 | 0.00378613453 | 0.000195497967 | 0.051635 | 0.016484 |
| C | 1 | 0.00259221555 | 0.0106768096 | 4.118797 | -0.015434 |
| C | 100 | 0.00965005718 | 0.0011530976 | 0.119491 | 0.026342 |
| C | 1000 | 0.00412112474 | 0.000173980749 | 0.042217 | 0.009317 |

All recorded auxiliary observations have a live path into the selected early parameters; no unused members are reported for either objective. The hook preflight separately confirmed exact generator predictions, direct zero gradients on unselected messages, and no direct auxiliary path to later blocks. Small positive or negative cosine at six sparse points cannot establish persistent cooperation or interference.

## Actual parameter updates and clipping

| Arm | Step | Generator actual delta norm | Head actual delta norm | Generator grad norm before clip | Head grad norm before clip |
|---|---:|---:|---:|---:|---:|
| A | 1 | 0.0235749967 | NA | 0.546882629 | NA |
| A | 100 | 0.0240381062 | NA | 0.239056662 | NA |
| A | 500 | 0.0248493496 | NA | 0.22989811 | NA |
| A | 1000 | 0.027257951 | NA | 2.01835084 | NA |
| B | 1 | 0.0236247778 | 0.197535306 | 0.564719021 | 0.131702363 |
| B | 100 | 0.0235965867 | 0.0309335142 | 0.235965416 | 0.0130884424 |
| B | 500 | 0.0249294546 | 0.0406820662 | 0.240365013 | 0.00477024214 |
| B | 1000 | 0.0273519661 | 0.0353394374 | 1.96189141 | 0.00272853137 |
| C | 1 | 0.023625074 | 0.197338313 | 0.561237335 | 0.124147058 |
| C | 100 | 0.0231968686 | 0.0313573554 | 0.23200193 | 0.0139317922 |
| C | 500 | 0.0247468296 | 0.0385903828 | 0.251040459 | 0.0045406092 |
| C | 1000 | 0.0272685196 | 0.0364646129 | 1.97221375 | 0.00266148522 |

Generator update norms are measured only at the four preregistered steps; blank values elsewhere are intentional. Head update norms are measured every update. The clips are separate, so head gradient magnitude is not included in the generator clipping norm. All three logs report generator AdamW 6000 at the endpoint; B/C head AdamW is 1000. These are runtime records, to be independently confirmed from saved state without a model forward.

Prediction norms are pre-normalization magnitudes and may grow even though the cosine criterion is normalized. That growth alone is not improved representation or a change in the fixed coefficient. The available evidence supports successful training-target fitting and real early-generator supervision, while generated control success remains to be tested by the registered old/new-noise S/I/E and complete actual visual review.

Full numerical details and immutable input hashes: training_interpretation_complete.json. Original partial note and all original training logs remain intact.
