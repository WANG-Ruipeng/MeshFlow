# Partial training evidence: A_FM and B_FREE_ALIGN

Scope: both arms completed 1,000 updates. This note reads finalized training logs and the frozen teacher summary only. No model forward, teacher encoding, scoring, checkpoint loading or parameter update was performed. C_KNOWN_ALIGN and all output-quality conclusions remain outside this partial note.

| Quantity | A_FM | B_FREE_ALIGN |
|---|---:|---:|
| Mean FM across all 1,000 online updates | 0.114649680 | 0.114675129 |
| Mean FM in online steps 901-1,000 | 0.112801802 | 0.112792877 |
| Raw alignment loss at step 1 | NA | 0.978782341 |
| Raw alignment loss at step 1,000 (four samples) | NA | 0.001318738 |
| Raw alignment loss, online steps 901-1,000 | NA | 0.001919199 |
| Teacher cosine, online steps 901-1,000 | NA | 0.998080801 |
| Generator global-clip trigger count | 15 | 16 |
| Auxiliary-head clip trigger count | NA | 0 |

The FM windows are paired by the same frozen inputs. Their small difference is a training-fit observation, not a generation-quality or causal estimate. The step-1 FM values are identical before either arm's first update. A's alignment columns are unused placeholders and must not be interpreted as a zero-loss predictor.

## Teacher comparison

The frozen teacher has 32 distinct complete-mesh targets from 128 images/8 batch forwards. Target pairwise cosine mean/median/min/max are 0.876316 / 0.872334 / 0.736769 / 0.990787. The targets are not numerically identical, but they are all chair-level global features.

A single normalized training-frequency-weighted teacher mean has frozen mean loss 0.061828205 across the actual 4,000 alpha=1 occurrences. B's mean online alignment loss across those same 4,000 occurrences is 0.025802339. This matches the occurrence denominator but averages predictions produced by changing checkpoints, so it is not a fixed-endpoint evaluation. The lower final-window loss (0.001919199) is also online and uses a different 400-occurrence subset; do not treat that comparison as an exact matched-window constant-baseline experiment.

B's prediction norm before L2 normalization grows from 15.762 at step 1 to 78.289 at step 1,000; teacher norms remain approximately one. Cosine loss normalizes both vectors, so a larger raw prediction norm is not itself improved alignment or a changed 0.1 coefficient.

## Sparse gradient observations

These are the first unaugmented microbatch at each registered step, restricted to the coordinate input projection plus fourth-layer attention parameters. FM uses the whole eight-sample free-coordinate denominator; the auxiliary gradient already includes 0.1/4. They are not whole-generator or whole-batch gradient ratios.

| Step | FM early-set norm | Weighted auxiliary norm | Auxiliary/FM norm | Gradient cosine |
|---:|---:|---:|---:|---:|
| 1 | 0.00259221555 | 0.0107783489 | 4.157968 | -0.022729 |
| 100 | 0.0108279614 | 0.000721201999 | 0.066606 | -0.034390 |
| 1000 | 0.00378613453 | 0.000195497967 | 0.051635 | 0.016484 |

The auxiliary early-set norm is nonzero at all three observations; both objectives report no unused member of that selected set. B's head has a nonzero actual parameter change at every logged update (minimum norm 0.0277495105); neither arm reports unused generator parameters. Together with the real preflight, this shows genuine auxiliary gradient connectivity beyond fitting only the head. The three near-zero signed gradient cosines do not establish either beneficial cooperation or harmful conflict over the full training trajectory.

## Interpretation boundary

Current evidence supports successful fitting of the specified training representation target and a live gradient path into the intended early generator. It does not establish S/I/E improvement, compatible use of C, preservation of a coherent chair body, diversity, superiority of known-side supervision, or a reason to replace START. Those require the registered endpoint generation, old/new-noise parent-paired metrics and actual complete visual review. The teacher has no local seam guarantee; training-target memorization or global semantics remain possible explanations.

Machine-readable numbers and source hashes: training_interpretation_partial.json. All original logs remain unchanged.
