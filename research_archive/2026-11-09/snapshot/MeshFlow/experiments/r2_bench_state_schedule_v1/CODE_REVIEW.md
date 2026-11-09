# Independent code review — R2_BENCH_STATE_SCHEDULE_V1

The review covered `common.py`, `register.py`, `training.py`,
`perturbed_core.py`, and the asynchronous path/masking interface. The review
used source inspection and CPU array/tensor checks only. It did not construct
a neural model, initialize CUDA, perform a model forward/backward, add a
training update, or change a frozen training file.

## Corrected before the training registration

One concrete boundary inconsistency was found and reproduced using a CPU toy
tensor: an FP32 time stored as `0.7` becomes `0.699999988079071` when converted
to a Python float. The original summary-side comparison against Python `.7`
therefore classified a tip as active while the core correctly treated the
FP32 cutoff as reached. The reproduction reported core active scalar count
9 versus summary count 12, and loss `0.75` versus `1.0`; CUDA remained
uninitialized. This could have caused a false `Loss accounting differs`
failure.

Before freezing, the trainer was corrected to consume
`active_free_corner_mask` and `seam_time` from the asynchronous transform;
the row audit uses the FP32 cutoff. This is a bookkeeping/precision repair,
not a changed scientific boundary or extra training attempt.

## Contracts checked

- Exact retained A step0 is imported: generator at global12000, fresh empty
  AdamW state, named parameter order, and runtime RNG hashes. Each arm owns
  its model, optimizer, files and RNG.
- The original 1500 updates × 8 schedule rows supply parents, conditions,
  augmentation, time, noise, epsilon and permutation identities. The
  perturbation derives additional gate/alpha uniforms from a salted
  canonical row hash and does not consume the original RNG.
- Both arms retain the original structural free-coordinate denominator.
  Original C and seam anchors remain fixed. Asynchronous inactive tips are
  excluded from the numerator, not converted into artificial zero-error
  observations.
- Idea1 implements the specified endpoint corrective regression on
  perturbed states. Its target is not the derivative of the perturbed
  interpolation, and no conditional-flow-matching marginal theorem is
  claimed for it. Target amplification near the upper time cutoff is
  measured explicitly.
- Idea2 predicts local-clock velocities, uses the corresponding tip clock
  in training, and advances tips by the clock increment during rollout.
  Generated tips freeze after their own integration ends; this does not
  make them original input C or remove train/rollout mismatch.
- The registered new budget is two arms × 1500 optimizer updates, each
  12000 training forwards and 12000 backwards. No new fixed neural probe
  was introduced. The old A probe calls remain old calls in its own ledger.

## Additional hardening recommendations, not applied to frozen training

The configuration hash chain was checked against the original A source.
As future defensive improvements, an explicit runtime comparison of
`model_config` and more direct resume checks of parameter-name order and
global-step identity would make the contract easier to inspect. The
resource-check loop could also explicitly inspect the first resumed batch
even when the cursor is not on its 25-update resource-check boundary.
These were not observed identity mismatches and did not justify altering
the already frozen running code.

## Independent completion summary

`r2/pilots/bench_state_schedule_v1/training_summary.py` reads actual
`calls.jsonl`, `exposure.jsonl`, `training_log.jsonl`, status and checkpoint
receipts. It imports no torch/model code. A read-only prefix mode can
inspect committed updates while workers run. Formal CSV/PNG/JSON artifacts
are written only when both new arms and the reused A baseline are
`COMPLETE`.

It checks actual call identities and counts, the complete source row
stream, t/N/K/free-denominator equality with A, and initial
generator/optimizer/RNG identities. Initial checkpoint file hashes are
verified without tensor deserialization. It preserves global12000+local
step numbering and summarizes bulk/active-tip local velocity and endpoint
errors within each arm, using summed SSE divided by summed valid scalar
counts for first100/last100 windows.

The baseline's endpoint SSE is derived per exposure from its already saved
velocity SSE and `(1-t)^2`, not obtained from a new neural probe. Native
loss magnitudes across these different objectives must not be ranked as
generation quality. These are changing training examples and times;
rollout geometry and the fixed evaluation panel remain the endpoint test.
