# Original SEAM continuation: recovery and evaluation

This branch is route 3.1 of the authorized four-route experiment. The root controller owns GPU dispatch. Do not launch another worker or replay an incomplete scientific call automatically.

The source is the retained `bench_natural_n_recipe_v1/training/NATURAL_HIGH/checkpoint_step1500.pt`, global step 22500. The fixed candidate is 1500 additional original owner-only SEAM updates, global step 24000. The generator, matching AdamW state and RNG are inherited. Training uses the source-owner layout, while inference uses only C and the frozen historical OLD_S15000 occupancy head. The head is a research artifact, not an oracle and not retrained in this branch. No shared source noise, new loss, new target geometry or new topology representation is introduced.

The 26-parent natural-N pool and registered 1500-step stream provide 9000 OLD, 1500 FOCUS and 1500 HIGH instances. Each effective batch has eight clean/width instances, microbatch one. All 26 source layouts passed the exact face-preservation contract; a relationship not expressible by an owner face remains in the original bulk target rather than being repaired by changing C or adding a face.

Run commands from `/mnt/e/MeshFlow` using the existing project environment. Completed stages verify their receipts instead of repeating scientific calls.

```bash
python -m r2.pilots.bench_four_routes_v1.seam.training --arm SEAM1500
python -m r2.pilots.bench_four_routes_v1.seam.closeout training

# Root dispatches these only after resource measurement and endpoint completion.
python -m r2.pilots.bench_four_routes_v1.seam.evaluation sample --model S
python -m r2.pilots.bench_four_routes_v1.seam.evaluation d0 --model S

# CPU worker counts are set by the root resource plan.
python -m r2.pilots.bench_four_routes_v1.seam.evaluation raw --models S --workers 2
python -m r2.pilots.bench_four_routes_v1.seam.evaluation post --models S --workers 2
python -m r2.pilots.bench_four_routes_v1.seam.evaluation collect
python -m r2.pilots.bench_four_routes_v1.seam.report_results tables
python -m r2.pilots.bench_four_routes_v1.seam.report_results render --workers 2
```

All 46 S RAWs must be saved before the first D0 call. The budget is 2300 Euler generator forwards, 21 unique nonclosed-C occupancy-head forwards and 46 official D0 forwards. Two of the 23 conditions are closed C and have no head call or SEAM interface. The fixed source-oriented GEO_C and HINGE_OPP015 → ALPHA → HOLE chain is unchanged. The B reference is entirely read-only: 46 cached outputs and 138 existing score rows. The new S scores add 138 rows for a total of 276, retaining the original 177 fields and applicability states. Primary I includes only its existing valid-support domain; LOW_SUPPORT remains NA in the primary view.

The V2 gallery contains 138 pages, each showing B/S with two views and the two original-position sections. `visual_review/assignments.json` fixes 8/8/7 condition ownership. No label is created before actual image inspection. After all three explicit manual-note shards are compiled:

```bash
python -m r2.pilots.bench_four_routes_v1.seam.review_record --input <notes.json> --shard <assigned_reviewer>
python -m r2.pilots.bench_four_routes_v1.seam.report_results visual
python -m r2.pilots.bench_four_routes_v1.seam.closeout evaluation
```

The B/S contrast measures the complete SEAM continuation. B receives no matched extra Native updates, so this comparison does not isolate representation from added training. `NEW_WIDE_TRAIN` is a legacy panel group name; these four additional parents are unseen by both B and S in this narrow-pool branch.

Recovery is permitted only from an explicitly verified complete update boundary, with the exact saved model/AdamW/RNG and call ledger. A failed or pending forward/backward/update requires review; do not silently rerun it. Keep original attempt files, source checkpoints, arrays, scores and receipts. This branch does not authorize cleanup, model promotion, commit or push.
