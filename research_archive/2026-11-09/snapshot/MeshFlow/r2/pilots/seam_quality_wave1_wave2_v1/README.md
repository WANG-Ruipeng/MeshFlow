# R2 SEAM Quality Wave 1 / Wave 2

This isolated experiment implements the user's frozen task in
`experiments/r2_seam_quality_wave1_wave2_v1/TASK.md`.
It does not change the maintained generator, scorer or postprocessor.

Use the existing WSL environment with repository and `r2/src` on `PYTHONPATH`.
The status entrypoint never starts work:

```text
python -B -m r2.pilots.seam_quality_wave1_wave2_v1.resume status
```

Ordered stages (explicit dispatch receipts are required before model calls):

1. Stage 0 whitelist cleanup, completed once and documented in `cleanup/`.
2. `exports`: CPU-only PLAIN step1000/2000 extraction and fixed FP32 alpha0.5 interpolation.
3. `generation prepare --wave wave1`; freeze release and resources.
4. `generation raw --arm <label>` for each of three Wave1 labels.
5. Only after all84 RAW: `generation post --arm <label>` for each label.
6. `scoring score --wave wave1`, `scoring summarize --wave wave1`,
   `binding run --wave wave1`, `aggregation summarize --scope wave1`.
7. The exact original8000-cache transport is frozen by `training_prepare`.
   After Wave1 numerical completion, release Wave2 and run `training --arm
   CLEAN_BULK_FM` and `training --arm SURGICAL_LATE4`, exactly1000 updates each.
8. Repeat generation/scoring/binding with `--wave wave2`; all56 RAW precede D0.
9. `aggregation summarize --scope final`, `training_summary`, CPU galleries,
   actual nonblind AI_PROXY image review, report, independent audit, completion.

W2 inference labels have `_S1000`; their files retain the training-arm names.
Each generation stage owns its model, RNG and output writer. Scientific inputs
and all original assets remain read-only. A pending/failed call cannot be silently
replayed; safe checkpoint continuation preserves the actual update cursor.
Once `completion.json` exists, the resume entrypoint permits status only.

The exact source-face weighting changes the whole-batch denominator, so a seam
weight of1 does not mean an unchanged absolute seam gradient coefficient.
Old PLAIN logs lack clean/dirty-category MSE; those values remain NA.
The observed historical DEV panel is exploratory, not a new untouched test set.
