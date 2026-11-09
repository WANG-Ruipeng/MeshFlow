# SEAM quality: four bounded directions

This independent pilot implements the frozen contract in
`experiments/r2_seam_quality_four_v1/PROTOCOL.md`. The maintained sampler and
historical scorers are unchanged. Final findings belong in the experiment's
`REPORT.md`, not in assumptions about these mechanisms.

Three branches resume the same retained SEAM200 live generator, AdamW,
occupancy head and head optimizer for 800 additional updates:

- `LONG`: unchanged continuation.
- `ANTI_COVER`: continuation plus the registered finite-C two-probe loss.
- `LOCAL_NOISE`: continuation with only the active seam tip noise mean moved
  to its C-edge midpoint, after the shared OT matching.

Two separate CPU branches use the old SEAM200 cache: `antiflatten` and
`redirection`. They are not combined with the newly trained models. The latter
is compared with the existing capacity-guard geometry, as well as original GEO.

All sampling uses C-only predicted occupancy. Source occupancy is used during
training and for explicitly marked audits, never as an inference answer.
The 16 requests are four seen TRAIN conditions with four reused noises each.
They cannot establish unseen-condition or fresh-noise generalization.

## Recovery

From the repository root, with the existing environment and
`PYTHONPATH=r2/src:.`, inspect:

```text
python -B -m r2.pilots.seam_quality_four_v1.resume --stage status
```

Stages are explicit; there is deliberately no automatic GPU dispatcher.
Before any new worker, measure resources using `resource_snapshot` and apply
the workspace dispatch policy. Do not rerun preparation, registration or a
failed scientific cell silently. Completed cells validate their receipts.

```text
python -B -m r2.pilots.seam_quality_four_v1.resume --stage train --arm LONG
python -B -m r2.pilots.seam_quality_four_v1.resume --stage raw --arm LONG
python -B -m r2.pilots.seam_quality_four_v1.resume --stage post --arm LONG
python -B -m r2.pilots.seam_quality_four_v1.resume --stage score
python -B -m r2.pilots.seam_quality_four_v1.resume --stage summarize
python -B -m r2.pilots.seam_quality_four_v1.resume --stage render
```

The arm commands also accept `ANTI_COVER` and `LOCAL_NOISE`. POST refuses to
start until all 48 registered RAW outputs exist. Each gets exactly one official
D0 call, original GEO_C and source-winding export. Anchor coordinate and index
identity checks are distinct from geometry quality scoring.

Completed CPU branches have verification-only closeouts:

```text
python -B -m r2.pilots.seam_quality_four_v1.antiflatten.resume
python -B -m r2.pilots.seam_quality_four_v1.redirection.closeout
```

`training.summarize` reads logs/checkpoints without model calls. Root
`summarize` reads saved scores without re-scoring. Actual visual review is
recorded separately as nonblind AI_PROXY, never inferred from rendering or a
numerical score. Full checkpoints and inference exports remain in the local
experiment directory; no automatic cleanup, default promotion or publication
is part of this pilot.

After completion, verify the sealed report, source snapshot, tables, images and
review without resuming scientific work:

```text
python -B -m r2.pilots.seam_quality_four_v1.closeout
```

Add `--deep` to also recheck the complete independent audit's frozen input and
weight hashes. Neither mode runs a model or recalculates geometry scores.
