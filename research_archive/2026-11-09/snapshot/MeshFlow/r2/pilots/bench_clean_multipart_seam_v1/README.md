# R2_BENCH_CLEAN_MULTIPART_SEAM_V1

This isolated pilot compares a fixed 1,500-update Bench recipe with retained
historical Bench results. The live generator, AdamW and RNG start at COMMON12000.
The source pool is filtered to 195 eligible conditions from 58 parents. Source
geometry is not repaired. The generator objective remains free-corner FM with
role-restricted OT and HYBRID 0.25.

The new 18,049-parameter C-only head has independent training gradients and
TRAIN-only feature normalization. This normalization, xyz augmentation, pool
filtering and short budget jointly define the new recipe. Historical models have
different update budgets; this is not a single-variable or matched-budget test.

Outputs are under `experiments/r2_bench_clean_multipart_seam_v1`.

## Bounded execution

Use the existing WSL Ubuntu environment:

```text
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python
```

Set `PYTHONPATH=/mnt/e/MeshFlow:/mnt/e/MeshFlow/r2/src` from
`/mnt/e/MeshFlow`. Inspect without dispatching anything:

```bash
python -B -m r2.pilots.bench_clean_multipart_seam_v1.resume
```

The explicit GPU stages are `train`, `raw-new`, `raw-old`, `post-new`, and
`post-old`, selected with `resume --execute-stage STAGE`. They use detached
workers, durable logs and stage receipts. Resource checks precede dispatch.
The training gate must already be frozen. A failed neural call is not retried
automatically. `--resume-training` accepts only an exactly saved safe state.
Do not relaunch a running stage or use a historic retired rolling checkpoint.

Both RAW labels must finish all 48 requests before either D0 stage. Every
request retains the original C, N, noise and query identities. Each output uses
Euler50 and one official Bench D0. The two labels differ only in their predicted
layout: `NEW_HEAD` and the exact cached `OLD_S15000_LAYOUT`.

## CPU evaluation and evidence

The independent tools have completion receipts and preserve previous products:

```bash
python -B -m r2.pilots.bench_clean_multipart_seam_v1.post_cpu --workers 2
python -B -m r2.pilots.bench_clean_multipart_seam_v1.comparison
python -B -m r2.pilots.bench_clean_multipart_seam_v1.gallery run --workers 2
python -B -m r2.pilots.bench_clean_multipart_seam_v1.execution_audit
```

After the actual image reviews and report are complete, run
`visual_review aggregate`, `archive_source`, and `closeout` in that order.
The final closeout verifies the saved weights and source archive before writing
`completion.json`; it performs no model calls.

See the final delivery index for the topology audit, RAW scoring, new-output
CPU processing and head-diagnostic stage records. The head diagnostic is a
separately budgeted GPU BF16 pass after the RAW barrier: 247 new calls per head,
with the 20 open evaluation conditions reused from caches.

The standard CPU chain is `BASE_POST → HINGE_OPP015 → ALPHA → HOLE`.
It uses the retained Chair operators and strict geometric safety checks.
`D0_GUARD` and `GEO_NO_SELF_COLLAPSE` are disabled. Native and closed-C
seam-specific quantities are NA, not zero-success claims. The original 179
metric fields and applicable fixed-BASE denominator variants are retained.

Rendering a gallery does not count as reviewing it. Actual nonblind AI_PROXY
records must cover all 48 request pages, both MAIN and AUX sheets, including
the original front and side cross-sections.

No best-checkpoint selection, extra training, default-profile change, cleanup,
commit or push is part of this experiment. Completion requires the final
weights, metrics, image review, report and independent execution audit.
