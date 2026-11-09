# Chair checkpoint retention audit

Read-only asset audit, 2026-10-25. No model loads, GPU work, checkpoint deletions or scientific-data changes were performed. Root owns all cleanup mutations. Per-file identities and decisions are in `chair_audit.json`.

Selected pipeline: **PLAIN_S2000 -> official Chair D0 -> GEO_C / source orientation -> HINGE_OPP015 -> optional ALPHA / HOLE**. The CPU geometry stages require no additional neural weights.

## Required retained files

- `E:/MeshFlow/native_t1/runs/control_mvp48_v1/checkpoints/official_chair_last.pt` (1.9411 GiB): Protected official Chair pretrained generator
- `E:/MeshFlow/native_t1/runs/chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt` (1.4576 GiB): Retained original START hardlink; not a second physical allocation
- `E:/MeshFlow/experiments/r2_full_seat_learning_curve_v1/inference/FULL_S6000_generator.pt` (0.4858 GiB): FULL_S6000 method reference inference
- `E:/MeshFlow/experiments/r2_seam_quality_four_v1/inference/LONG_occupancy.pt` (0.0000 GiB): Actual C-only head used with PLAIN_S2000; retain cached layouts as well
- `E:/MeshFlow/experiments/r2_full_seat_learning_curve_v1/checkpoints/FULL_SEAT_step6000.pt` (1.4578 GiB): FULL_S6000 sufficient-training method reference, matching optimizer/RNG
- `E:/MeshFlow/experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step2000.pt` (1.4576 GiB): Actual complete restore for selected PLAIN_S2000; global15000/local2000
- `E:/MeshFlow/native_t1/checkpoints/chair_hybrid_start_total5000.pt` (1.4576 GiB): Maintained chair-hybrid CLI START; canonical hardlink
- `E:/MeshFlow/experiments/r2_seam_quality_wave1_wave2_v1/inference/PLAIN_S2000_generator.pt` (0.4858 GiB): Selected Chair generator for HINGE_OPP015 + optional ALPHA/HOLE
- `E:/MeshFlow/experiments/r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_0030000.pt` (6.7893 GiB): Protected official Chair D0; needed for retained pipeline

## Distinct BACK-positive branch

MULTIPART_S1500 has a separate bounded BACK result; it is not a uniform replacement for S2000. Root requested retaining both its inference assets and full optimizer/RNG restore. This changes one retention recommendation, not any experiment result.

- `E:/MeshFlow/experiments/r2_chair_multipart_seam_v1/training/checkpoint_step1500.pt` (1.4578 GiB): Root requested retaining full restore for the distinct BACK-positive MULTIPART_S1500 branch; preserve matching optimizer and RNG for an explicitly authorized future continuation.
- `E:/MeshFlow/experiments/r2_chair_multipart_seam_v1/inference/MULTIPART_S1500_occupancy.pt` (0.0001 GiB): Required companion head if MULTIPART inference is retained
- `E:/MeshFlow/experiments/r2_chair_multipart_seam_v1/inference/MULTIPART_S1500_generator.pt` (0.4858 GiB): Distinct BACK positive capability; retains a compact positive branch, not a replacement for S2000

## Obsolete checkpoint candidates

The candidate set includes old SEAM200/FULL4, LONG generator, ABC, nonselected COUPLE/PLAIN endpoints, CLEAN/SMOOTH/SURGICAL/WISE, DEDUP, CREF/SMALL, USER/EXP67, original R2 main/confirmation, FULL1000/3000, R29/R30 final adapters and old Q/G or N112 weights. Preserve their source, data, RAW/POST outputs, metrics, images, reviews, reports and provenance. Retirement ends exact model replay or optimizer continuation; do not silently download or regenerate retired checkpoints.

The 2.105 GB checkpoint in `.worktrees/a-layout-value-20260924/.../step00020000.pt` belongs to an abandoned fresh-random CLOSED16/N512 run. Its plan was 50,000 updates; the log reached about 23,000 and the saved checkpoint is at 20,000. It is **not an official pretrained model**, despite `official` appearing in the run directory name. Preserve the protocol and failure records.

## Current dependency review

- **native_t1_default**: native_t1/__main__.py defaults to chair-hybrid; native_t1/working_model.py DEFAULT_WORKING_CHECKPOINT is CHAIR_START_CHECKPOINT.
- **start_loader**: native_t1/chair_checkpoint.py loads the complete START state directly. Ancestral source_checkpoint_sha256 is identity metadata, not a file dependency; no pretrained download is required.
- **legacy_n112**: fm-geo and edge5 are explicit legacy profiles. Their old weights are proposed for retirement, including fm_cumulative3000.pt and jedge5_cumulative3000.pt. WORKING_CHECKPOINT still aliases EDGE5 for compatibility, but is not the maintained default.
- **r2_baseline**: r2/src/meshflow_control/r2_cli.py sample requires explicit --checkpoint. Original R2 main/confirmation weights are proposed for retirement; source, recipe and results remain.
- **s2000**: r2/pilots/chair_back_transfer_v1/worker.py loads only PLAIN_S2000_generator.pt, LONG_occupancy.pt and official chair_denoiser_0030000.pt. Generator construction is independent of prior model files.
- **official**: Official Chair generator and D0 are protected. All official/public pretrained assets outside this Chair audit also remain protected.
- **remaining_small_files**: Retain YAML/config, JSON identity sidecars, condition metadata and cached LONG layouts; cleanup must not retire these as if they were model checkpoints.
- **checks_performed**: Read source text and inventory only. No neural imports, checkpoint deserialization, model forward or GPU operation.

## Counts and storage accounting

| Decision | Files | Logical GiB | Unique listed inode GiB |
| --- | ---: | ---: | ---: |
| KEEP_DISTINCT_POSITIVE_INFERENCE | 2 | 0.4859 | 0.4859 |
| KEEP_DISTINCT_POSITIVE_RESUME | 1 | 1.4578 | 1.4578 |
| KEEP_EVIDENCE_NOT_WEIGHT | 1 | 0.0000 | 0.0000 |
| KEEP_REQUIRED | 9 | 15.5327 | 14.0751 |
| OUT_OF_SCOPE_TEST_FIXTURE | 6 | 0.0012 | 0.0012 |
| RETIRE_OBSOLETE_WEIGHT | 86 | 41.4302 | 39.9726 |

START canonical/original names are hardlinks to one allocation; retain both names. The historical R2 `latest.pt` and step6000 checkpoint are also hardlinks. Unique-inode totals only deduplicate the listed paths; actual free-space change depends on any additional hardlinks and filesystem accounting.

The old N112 profiles and original R2 checkpoints are included among retirement candidates. They are explicit historical routes, not maintained default dependencies. Cleanup documentation must mark their local weight replay as retired. No default selection was changed by this audit.

The original machine-readable audit is preserved as `chair_audit.before_BACK_resume_retention.json`. All size, identity and scientific evidence fields remain unchanged. The revised Markdown is ASCII-compatible UTF-8 to avoid terminal code-page corruption.
