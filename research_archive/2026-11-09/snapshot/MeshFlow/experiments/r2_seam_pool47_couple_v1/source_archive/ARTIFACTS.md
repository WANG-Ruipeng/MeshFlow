# Artifact index and small-source archive

This archive contains 212 Python source snapshots (30 new experiment files), 2,174,382 source bytes. It contains no weights, meshes, noise arrays, training cache or dataset copies.

`code.diff` is an explicit `/dev/null` → new-file review patch for the new experiment sources. It is not a Git working-tree diff, a diff against historical experiments, or evidence of a commit. Reused source files are preserved as snapshots and are not represented as newly authored code.

Source selection comes from this experiment's registered code hashes, plus its owned Python package and direct experiment helper scripts. The training dispatch registration intentionally covers a dependency envelope; inclusion alone does not prove that every dependency function executed. Exact provenance is retained in `source_hashes.csv`.

| Artifact | Local path |
|---|---|
| Final report | [path](../REPORT.md) |
| Final audit | [path](../final_audit.json) |
| Frozen scientific protocol | [path](../protocol.json) |
| Actual training analysis | [path](../training_analysis/complete.json) |
| Final 252-output numerical table | [path](../results/per_output.csv) |
| Per-noise paired results | [path](../results/paired_per_noise.csv) |
| Per-parent paired results | [path](../results/paired_per_parent.csv) |
| FF role ledger | [path](../results/illegal_pairs_by_role.csv) |
| Deletion ledger | [path](../results/deleted_faces.csv) |
| All-bound connectivity/cost ledger | [path](../results/binding_ledger.csv) |
| Saved-layout CPU feasibility result | [path](../cpu_binding/NUMERICAL_REPORT.md) |
| Saved-layout CPU DEV detail | [path](../cpu_binding/NUMERICAL_ADDENDUM.md) |
| DEV finite-edge and alias records | [path](../cpu_binding/dev_diagnostic/completion.json) |
| Final same-1.5% binding receipt | [path](../endpoint_binding/completion.json) |
| RAW/POST gallery | [path](../gallery/training/index.html) |
| Same-bound gallery | [path](../gallery/endpoint_binding/index.html) |
| Actual visual-review records | [path](../visual_review_endpoint.csv) |
| Scientific summary figures | [path](../summary_figures/completion.json) |
| Resumable explicit stage entry | [path](../../../r2/pilots/seam_pool47_couple_v1/resume.py) |

Weights remain in their existing locations and are not read or copied by this archiver:

- COUPLE3000: `/mnt/e/MeshFlow/experiments/r2_seam_pool47_couple_v1/inference/COUPLE3000_generator.pt`; registered SHA-256 `621c55ef91c41985ef68cbde1415f56a2a9729251495a3ba8a39f5dad1ff766f`.
- PLAIN3000: `/mnt/e/MeshFlow/experiments/r2_seam_pool47_couple_v1/inference/PLAIN3000_generator.pt`; registered SHA-256 `89614faf3877c20fd99eb346f39f4f762302ebe06d565be64f39d5c741ba286d`.

The gate records the report and final audit hashes in `completion.json`. This archiving action executes no experiment and changes no maintained default. It does not commit, push, remove files or promote a model.
