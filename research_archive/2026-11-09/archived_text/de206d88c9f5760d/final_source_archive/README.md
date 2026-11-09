# Final source snapshot (additional version)

Status: SOURCE_SNAPSHOT_ONLY. Evaluation/image review is not certified complete by this archive. The original source_archive is preserved. This snapshot includes current experiment Python sources, frozen training/scoring/inference dependencies, original D0/GEO and final geometry/scoring dependencies, metric policy and execution contracts.

code.diff describes each new experiment module as an addition from /dev/null. It is a review artifact, not a patch applied to Git. Original model, scorer and existing postprocessor implementations are reused and their registered hashes verified; they are not listed as method changes. No Git index operation occurred.

report_results.py has the documented JSON-idempotence fix. versions.json preserves the recorded pre-fix and current SHA values; the pre-fix source bytes were not present and were not reconstructed. The original training-only archive did not contain this reporting module.

No weights, large arrays or training logs are copied. All paths are relative beneath repository/ or external/MeshFlow-Control/. This is a source snapshot with dependency identities, not a standalone weight/data package. RUNBOOK.md in the original experiment folder gives manual status/remaining-stage entry points; completed scientific stages must not be replayed.
