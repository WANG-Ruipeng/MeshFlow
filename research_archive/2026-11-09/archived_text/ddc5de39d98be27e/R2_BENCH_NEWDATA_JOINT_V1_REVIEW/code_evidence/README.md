# Actual code evidence: Bench new-data joint 1500

This archive contains source code only: no weights, optimizers, raw meshes, generated
images, or copied third-party Python installations. It is a review snapshot, not a
standalone runnable installation. MeshFlow files preserve repository-relative paths
under `MeshFlow/`; the original scorer from the sibling checkout is under
`external/MeshFlow-Control/` with that checkout's relative paths preserved.

`source_hashes.json` distinguishes code whose bytes are bound by actual producer
registrations from current helper snapshots, historical comparison files, and the
conservative AST-resolved local import closure. An imported branch or a supporting
core module being present does not certify that its functions executed. No module
was imported by this copier; it parsed source as text only. No source was edited.

`code.diff` compares new wrappers to actual RELATION_V1 counterpart files when one
exists; new standalone helpers are full additions. `eval_runtime.py` is explicitly
marked as an API refactor from the historical combined eval_plan. The failed
pre-forward `sample.py` is preserved separately; successful `sample_v2.py` has its
own explicit engineering delta. This diff is review evidence, not a patch to apply
blindly to the historical experiment. All legacy scoring and geometry formulas
are retained verbatim in their archived source files.

Run the top-level experiment `build_code_evidence.py verify` to verify archived
bytes and current source equality. This does not load models or score geometry.
Rendered pages and actual visual-review labels have separate receipts and are not
certified by this source snapshot. Later helper changes must receive a new version,
not silently replace this archive.
