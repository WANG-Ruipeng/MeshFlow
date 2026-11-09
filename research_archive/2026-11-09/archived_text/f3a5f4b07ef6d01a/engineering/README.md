# CPU scoring wrapper correction

Attempt 001 completed the first request's geometry calculations, but the original immutable scorer rejected its output directory because the new wrapper had already put `attempt.json` inside that directory. No score files were written, no model was called, and no scientific settings changed.

Preserved: `score_original.py`, the original attempt directory under `score_attempt001_directory_conflict/`, and the first resource snapshot. Fix: store wrapper attempt metadata in the parent directory; keep the original scorer and its empty-directory guard unchanged. Resource snapshots now get unique attempt timestamps.

One explicit CPU-only recomputation of this first request is recorded; no GPU replay. `score_wrapper_fix.json` records the frozen and corrected wrapper hashes. This is an engineering exception to the wrapper source freeze, not a metric-definition change or a scientific retry.

Attempt 002 successfully wrote the first complete original score, then the wrapper tried to overwrite the scorer-owned `complete.json` with timing data. The original immutable writer refused. That complete original score is valid and reused without recomputation. Wrapper timing now lives in the parent `wrapper_complete.json`. `score_fix1.py` and `score_wrapper_fix2.json` preserve this second correction. First completed request timing/RSS are unavailable; the next already-authorized request provides the representative CPU measurement.

Packaging attempt 001 stopped before writing snapshots when a read-only original scoring dependency was found in sibling `E:/MeshFlow-Control`, outside the repository-relative snapshot assumption. The original code is preserved in `closeout_before_external_path_fix.py`. Packaging now accepts that exact known sibling tree and copies it into `source_snapshot/external/MeshFlow-Control/`; no external source is edited. This caused no scoring or model recomputation.
