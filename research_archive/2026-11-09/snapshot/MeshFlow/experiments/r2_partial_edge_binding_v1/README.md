# Partial C/F edge binding: recovery and reproducibility

This directory contains the completed, CPU-only 24 requests x 2 binding modes x 3 displacement caps experiment. All 144 scientific cells are saved. No neural network, model forward, GPU work, new noise or training is required by this recovery entry.

## Read-only status (recommended)

Run from the repository root in the existing WSL environment:

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.tools.partial_edge_binding_v1.resume --status
```

The wrapper validates the frozen input manifest, original asset/source hashes, documented engineering amendment chain, every completed cell file hash, execution completion, and the identity of the existing independent verification. It returns JSON and writes no files. When the experiment is complete, `--resume` also only verifies and returns; it does not rerun the 144 cells.

## Resume only unfinished authorized work

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.tools.partial_edge_binding_v1.resume --resume --workers 2
```

Recheck current host resources before dispatching unfinished work. The wrapper delegates to the existing frozen `run.py`; it does not implement a second decoder, relax constraints, increase the experiment budget, or reset any result. Existing unfinished/failed cell directories cause an explicit stop. Preserve them and document any engineering recovery before proceeding. Historical failed pre-decode attempts remain under `engineering_failures/` and are not silently deleted.

## Independent verification only

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.tools.partial_edge_binding_v1.verify
```

This checks saved arrays, source identities, original GEO_C cluster membership, actual pre-GEO displacement caps, candidate ordering/funnels, shape/normal gates and edge/pair accounting. It runs no decoder, model or quality scorer. It verifies the original pair-classification results rather than claiming an independently reimplemented intersection test.

## Evidence files

- `input_manifest.json`, `registration.json`: frozen requests, assets, budget and contracts.
- `engineering_amendments.json`: explicit source-hash chains for integration/import fixes.
- `cells/<A_or_B>_<010_or_020_or_035>/<request>/`: indexed meshes, candidates, scores, connectivity and paired ledgers.
- `per_output.csv`, `summary.csv`, `rejection_funnel.csv`: complete results.
- `independent_verification.json`: successful 144-cell independent arithmetic audit.
- `actual_code/`: exact scorer and experimental source snapshots, including preserved earlier revisions.
- `code.diff`: addition-only patch for the seven experiment/recovery source files and this README. No original scorer or asset is modified.
- `source_hashes.csv`: SHA256 and byte size for the seven source files, README and all archived source snapshots.

The geometry contract is source-preserving indexed assignment to baseline POST representatives, followed by exact cleanup and fixed-C orientation. It does not rerun radius clustering or revive previously deleted faces. Source orientation is not a global outward-normal guarantee. A new legal interface plus an inherited RAW adoption label remains a screening result until separately reviewed on the modified POST geometry.
