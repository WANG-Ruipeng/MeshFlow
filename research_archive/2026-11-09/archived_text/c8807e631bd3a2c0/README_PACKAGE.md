# Review package

This package contains the supplied scientific REPORT, METHODS and README_RESUME, all existing per-output/per-parent/noise CSV tables, all 56 compact JPEG comparison sheets, actual AI_PROXY review records, the data whitelist and exclusions/source interfaces, actual source snapshots/code.diff/hashes, and resource/call/update ledgers.

Large checkpoints, NPZ/NPY arrays, PLY meshes and full-resolution PNGs remain local and are deliberately not embedded. `local_binary_artifacts.csv` lists paths, sizes, saved SHA256 and the precise receipt/audit provenance. The finalizer does not reload or re-hash large weights; final full/inference checkpoint hashes come from the completed independent CPU checkpoint audit. Recovery-checkpoint hashes come from their writer receipts and are labelled accordingly.

`small_file_hashes.csv` covers the local small text/image artifacts, including PNGs not embedded; its embedded flag distinguishes availability. `PACKAGE_MANIFEST.csv` hashes the actual portable package members. Source data/interface identifiers are evidence only, not model inference inputs. Absolute paths in provenance retain their original local identities and are not download links.

The archive first uses standard DEFLATE. If that exceeds 10,000,000 bytes, CSV members use lossless ZIP-LZMA (method 14), leaving their exact bytes and all JPEG bytes unchanged. ZIP-LZMA requires a compatible extractor such as Python zipfile or 7-Zip; older Windows Explorer versions may not support it. The archive compression methods are recorded in the local finalization completion.

The portable gallery uses JPEG copies only; no scientific metric was recomputed to build this archive. Main numerical tables preserve all generators, parents, noises and thresholds. Reports are copied unchanged; finalization does not invent success, select images, change defaults, run models, or commit/push.

`OMITTED_JSON.csv` records the exact local path, SHA256, bytes and reason for recoverable stream plans, superseded registration copies and duplicate per-output/render detail JSONs excluded from this compact archive. Every original remains unchanged locally. All CSVs, official resource/call/update JSONL ledgers, current formal registrations, source interfaces/whitelist, final audit and all 56 unchanged JPEGs remain embedded.
