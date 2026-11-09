Four-route evidence package. Reports and full per-output/per-parent CSVs retain all scientific statuses.
ZIP compression: ZIP_LZMA. Extract with Python standard library: python -m zipfile -e FOUR_ROUTES_REVIEW_SMALL.zip extracted
No model weights, meshes, RAW/noise arrays or full gallery PNGs are included.
Full training calls/exposure event logs and large schedules remain local; indexed_logs.json identifies exact bytes.
Execution closeouts, all final per-output/parent/step CSVs, training loss logs/plots and actual visual records are included.
Large internal face/pair scans remain local with hashes; final full-field output and parent tables are retained.
The three deletion_fate_per_slot ledgers are fully retained in compact_deletion_v2 as reversible CSVs.
normalize_shared_tables.py reconstructs original field strings, row order, and exact CSV bytes; verification is included.
Registered external scorer sources are copied under _external_sources, with original paths in source_hashes.json.
Source bytes are stored once per SHA in source_blobs; every original path is mapped in source_hashes.json.
From the extracted workspace, restore exact source paths using:
  python experiments/r2_bench_four_routes_v1/review_package/restore_sources.py --package-root .
Then reconstruct each original deletion CSV to a new path, for example:
  python -B -m r2.pilots.bench_four_routes_v1.bestof.normalize_shared_tables reconstruct --source seam --output reconstructed/seam_deletion.csv
Use --source exposure or bestof for the other original ledgers. No models or scores are run.
code.diff is against repository HEAD, with untracked source as additions; it is not a method ablation diff.
WEIGHTS.md is an index only. Source/report hashes are preserved; no experiments were run by this packager.
