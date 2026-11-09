# Small review evidence package

All 48 request pages and all 8 POST noise contact sheets are present, without image selection.
The image files in this archive are review copies: maximum width 2048 pixels, JPEG quality 78.
Full-resolution master images remain unchanged in the local experiment. The saved AI_PROXY judgments
were made on those masters, not on these compressed copies. Original and copy hashes and dimensions
are separately recorded in IMAGE_COPY_MANIFEST.csv. Existing CSV image hashes identify originals;
they were not silently rewritten to identify the compressed copies.

Tables, registration, review records, report, audit and included code are byte-identical copies.
SOURCE_FILES_SHA256.csv records their source paths and hashes. The actual original metric/scoring
implementations are included under code/external and code/MeshFlow, alongside the new pilot code.
code.diff describes these newly added pilot files against /dev/null, not a modification of old scorers.

Weights, RAW/POST geometry arrays, saved Gaussian inputs, trajectories and per-point arrays are excluded.
The full local files remain hash-addressed by the manifests. This is a review/re-aggregation package;
it cannot independently rerun geometric scoring without those excluded arrays and the local dependencies.
The recovery entry remains README_RESUME.md; it is not a promise that this small ZIP installs the model.

The model key is included separately under evaluation/blinding, after the anonymous reviews were saved.
Open REPORT.md for interpretation and material costs; use evaluation/gallery/index.html for figures.
