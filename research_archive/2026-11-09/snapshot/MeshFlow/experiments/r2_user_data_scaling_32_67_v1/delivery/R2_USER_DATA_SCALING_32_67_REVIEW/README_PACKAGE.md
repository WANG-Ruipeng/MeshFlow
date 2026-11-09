# R2 32-versus-67 review package

Start with REPORT.md, then evaluation/gallery/index.html for every registered RAW/POST request. All original per-output, per-condition and per-parent CSVs and actual review records are included.

This small review archive cannot independently train or regenerate models. Weights, RAW, trajectories, D0 proposals, POST arrays, source meshes, Gaussian arrays and query arrays remain local. local_artifacts.csv lists actual paths, sizes, recorded SHA provenance and explicit missing/NA states. Recorded weight SHA values were not recomputed during packaging.

The full existing JPEG gallery is copied unchanged; no image was redrawn, cropped or recompressed. Any existing source PNG remains local. Actual visual-review coverage is COMPLETE (840/840 rows).

code/new contains this experiment source; code/dependencies contains the actual small source files named by execution/scoring receipts. code.diff compares each new pilot file with the preserved same-name prior blocked-run pilot file, or /dev/null for a new file. It is not a fabricated Git commit.

inherited/ contains the actual old constructor snapshots, region face-ID lists, coverage matrix, complete acceptance/exclusion tables, all five frozen AI_PROXY input-review files and the original INPUT_REVIEW_GALLERY.html with its existing images. INHERITANCE_PROVENANCE.json records original paths, hashes and snapshot identities. input_review_decisions_compact.csv retains every original decision row and exact strings while omitting only the repeated quality JSON column. The full 82.6MB condition_decisions.csv remains read-only locally, with its original path, size and SHA in INHERITANCE_PROVENANCE.json; the projection does not replace the full original table. No input was newly reviewed or reconstructed.

The earlier blocked run and qualification-order deviation remain unchanged in inherited/. Use README_RESUME.md for local recovery; this archive authorizes no further scientific work.
