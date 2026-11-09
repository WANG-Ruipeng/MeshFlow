# Verify / resume

This four-request CPU backend experiment is COMPLETE. Do not run any model.

`python -B -m r2.pilots.bench_high_joint_coverage_v1.geo_only verify`

checks protected inputs, frozen geometry/scoring code, complete receipts and all
geometry/cascade files. Completed `run` is verification-only; partial science
attempts stop for explicit review rather than silently replaying.

`geo_report collect` and `geo_report render` reuse sealed outputs. The literal
review in `geo_review.py` records actual view_image events, not generated labels;
do not rerun it over the immutable review JSON. No additional experiment is needed.

REPORT table FF, CF and small-angle columns are percentages; other columns use
their original count denominators. I remains primary NA (zero/2048 interior
queries; minimum32), with legacy numeric fields preserved only for traceability.

No D0 or model was called. Four original GEO/source exports, four HINGE_OPP015,
four ALPHA, four HOLE passes and eight expanded original metric evaluations ran.
The old RAW/BASE/FINAL twelve scores were reused read-only. CPU workers2, threads1.
No failed science call, restart, replay, source mutation, checkpoint or new noise.
