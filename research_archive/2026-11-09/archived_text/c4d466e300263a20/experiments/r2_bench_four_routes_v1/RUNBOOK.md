# Recoverable four-route entry

Run from repository root in the existing WSL environment. Python:
`/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`.

`python -B -m r2.pilots.bench_four_routes_v1.run status` only reads status.
`...run resources` records actual resource information to stdout without model calls.

Each training route must have a frozen registration and CPU gates, then a fresh
resource-aware `...run dispatch --route exposure|seam --peak-mib <measured-estimate>`.
Run `...run train --route exposure|seam` once. Logs and PID receipts are retained.
`--resume` is allowed only for an exact saved `INTERRUPTED_SAFE` boundary. FAILED/OOM
or unrecorded work is never automatically retried. Existing completed work is not rerun.

The exposure arm ends after exactly3,000 new updates; the SEAM arm after1,500.
Each has its own optimizer/RNG and immutable input registration. CPU backend and
Best-of-four stages have separate branch entries and receipts under their directories.
No stage is authorized to expand the registered experiment or choose a best checkpoint.

## Stage entry points

The entries below use the same Python environment and repository working directory.
Read completion receipts first. A command encountering existing incomplete scientific
calls must stop for investigation; these are not instructions to replay them.

- `...hybrid.run run`: registered46 CPU mixtures; zero models. Existing per-request
  completion receipts protect completed work. `...hybrid.render_stream` renders
  completed cases under the original frozen display contract.
- `...bestof.run sample|score-raw|select|d0|score-post|collect`: fixed B candidate
  generation/scoring in that dependency order; `...bestof.closeout`
  and `...bestof.delivery` account for the existing92 candidates and23 selected
  outputs. Do not generate more noises or D0-process unselected new candidates.
- `...seam.evaluation sample --model S --estimated-peak-mib 2800`, then
  `...seam.evaluation d0 --model S --estimated-peak-mib 3500`; the full46 RAW
  barrier precedes all D0. CPU `raw --models S --workers 3`, `post --models S
  --workers 3`, then `collect`. `...seam.report_results render|tables|visual`
  provides the frozen gallery and summaries.
- `...exposure.closeout training` validates the actual3000-update endpoint.
  `...exposure.evaluation sample --model EXPOSURE --estimated-peak-mib 2800`, then
  `...exposure.evaluation d0 --model EXPOSURE --estimated-peak-mib 3500` uses the
  same full RAW barrier. CPU `raw --models EXPOSURE --workers 3`, `post --models
  EXPOSURE --workers 3`, then `collect`.
- For exposure rendering use `...exposure.render_labels_v2 --workers 3`,
  which calls the recorded `prepare_review` metadata adapter and corrects exactly
  one inherited W/M caption string before rendering. Do not use the inherited
  `report_results render` entry. Image geometry, views and sections are unchanged.
  `...exposure.report_results tables|visual` and `...exposure.evaluation_closeout`
  validate complete scores and actual reviews; `...exposure.lossplot` plots real logs.

Every new GPU wave still requires a current resource measurement. The estimates
above are starting estimates from this run, not universal memory guarantees.
Completed gallery images and actual review notes are immutable; reviewing code
never fills missing human/AI observations from scores. The final report identifies
the precise files and weight hashes; completed training is not resumed again.
