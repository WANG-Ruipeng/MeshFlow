# Independent execution and resource review

PASS. Registered 24 new Euler50 outputs used exactly 1200 reserved/returned generator forwards; 24 new outputs each used exactly one official D0. There are also 16 verified historical RAW/POST pairs. Logical totals are 40 RAW and 40 POST. No unresolved/failed model calls or training updates were found.

All new RAW completed 62.298 seconds before the first D0 reservation. Independent array checks verified all 51 C states and all 50 Euler recurrences for each new RAW; POST retained the registered source C corner order.

Both stages had two configured partitions of 12, but both actually ran sequentially: measured generator partition overlap = 0 seconds; D0 partition overlap = 0 seconds. This is not evidence of two concurrent GPU workers or speedup. CPU work may overlap independently.

Observed card peak used 7084.0 MiB and minimum free 8896.0 MiB, from discrete snapshots; these are not continuous/process memory peaks. Unrelated allocations are included. PyTorch generator max allocated/reserved = 560.2/614.0 MiB; D0 = 1793.2/1824.0 MiB. No process-specific VRAM was returned by WSL/WDDM.

149 protected preparation/source/code identities remained unchanged; 616 recorded artifacts were SHA checked. Large checkpoint identity relies on the actual loaders' SHA checks and the generator after-state checks, rather than another large-weight hash pass.

See execution_review.json, resource_timeline.csv and worker_time_intervals.csv for exact times, counts and scopes.
