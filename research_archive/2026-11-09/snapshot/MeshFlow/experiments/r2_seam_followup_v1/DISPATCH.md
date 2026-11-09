# Dispatch record

The frozen budget is 112 CPU binding cells and 28 B_LONG_LAYOUT RAW + 28 single-D0 outputs. No training or head calls. Inputs and output directories belong to each independent stage.

Initial WSL snapshot: RTX5080 16303MiB total, 13017MiB free, GPU10%; 24 logical CPU, approximately61GiB RAM available, no listed compute process. Keep2GiB additional GPU headroom. Start one actual authorized GPU sampler and overlap independent CPU binding; no synthetic probes. Additional workers require representative memory and utilization evidence. Actual resources and timing are saved under resources/, layout/ and binding/.

The implementation review and cached source-topology query are CPU-only. They do not launch Ideas3/4 training. All RAW must exist before the layout D0 phase. No unchanged historical mesh is rescored merely to reproduce a previous table.

Engineering event: the first layout attempt loaded the frozen generator but encountered a cached-NPZ scalar-type mismatch before any generator forward (no calls.jsonl, no RAW output). The original attempt and code hash are retained. Explicit repair converts zero-dimensional N/K scalars to Python integers at the runtime boundary; geometry/layout/noise bytes and the scientific definition remain fixed. The repaired execution consumes the original 28-request budget; this is not an unreported model-output retry. Detailed receipt resides in layout/.
