# Completed R2/R3 accounting

Status: PASS. Full three-arm accounting: PENDING.

Only immutable completed R2/R3 endpoints and their completed logs were read. R4 and mutable latest.pt were not read.

|Recipe|Unique updates|Physical updates|Exposure rows checked|Replay|
|---|---:|---:|---:|---:|
|R2_MIX_H_ALL|6000|6000|48000|0|
|R3_40_H_LATE|6000|6000|48000|0|

- Only completed R2/R3 audited. R4 completion has not been asserted.
- Actual immutable file bytes hashed; tensor hashes and optimizer lineage from completed save receipts, not tensor deserialization.
- Input exposure tables and training logs were fully checked; small audit bundle keeps complete aggregate/count results and source hashes, original per-row CSVs remain canonical.
- Worker wall/update durations reflect concurrent training, not isolated throughput benchmarks.

Checks not passing:
