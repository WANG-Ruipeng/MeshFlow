# Existing curriculum preflight

Status: PASS, CPU metadata only. No authorization, model, GPU, new OT, new noise or sample-array reconstruction was performed.

Both C20_40_MIX and C40_20_MIX under explicit H_ALL contain exactly the existing R2 48,000-record multiset. Every existing paired row, including shared input, input/label hashes, OT key, free-coordinate count and origin index, is exactly equal to the corresponding R2 origin. Each curriculum ends with R2's identical ordered last 16,000 records and paired rows (updates 4001–6000).

The first 4,000 R2 mixed batches are split in adjacent pairs into pure-ratio batches. C20_40_MIX uses 2,000 pure20 then 2,000 pure40 updates; C40_20_MIX reverses these blocks. Every resulting batch has eight distinct parents and four clean/four augmented samples. The frozen trainer's exact metadata validators were extracted without imports and executed for every batch. This is not a model or training test.

The complete sample-level free supervision total is equal. However, the original loss normalizes by each effective batch's total free scalar count. Regrouping pure-ratio batches changes that denominator and hence sample weighting within the optimizer trajectory. This experiment compares the registered curriculum recipe, not an abstract ordering intervention guaranteed to preserve every scalar's effective optimization weight. The identical final MIX suffix does preserve its batch denominators.

All 97 frozen source hashes still match. No old plan, paired table, source or metric result was changed. The original heterogeneous candidate-selection pools remain unchanged and must still be disclosed. The existing CLI needs an explicit all schedule and a separately authorized output/recipe registration before any execution.

Evidence: preflight.json, segment_exposure.csv, per_batch_metadata.csv, source_hashes.json and frozen_source_verification.json. Target equality here is evidence from the existing actual-precompute hashes (shared hash contains x1 and OT maps), not a new numerical reconstruction of cached arrays.
