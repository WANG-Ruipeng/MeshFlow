# Three-family production input contract: actual CPU result

The fixed run completed with **one occupancy-head forward**, zero generator/D0/OT/backward calls, and no new noise or training. The existing OLD_S15000 head, 24-feature preprocessing and `logit >= 0` threshold were unchanged. Parameters were FP32 under CPU BF16 autocast; this does not establish bitwise equivalence to historical GPU BF16 inference.

| Condition | Actual result |
|---|---|
| 27eb79 R1, N255/K85, open C | Head activated 20 of 31 boundary edges. All 20 lack a legal source Y neighbour. More broadly, none has any original Y face containing both required endpoint IDs. Exact fixed-anchor encoding is blocked. |
| 27eb79 R2, N255/K148, closed C | Existing empty-SEAM layout; zero head calls. All 255 target faces retain one-to-one identities, exact coordinates and reversible source winding. PASS. |
| 945214 R1/R2 | N448; skipped without head calls. Production N guard unchanged. |
| 585673 R1/R2 | Target fidelity/part-area/total-area holds; skipped without head calls. |

There was no source-owner correction to the predicted mask, no duplicate face, no added degenerate target, and no target deletion. The R1 failure is stronger than merely failing a legal-contact policy: every activated edge also lacks any source-ID-compatible original Y face. This is a mismatch between this production layout and this particular reference completion, not proof that no other valid completion could attach there.

Both per-condition ledgers record `LAYOUT_FROZEN_BEFORE_Y` before `SUPERVISION_READ_STARTED_AFTER_LAYOUT`. R1 has exactly one `HEAD_FORWARD_STARTED` and one matching return; R2 has neither. Minimum absolute R1 logit was 0.04345703125, recorded descriptively without a device-stability claim.

The single passing condition is the pre-reserved **same-parent R2**, not a training condition. R1 failed; R2 cannot be moved into training to manufacture a usable pair. Therefore this frozen small pilot adds **zero eligible main-training conditions and zero main-training parents**. Its R2 endpoint pass demonstrates representability, not generation, use of C or connection quality.

Before/after resources are preserved. The initial resource-only inline shell command lost quoting and failed before computation; its corrected stdin script is documented separately. No neural attempt was retried. Stage wall time was approximately 5.44 seconds including CPU setup; the sole head call occupied approximately 0.126 seconds according to its ledger. No sampling or training performance inference follows from these timings.

Primary evidence: `HEAD_PROTOCOL.json`, `RUN_PROTOCOL.json`, `COMPLETE.json`, and each condition's `layout_receipt.json`, `ledger.jsonl`, `packing.json` and `per_face_roundtrip.json`.
