# Independent final evidence audit

Status: **PASS**. Numerical aggregation and final anonymous-review aggregation independently checked after the 73b visual records were fixed.

144 rollouts contain exactly 7200 generator RESERVE/RETURN pairs. The 144 D0 calls each have one proposal receipt bound to the registered official checkpoint. All 144 RAW outputs completed before the first D0 call, with a 127.919-second gap. No new training, backwards, model calls or geometry scoring were performed by this audit.

Recomputed 125 protected source/checkpoint/code hashes. Verified output hashes and preservation receipts. Recomputed all 28,800 metric-pair rows, 4,800 per-condition/parent averages, and the final 48 condition, 12 panel and 288 decision records. Each panel has **four parents**; eight/four noises are repetitions within C.

| POST, SMALL vs EXP67 | Outputs with >=1 valid local C/F edge | Adoption YES | Definite NO->YES | Definite YES->NO | UNCERTAIN->YES |
|---|---:|---:|---:|---:|---:|
| TRAIN_FAMILIAR | 12->15 / 32 | 16->16 | 1 | 1 | 0 |
| DEV_HELDOUT | 0->1 / 16 | 3->5 | 0 | 0 | 2 |

The familiar-panel edge-output increase is a local interface result; adoption YES remains unchanged, with one definite gain and one definite loss. DEV adoption YES rises from 3 to 5 through uncertain judgments becoming YES, not two demonstrated NO-to-YES fixes. A local valid edge is not adoption, an entire closed seam, or a globally valid mesh.

Actual 73b review: eight full request pages and one contact sheet, 48 categorical rows and three style rows, image hashes checked. Prior condition/experiment context was known; current names and numeric scores were not accessed before saving those rows. These are AI-proxy observations, not human ratings or certified perfect blinding.

Execution was one GPU worker with sequential model/stage phases, plus two CPU scoring workers. PyTorch allocated/reserved and whole-device nvidia-smi observations are not WDDM per-PID process peaks and do not prove a strict 6 GiB process cap. No concurrency speedup is claimed.

Audit-only engineering corrections: explicit UTF-8 decoding and RETURN schema adaptation. Neither correction changed experimental artifacts or scientific definitions.
