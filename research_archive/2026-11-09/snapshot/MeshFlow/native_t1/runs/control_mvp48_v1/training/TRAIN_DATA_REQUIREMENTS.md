# Session B data requirements and receipt

Current data: repair_v2, all 81 frozen file hashes and sizes verified.
B consumed 32 training objects, 128 training patches and 8000 fixed records using FrozenStream.
No OT, model forward, optimizer update or CUDA initialization occurred in this receipt.

| Input | Required role | Current status |
|---|---|---|
| DATA_READY.json | Freeze identity and hash inventory | Bound |
| registration/data_contract_repair_v2.json | Data API and retained/source index semantics | Bound |
| data/manifest.json, splits.json, patches.json | Object identity, 32/8/8 split, patch membership/interfaces | Verified |
| data/meshes/<uid>.npz | Original provenance, retained topology, FP32 target/C/pre-OT coordinates | 48 frozen, train32 consumed |
| data/train_plan.json | 1000 batches x8; UID/patch/Gaussian/t/permutation seeds | 8000 records validated |
| data/load_sample.py | Shared exact loaders and stateless RNG recipes | Hash pinned and consumed |
| data/independent_repair_validation.json | Independent audit | PASS |
| Official Chair last.pt | Official EMA backbone only | Restored and SHA256 verified |

Per-record plan fields: sample_index, effective_batch_index, slot_index, uid, patch_id, N, K, gaussian_seed, t_seed, permutation_seed.
Mesh/patch fields consumed: model_vertices, source_face_vertex_ids, pre_ot_full, full_target, constraints, source_face_ids, free_source_ids, bbox_diagonal, interface.
N counts retained genuine faces. source_face_ids/free_source_ids index retained dense rows; original_* fields are traceability only.
pre_ot_full is captured before official final x2. full_target and C already include x2; they are never rescaled.
Validation/test files are hash-verified but never used to fit statistics, select training batches, or train. Geo fits no statistics.

Verified official asset: 2,084,287,208 bytes; SHA256 bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d.
Restored location: checkpoints/official_chair_last.pt. Original Downloads path remains absent. Bench/old tuned models cannot substitute.
After this exact asset is restored, the registered real-data prechecks and 2500 updates can run; no extra arm or retry was added.

Receipt: registration/train_data_ack.json. Binding: registration/train_data_binding.json.
