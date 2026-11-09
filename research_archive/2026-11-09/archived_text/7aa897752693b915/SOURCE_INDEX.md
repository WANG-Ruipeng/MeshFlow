# Schedule40 source delivery index

This delivery separates the original frozen scientific dependencies from independently registered engineering additions. All live sources and copied source bytes were checked by SHA256. This index is not a claim that every planned experimental phase has completed.

- Original frozen dependency manifest: [audit/source_hashes_before.json](audit/source_hashes_before.json), 63 files, unchanged.
- Existing experiment code manifest: [audit/new_source_hashes.json](audit/new_source_hashes.json), 12 files, unchanged; 5 also belong to the original frozen set.
- Existing later CPU-analysis manifest: [audit/postfreeze_analysis_hashes.json](audit/postfreeze_analysis_hashes.json), 3 files, unchanged.
- Independent engineering additions: 6 files, separately registered below; they do not replace the original frozen 63-file manifest.
- Unique copied sources: 79. Full-addition review diff: 21 files.
- Complete identities: [audit/delivery_source_hashes.json](audit/delivery_source_hashes.json).
- Previous review diff retained exactly: [code_diff_before_delivery.diff](audit/engineering/code_diff_before_delivery.diff).
- Current [code.diff](code.diff) contains the 12 existing repository sources, 3 existing audit sources and 6 engineering additions as full additions; every added source round-trips to its exact bytes.

## Independent additions and limits

| Source | Purpose and boundary |
| --- | --- |
| tools/run_schedule40_dispatch_r1.py | Independent dispatch recovery correction. Clears a surviving worker's old parallel-retry eligibility; preserves bounded same-tick double-OOM handling. Original controller and scientific definitions remain frozen. |
| tests/test_schedule40_dispatch_r1.py | CPU mock regressions for the dispatch boundary; no real GPU workers. |
| experiment_audit/guard_serial_tail.py | Existing verified controller-only SIGSTOP/SIGCONT guard. Does not pause or signal the GPU worker; copied identity matches its own original guard snapshot. |
| experiment_audit/analyze_actual_resources.py | CPU log audit assigns phase by timed worker identity, including training/sampling name reuse, and adds guard resource logs. Does not replace the original accounting output. |
| experiment_audit/capture_environment.py | Existing environment-capture source, archived without executing it again. |
| experiment_audit/finalize_source_delivery.py | This idempotent source-copy/index/diff finalizer; no scientific execution. |

## Complete source inventory

Group tags: F63 = original frozen dependencies; R12 = existing new repository sources; A3 = existing analysis scripts; E6 = separate engineering additions.

| Copied source | Groups | Bytes | SHA256 |
| --- | --- | ---: | --- |
| [experiment_audit/analyze_actual_resources.py](source_code/experiment_audit/analyze_actual_resources.py) | E6 | 23539 | `97fe8cc04bae65052b536476582131b5f88f303e8f405252e186cf31e9f35339` |
| [experiment_audit/capture_environment.py](source_code/experiment_audit/capture_environment.py) | E6 | 1044 | `fc8b6f42c364aa163b2c91d80ddc683a2fedbac222d084e19ea45773df09f395` |
| [experiment_audit/capture_final_analysis_sources.py](source_code/experiment_audit/capture_final_analysis_sources.py) | A3 | 2005 | `a266e50204dddf56aaa7f4aa9cd30551e4ac2c9ff6c666822a8206e7b3959426` |
| [experiment_audit/finalize_source_delivery.py](source_code/experiment_audit/finalize_source_delivery.py) | E6 | 15242 | `7a3ef010975789b91bd29f33571b85cf86aa5e5e5422c0edb6276daec934644e` |
| [experiment_audit/guard_serial_tail.py](source_code/experiment_audit/guard_serial_tail.py) | E6 | 10383 | `4c03ca5c1c6dd276afbb074215d6dc2c8fdcd3131483d61be4144f40749ecddc` |
| [experiment_audit/summarize_fixed_panel.py](source_code/experiment_audit/summarize_fixed_panel.py) | A3 | 21540 | `8c3cc3a60c3cc2fe551a41f7c1ca208c9b99d867cdbfdb8c8802c529914da65e` |
| [experiment_audit/verify_final_checkpoints.py](source_code/experiment_audit/verify_final_checkpoints.py) | A3 | 25263 | `60cf6c586f7ec3cee614eddcf20c66f23f5505ef066b6d0631d773b3cacfea4e` |
| [src/meshflow_control/__init__.py](source_code/src/meshflow_control/__init__.py) | F63 | 106 | `c30b647e43bdca95d21bb195e9c09745be8ec0ef4351f93d7bcad245a0794b87` |
| [src/meshflow_control/__main__.py](source_code/src/meshflow_control/__main__.py) | F63 | 60 | `772087eebf3a3bd370e75e84dfc1a6d20423712603779edd84b19104fa028bbb` |
| [src/meshflow_control/accounting.py](source_code/src/meshflow_control/accounting.py) | F63 | 3105 | `425aa3d64c72d55a22b1b4370d06a8fcd959ee919609e2a0498deb8b373c9a2e` |
| [src/meshflow_control/artifacts.py](source_code/src/meshflow_control/artifacts.py) | F63 | 1404 | `2a6fbe4c00d937e6e0de0d0456b07b59147a174fc4127ba6a6b35b85372f95a0` |
| [src/meshflow_control/checkpoints.py](source_code/src/meshflow_control/checkpoints.py) | F63 | 10297 | `9a4de162be3a55e4e796a12640f59cc80614e46c612dcd497fc1a607280ba566` |
| [src/meshflow_control/cli.py](source_code/src/meshflow_control/cli.py) | F63 | 4346 | `bd9e66288e1eed6964109410095360ff2257f7383fed8424e425addddec0f6b2` |
| [src/meshflow_control/config.py](source_code/src/meshflow_control/config.py) | F63 | 2076 | `e3f50ecef6dabb8816ff65394ff1b764b1c2f9df3088065583fe173e60430371` |
| [src/meshflow_control/data/__init__.py](source_code/src/meshflow_control/data/__init__.py) | F63 | 67 | `829001a015c06a0a203464105eeee560a68ba2c43a7bded803ec673e679b3446` |
| [src/meshflow_control/data/contextual_stream.py](source_code/src/meshflow_control/data/contextual_stream.py) | F63 | 6753 | `2959e57dfcfd68d5cbc4bfffedb5f26f3786f2215a9894179938313b6afefa20` |
| [src/meshflow_control/data/dataset.py](source_code/src/meshflow_control/data/dataset.py) | F63 | 4413 | `4ed0c8068809766f69c7e24f0020002764e9a1f6c3760ff5e41dd6ac776c227e` |
| [src/meshflow_control/data/io.py](source_code/src/meshflow_control/data/io.py) | F63 | 2247 | `ec3dbcf69e499ad567f81950fd743dcb45c384f0b89da59a855c184ebb2de182` |
| [src/meshflow_control/data/prepare.py](source_code/src/meshflow_control/data/prepare.py) | F63 | 10267 | `5a12d86dc0fecce0f8e8506434deb2f0e8aab811eeef995f4f3c955c22a1e224` |
| [src/meshflow_control/data/schedule40.py](source_code/src/meshflow_control/data/schedule40.py) | F63, R12 | 25642 | `b5300d3523c6e9ef849d56b7163526e2f3c0634a4d27ef8692c132973ccdc8fc` |
| [src/meshflow_control/data/stream.py](source_code/src/meshflow_control/data/stream.py) | F63 | 14622 | `90e5adb13a857483d91fbecbde033112585120ab6215aec2e3c6307b042a5b05` |
| [src/meshflow_control/evaluation/__init__.py](source_code/src/meshflow_control/evaluation/__init__.py) | F63 | 302 | `1d37639e6beb37e24f8c8fabb65a45633ea39297e185496a1732fcfcf4c58bca` |
| [src/meshflow_control/evaluation/aggregate.py](source_code/src/meshflow_control/evaluation/aggregate.py) | F63 | 5664 | `331af2191ed28b69de96722c3f53952bd3be919abce57276c2e342a041a6f2ae` |
| [src/meshflow_control/evaluation/control.py](source_code/src/meshflow_control/evaluation/control.py) | F63 | 30008 | `3de139a98beb766697ed163e402a568cd81be69f1028b1e29f471f6df98954b3` |
| [src/meshflow_control/evaluation/diversity.py](source_code/src/meshflow_control/evaluation/diversity.py) | F63 | 2012 | `f7f79e87941ed17a1190f3e3b5ce13c0e5ebe9a2dc7bd369918f8682cbab6f15` |
| [src/meshflow_control/evaluation/evaluate.py](source_code/src/meshflow_control/evaluation/evaluate.py) | F63 | 13294 | `21486dbc6695e51a9581a81904a9592cd35f5fd6c9297b7f47cefbe528fab142` |
| [src/meshflow_control/evaluation/finite_edges.py](source_code/src/meshflow_control/evaluation/finite_edges.py) | F63 | 5205 | `3889d84895b91648bab4103341b313a94e0153afdb87836561fbfd6bac957f98` |
| [src/meshflow_control/evaluation/geometry.py](source_code/src/meshflow_control/evaluation/geometry.py) | F63 | 3172 | `10aadbf3d82f718420c15ba7694a08127d20ccaec7076eb5e714e1a07373e0b5` |
| [src/meshflow_control/evaluation/hashing.py](source_code/src/meshflow_control/evaluation/hashing.py) | F63 | 296 | `b8b5f59bbacff9f14db8b26ab95b0cef594e55f47ead1ea0da35c60a632b6c75` |
| [src/meshflow_control/evaluation/intersections.py](source_code/src/meshflow_control/evaluation/intersections.py) | F63 | 14755 | `23a2e8c8deefcceb51ace60ad8d8a58d352d7c38bdcba2e8f85d6ceabd447d86` |
| [src/meshflow_control/evaluation/io.py](source_code/src/meshflow_control/evaluation/io.py) | F63 | 2384 | `56350fed1c799015cc50f85b8c40f6ad868ffbdab005dd0db767b9e7d6a81d06` |
| [src/meshflow_control/evaluation/legacy_quality.py](source_code/src/meshflow_control/evaluation/legacy_quality.py) | F63 | 19702 | `7616d3807d2d507ae63bdbdd4c06c39a0232dceb175c677eb8d2a69e61fca9e9` |
| [src/meshflow_control/evaluation/pair_audit.py](source_code/src/meshflow_control/evaluation/pair_audit.py) | F63 | 3054 | `2e1e1b5520d808d6f11e3660373118d309dfadf18dd2e3cf90f15a7856ae3d8d` |
| [src/meshflow_control/evaluation/quality.py](source_code/src/meshflow_control/evaluation/quality.py) | F63 | 4175 | `be19e48fdb2ac4f13fbb9fc8c892872090cd4999f87e58505b3ae15b20f48044` |
| [src/meshflow_control/evaluation/reference.py](source_code/src/meshflow_control/evaluation/reference.py) | F63 | 9220 | `4102e646bca8d9021a590ab94151d001bc34d5a891878dda279d6bb372035461` |
| [src/meshflow_control/evaluation/registry.py](source_code/src/meshflow_control/evaluation/registry.py) | F63 | 4356 | `006533af1ef4104221e101ca22c69a2cfc3da223cf61f444b3a22b71d0e954d1` |
| [src/meshflow_control/evaluation/render.py](source_code/src/meshflow_control/evaluation/render.py) | F63 | 6029 | `74bc5d3c7cb8fc30641af3b763d11895d462a95706dcc010c8c9b118689f6660` |
| [src/meshflow_control/evaluation/seams.py](source_code/src/meshflow_control/evaluation/seams.py) | F63 | 17775 | `9ce6927710d80eedec19aa10e2100ade582cf00e6942e9de4ef987618a005e4b` |
| [src/meshflow_control/evaluation/shape.py](source_code/src/meshflow_control/evaluation/shape.py) | F63 | 6063 | `783466f59436ef1640801bd532ee20bf100db83ce650b86b9adafe3ac8495785` |
| [src/meshflow_control/evaluation/surface.py](source_code/src/meshflow_control/evaluation/surface.py) | F63 | 3527 | `1b5863d105309e800aff5bcd8c0457fb0935debe70bda647e36699253d3bf470` |
| [src/meshflow_control/mesh_io.py](source_code/src/meshflow_control/mesh_io.py) | F63 | 3907 | `21c4c9b34b692011c25cc0bf0b5796a629ccbd39889145df9c96fed753873e34` |
| [src/meshflow_control/models/__init__.py](source_code/src/meshflow_control/models/__init__.py) | F63 | 86 | `1237e536e8db5805408f02c64b204fea657b851dbfa3ba20f529fae9eb604f00` |
| [src/meshflow_control/models/backbone/__init__.py](source_code/src/meshflow_control/models/backbone/__init__.py) | F63 | 54 | `e206e3eb6ce1f50e3b76364b91594f647983d7d9d84bbfe932c28cfe7b476657` |
| [src/meshflow_control/models/backbone/attention.py](source_code/src/meshflow_control/models/backbone/attention.py) | F63 | 4650 | `922601d6c42ed860c7dc2d19b1a0c7bc4a3b2c4feded1e701e77fb7c5a79998b` |
| [src/meshflow_control/models/backbone/equidit.py](source_code/src/meshflow_control/models/backbone/equidit.py) | F63 | 13608 | `f24b55461dfb791be3f8195f34a7b636f63e8a404ac28b3e4a72caf71069e0ab` |
| [src/meshflow_control/models/backbone/utils.py](source_code/src/meshflow_control/models/backbone/utils.py) | F63 | 4633 | `c9023363ea42d59c5d0a4236e7985fd216a6574355dc44f232baeba6ef91fe40` |
| [src/meshflow_control/models/geometry.py](source_code/src/meshflow_control/models/geometry.py) | F63 | 13890 | `634914de68b02f88e335a6cf1bae8959ff5c07e7849094829e42e5d892b5c1be` |
| [src/meshflow_control/models/native.py](source_code/src/meshflow_control/models/native.py) | F63 | 14725 | `e0c5d95787142f1ae8283aff37c044028fa98187ef6ab2d98f029c8f09e6cb80` |
| [src/meshflow_control/models/readout.py](source_code/src/meshflow_control/models/readout.py) | F63 | 8468 | `93ec50905e762e817039fc632e2b1bad52c0a5c9019a9bc17f4c1b50fa7e3117` |
| [src/meshflow_control/precision.py](source_code/src/meshflow_control/precision.py) | F63 | 8235 | `4dd7717a480175bcd612f2bd71ebe075553b2a95dd43988f2e12d6d60fafa4b3` |
| [src/meshflow_control/runtime.py](source_code/src/meshflow_control/runtime.py) | F63 | 8396 | `b2600af3d7f618ec294e3ad5122b6fc0ffa71bee2c57dd424e1ef1f5456e9a5d` |
| [src/meshflow_control/sampling.py](source_code/src/meshflow_control/sampling.py) | F63 | 3463 | `30ca9c68d552b03ed7b429a28d935fc68923615d9096e6c107f310ea822ba4d0` |
| [src/meshflow_control/teachers/__init__.py](source_code/src/meshflow_control/teachers/__init__.py) | F63 | 79 | `f93f7b9e3efcfb8cdbce4f436d57dd809385673117ce783d7909d785cbb11353` |
| [src/meshflow_control/teachers/siglip_assets.py](source_code/src/meshflow_control/teachers/siglip_assets.py) | F63 | 3511 | `fd9b55c47459397c3877ea174b6f669b597816ab6706dd0b2532f013089d635f` |
| [src/meshflow_control/teachers/siglip_render.py](source_code/src/meshflow_control/teachers/siglip_render.py) | F63 | 5239 | `959ef43056825f596968cbae4e01e007aa2ff30cccd5f3208ae00ff93930502a` |
| [src/meshflow_control/teachers/siglip_targets.py](source_code/src/meshflow_control/teachers/siglip_targets.py) | F63 | 24656 | `f653980c4e1be974a452523614873a9bf52b476da77e2ec8482a324e8550c57e` |
| [src/meshflow_control/training/__init__.py](source_code/src/meshflow_control/training/__init__.py) | F63 | 76 | `d361eec09b0adcdac09611635260665c7f7b8d4799d271cab7c8f7f5eae1cd5f` |
| [src/meshflow_control/training/context_alignment.py](source_code/src/meshflow_control/training/context_alignment.py) | F63 | 6480 | `fa84937808ded18c8bd60811a36e6b2148cfdae690c898b96898429b04c9c237` |
| [src/meshflow_control/training/context_budget.py](source_code/src/meshflow_control/training/context_budget.py) | F63 | 1986 | `47c9ab29550c42deff8685ebb8c2aeb49225abd1b9e68102ef13c72fbaebc55e` |
| [src/meshflow_control/training/context_checkpoint.py](source_code/src/meshflow_control/training/context_checkpoint.py) | F63 | 13112 | `5bd29c553833498d820a057807f9d5ce6d47f38803509d689d309e37fe32b9b2` |
| [src/meshflow_control/training/context_contract.py](source_code/src/meshflow_control/training/context_contract.py) | F63 | 1832 | `634b1f515b4b84e3ca5d9b0dfe528036f8485dfb2aadcb6a1caaabfe5e6283c1` |
| [src/meshflow_control/training/context_preflight.py](source_code/src/meshflow_control/training/context_preflight.py) | F63 | 11370 | `7cb143416ff302accd370db62576340fb5ee1a7c65dda5e106a5061c71ae14ca` |
| [src/meshflow_control/training/context_trainer.py](source_code/src/meshflow_control/training/context_trainer.py) | F63 | 19765 | `ebdb44d5c625808f52cc242bbb9389a18b0cae430ac4ceec73372d37fb0c294b` |
| [src/meshflow_control/training/coupling.py](source_code/src/meshflow_control/training/coupling.py) | F63 | 3423 | `50143af01c7fe354151423783a5b741778b02553249257654bc9fe66ac72705d` |
| [src/meshflow_control/training/losses.py](source_code/src/meshflow_control/training/losses.py) | F63 | 916 | `8e84aa20bfec5865b932104a64b0ad3539957b2b186999735b1e61abf274f7c9` |
| [src/meshflow_control/training/schedule40.py](source_code/src/meshflow_control/training/schedule40.py) | F63, R12 | 34511 | `702e30848602bee0aa53a8950da83e62bab338741fb65383b0ca4d04009132ac` |
| [src/meshflow_control/training/trainer.py](source_code/src/meshflow_control/training/trainer.py) | F63 | 16993 | `3c03866c231f73d69967bb7b26629de332b0981db7cd02c20b8e6cbb649a3c34` |
| [tests/test_schedule40_accounting.py](source_code/tests/test_schedule40_accounting.py) | R12 | 4675 | `108af59e5ca989af0afcde99db43d5bea3e6bd7c43e36d540e7a547401d4b3f4` |
| [tests/test_schedule40_conflict.py](source_code/tests/test_schedule40_conflict.py) | R12 | 5962 | `5809e102b30c64169022e255f4516f4d1128d497e75814e5426caa0f8e5a9d3c` |
| [tests/test_schedule40_data.py](source_code/tests/test_schedule40_data.py) | R12 | 7358 | `92d0b0a93083dda383171b241fc1d23487d2caf2d1a730dc33959fc8c21205ec` |
| [tests/test_schedule40_dispatch_eval.py](source_code/tests/test_schedule40_dispatch_eval.py) | R12 | 4431 | `42b50c51ae43a67e651f010695619d9af29c5004d9645efd1ee2df26bd80c625` |
| [tests/test_schedule40_dispatch_r1.py](source_code/tests/test_schedule40_dispatch_r1.py) | E6 | 6501 | `5970b9bcab2ecb8443555bb14ecd2f9dea7fe230abad7c0d6a47b034d8ee53c0` |
| [tests/test_schedule40_train.py](source_code/tests/test_schedule40_train.py) | R12 | 5623 | `4bdae478ddc3f85b217124319c9e56f4a0d01f077176831e25880da5b5720a02` |
| [tools/analyze_schedule40.py](source_code/tools/analyze_schedule40.py) | R12 | 34927 | `ff802b85b10ea4154e3a80fa95c4609bd71ac16d12e6d8a920982ad2fe829b3b` |
| [tools/evaluate_schedule40.py](source_code/tools/evaluate_schedule40.py) | F63, R12 | 21476 | `3366dcfdd021b53c973f6d9d9d3d28f1fe263016ec23234c7a53fe923278d412` |
| [tools/run_schedule40.py](source_code/tools/run_schedule40.py) | F63, R12 | 17668 | `5785f7c85db8affda14ab6fa9729f5c2bc35468d7274db9a31c147120f535a65` |
| [tools/run_schedule40_dispatch_r1.py](source_code/tools/run_schedule40_dispatch_r1.py) | E6 | 20097 | `57756ff6e0e5d51a1395d006e014957b119b9ccf2be4519033041b91091cabbd` |
| [tools/score_schedule40_conflict.py](source_code/tools/score_schedule40_conflict.py) | R12 | 20415 | `d4e7e12074485c2293b8603922bbc9fdc6a26110a00a545ea115ed7e45d77931` |
| [tools/train_schedule40.py](source_code/tools/train_schedule40.py) | F63, R12 | 1145 | `07fba3d025e2017ddac65d1f6dc42e5f834f1bc919a428fc18adb01e86116290` |

## Re-run

```bash
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1/audit/finalize_source_delivery.py
```

Repeated runs verify the same identities and preserve identical existing bytes. A changed or missing previously registered source blocks finalization instead of silently replacing it. Only the review diff may be updated after its exact previous bytes are archived. No weights, RAW meshes or experiment results are copied by this tool.
