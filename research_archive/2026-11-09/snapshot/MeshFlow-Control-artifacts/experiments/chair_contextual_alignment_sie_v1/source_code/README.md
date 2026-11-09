# Source review evidence

This directory is a byte-preserving source snapshot, not a runnable model bundle.
No model, teacher weight, raw mesh, or training output is copied here. No ZIP is produced.

- `repository/`: all 94 files in the before-training freeze, plus the two new evaluation/teacher documents and necessary existing provenance documents and notices. Exact scope is in `source_inventory.json`.
- `original_scoring/`: the 19 **actual migrated standalone runtime scoring modules** used by this experiment. The historical directory name does not mean these are byte-identical to their earlier pre-migration files.
- `historical_scoring/`: 10 original small scoring sources from the read-only sibling workspace. Each old source hash and its separate migrated hash is verified in `scoring_identity_verification.json`.
- `repository/tools/contextual_alignment_{evaluation,analysis,gallery}.py`: the frozen experiment adapters; original metric definitions are unchanged.
- `repository/src/meshflow_control/training/context_alignment.py`: the new training-only attention-message capture/head. `repository/src/meshflow_control/models/native.py` and `models/backbone/{attention,equidit,utils}.py` preserve the exact model/capture dependencies as full source files.
- `repository/configs/coal_{fm,free,known}.yaml`: the three actual experiment recipes.
- `repository/tests/test_{context_alignment,contextual_alignment_evaluation,siglip_targets}.py`: task-specific CPU/contract tests. Capturing tests does not claim that any new test execution happened during source delivery.
- `identity/`: original before/at-training source identities and execution binding, copied without edits.
- `audit_tools/`: small run auditing helpers, distinguished from training/scoring code.
- `../code.diff`: 27 explicit new files as unified full-file additions against `/dev/null`, with exact byte reconstruction checked. This is **not** `git diff HEAD`: this independent repository has no HEAD. The initial source hash list covers only 47 src files; it cannot establish historical membership of every old tools/configs/tests file. Those supporting files are not falsely listed as new additions.

`source_inventory.{json,csv}`, `frozen_source_verification.json`, and `scoring_identity_verification.json` give the hashes, scope, classifications and snapshot checks. The source freeze is rechecked after copying. This is a source-only capture during the fixed run, not a final experiment completion audit.

## Notices

`LICENSE` and `THIRD_PARTY_NOTICES.md` are preserved byte-for-byte. The NVIDIA notice in the captured `models/backbone/utils.py` remains intact. Its independent distribution-license basis has not been verified; the repository MIT notice does not override that retained file notice. Historical source headers are also preserved. Model/teacher weights and datasets have separate terms and are not included.
