# Curriculum analysis and training-accounting preparation

Prepared only. No actual curriculum logs/results were analyzed, no model/OT/scoring/generation was executed. Synthetic tests: 46 PASS (16 new analysis + 11 original analysis + 8 new accounting + 11 original accounting). Final test stdout and JUnit XML are adjacent.

Run after the corresponding immutable results are complete, from E:/MeshFlow-Control in WSL:

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B tools/audit_recipe_curriculum_training.py \
      --run /mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B tools/analyze_recipe_curriculum_results.py \
      --out /mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1

Training accounting defaults to audit/curriculum_training_accounting/ and refuses overwriting an existing audit; --output may name a fresh audit directory. If either completed.json is missing it returns PENDING and reads no mutable logs/latest checkpoint. Once complete, it reads only course logs, registration/plan/paired tables, R2 prior initialization, and actual immutable endpoint/export bytes. It never deserializes tensors. Internal generator/AdamW/RNG hashes remain receipt claims bound to actual file SHA.

The accounting explicitly maps stable origin indices to current course step/slot. It verifies exact ordered R2 paired rows, 48000-sample multiset, fixed final16000-slot MIX suffix, H_ALL, per-update valid-free denominator, input hashes, actual parent/family/ratio exposure, physical calls, failed/unclosed calls and completed replay. Reordering changes batch denominators; scalar exposure equality does not imply equal per-scalar optimizer weight. Expected hash I/O is approximately four large endpoint/export files plus scalar logs; it can overlap with separately budgeted work without CUDA use.

Analysis directly reads main/evaluation (select R2) plus course/evaluation (both courses), with matching request-file hashes and C/noise/query/denominator identities. It writes analysis/course/recipe_curriculum_table_analysis_v1_<identity>/ and an immutable receipt. Missing result files return PENDING without creating a result directory. All 192 primary outputs and 192 raw auxiliary rows are retained, with the96 A20 diagnostics summarized separately. Both C ratios have ALL/original6/extended2 summaries. Original6/extended2 at C20 denotes parent membership, not a newly registered cohort.

Pairs are C20_40_MIX minus R2, C40_20_MIX minus R2, and C20_40_MIX minus C40_20_MIX. All numeric fields receive noise/parent distributions, fixed denominators, tail identifiers, and applicable concentration/leave-one-parent-out summaries. Nonnumeric, boolean, all-NA and status fields remain in the catalog and original rows. All six diversity pairs remain; larger variation is not a quality certificate. ReferenceGT/S/E have no promotion direction or gate. No automatic ranking, new metric definition, score call or candidate choice is introduced.

The analysis reuses only stdlib-based pure-table helpers from the old analyzer. Accounting reuses read/hash/receipt helpers; its course-sensitive scan/update checks are separately adapted source. Original modules and the frozen97 execution files remain unchanged. The initial builder file is preparation provenance only; do not rerun it over the final source.

