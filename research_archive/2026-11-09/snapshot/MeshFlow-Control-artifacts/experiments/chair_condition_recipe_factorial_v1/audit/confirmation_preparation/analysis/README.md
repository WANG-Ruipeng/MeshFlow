# Confirmation analysis/review preparation
Only new tools/tests were added; frozen97 source files are unchanged. Final synthetic suite:49 PASS.
Metadata-only actual registration audit:64 requests,16 query banks,198 registered files hashed; n8/n9/n10/n11 PASS. No confirmation quality or images were read.
Peer schedule40_train read-only checkpoint-receipt and phase/label-selection review: PASS, no execution blocker.

Commands after the respective complete prerequisites (do not rerun training):
~~~
cd /mnt/e/MeshFlow-Control
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B tools/analyze_recipe_confirmation_results.py --out /mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B tools/prepare_recipe_confirmation_review.py --out /mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1
~~~
The analyzer fixes C20_40_MIX minus R2_MIX_H_ALL. It saves128 current and128 first-stage pair rows; all49 metrics (47numeric); per-noise/per-parent effects; ALL/ORIGINAL6/EXTENDED2 at both ratios; complete parent LOO; all six diversity pairs and diversity LOO; transparent cross-repeat parent/group effect signs. No output-level pairing of n0/1/6/7 with n8/9/10/11.
Output: analysis/confirmation/recipe_confirmation_table_analysis_v1_<inputhash>/.

The anonymous manifest helper writes audit/visual_review/confirm/review_manifest.json and never opens private_method_mapping.json. It verifies128 originalPNG hashes and copies only explicit request/image metadata, Method1/2 and assigned reviewers (32/48/48PNG). Prior main/course knowledge and AI review are disclosed; actual viewing remains pending.
