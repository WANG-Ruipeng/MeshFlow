# Confirmation initial-review schema and immutable merge

This is preparation only. Never count these templates as actual review notes. Fill observations and evidence only after actually viewing the original PNGs.
All three reviewers have prior main/course knowledge; this is AI review, not an independently blinded human study.

Authoritative safe input (after rendering):
    RUN/audit/visual_review/confirm/review_manifest.json

Each reviewer writes:
    RUN/audit/visual_review/confirm/REVIEWER/requests/REQUEST_ID.json
    RUN/audit/visual_review/confirm/REVIEWER/cross_noise_initial.json

Copy request identity, ratio/noise, PNG paths/hashes from the safe manifest. Use request_note_template.json as the schema. Each request has exactly colored+neutral image evidence and two notes named Method 1 and Method 2. Every image requires actually_viewed=true, all_five_views_examined=true, actual UTC time, tool name and actual viewing evidence. These are reviewer records, not independent proof of human viewing.
Each note requires original observation and uncertainty strings. Extra original fields (organization, collapse/missing parts, isolated/repeated surfaces, C relationship, common context) are preserved unchanged. Do not put true labels/recipe into initial notes. No invented numerical ratings.
The cross-noise file has one observation/uncertainty entry per assigned parent, using cross_noise_template.json.

Assignments: root33ec/86ec:16requests32outputs32PNG; schedule40_data341b/bdd/866e:24/48/48; schedule40_train534b/c1b/f596:24/48/48. Every parent covers both ratios and n8/9/10/11. No secondary recheck enters initial coverage.

After every assigned PNG was actually viewed and all original notes saved, run:
    python -B tools/merge_recipe_confirmation_review.py --out RUN --stage freeze-review --reviewer REVIEWER

This writes the reviewer's initial_freeze_receipt.json with all source hashes and timestamp. Repeating the command verifies the same files and cannot silently update the receipt. Do not edit frozen notes. A later correction/recheck needs a separate file and must not overwrite the initial evidence.

Only after ALL THREE reviewers have frozen initial notes:
    python -B tools/merge_recipe_confirmation_review.py --out RUN --stage merge

The merger validates all three receipts, each source-note hash,128 distinct image paths/bytes and explicit viewing records BEFORE reading private_method_mapping.json. It then writes audit/visual_review/confirm_complete/per_output.csv + per_output.json + coverage.json + image_coverage.json + cross_noise_initial_sources.json + bilingual README.md.
The merge preserves all64 original request documents and cross-noise wording, creates128 distinct request/alias rows, and does not double count colored/neutral images. It performs no quality-table read, render, scoring, model call, subjective numeric score or ranking.
