# Joint-shift actual image review protocol

Prepared while JOINT_SHIFT_S1500 is still training, before any new evaluation images are available. This is a nonblind AI_PROXY review, not an independent human study. Review all 144 registered comparison pages: 48 fixed requests times RAW, BASE_POST, and CLEAN_FINAL_POST. Each page has three model columns, so the complete review contains 432 model-stage records.

The left-to-right order is **RELATION_V1_S1500 / USER_V2_S1500 / JOINT_SHIFT_S1500**. RELATION is the primary paired reference: identical 26 parents, 91 conditions, initialization, budget and original sample stream, with joint C/Y translation being the new training variable. USER is a secondary historical reference. Do not treat the column order or labels as a quality ranking.

Use both inherited views (azimuth -60 and 120, elevation 23) and both fixed-position front/side sections on every complete page. All columns and stages use the same inherited frame, scale and section planes. Blue is the supplied C, orange is SEAM, gray is BULK and green is an added bridge. Do not grade resemblance to a unique source completion. A plausible new shape need not reproduce its source parent.

Use the same five-axis vocabulary as the preceding RELATION review:

- `overall_shape`: `REASONABLE` / `INCOMPLETE` / `UNCERTAIN`
- `C_use`: `PLAUSIBLE` / `REPLACED_OR_IGNORED` / `UNCERTAIN`
- `support_retention`: `PRESENT` / `MISSING_OR_DAMAGED` / `UNCERTAIN`
- `visible_fragment_or_gap`: `YES` / `NO_OBVIOUS` / `UNCERTAIN`
- `layering_or_fold`: `YES` / `NO_OBVIOUS` / `UNCERTAIN`

The first three axes assess plausible whole Bench structure, C's participation in that structure, and visible support. A complete-looking Bench can still replace or ignore C. Judge each axis separately. Use `UNCERTAIN` where occlusion, input ambiguity, a finite section, or projection prevents a reliable observation. `NO_OBVIOUS` does not certify absence of defects. Apparent fragments, gaps or folds are image observations, not exact intersection or connectivity measurements. Section endpoints alone do not prove cracks, isolated faces or floating surfaces.

For each page, actually open its full comparison PNG, inspect all three columns, both views and both sections, and then record all three model cells in that same review round. Contact sheets may help navigation but cannot replace the full-page review. Do not copy or transform old USER/RELATION ratings, and do not infer labels from CF/FF/I, reachability, loss curves, numerical rankings or historical conclusions. The previously generated reference meshes are read-only, but their images must be viewed afresh in this comparison. Do not prefill any visual labels before seeing the corresponding new page.

Every record must contain `request_id`, `parent_id`, `condition_id`, `label`, `stage`, `page_id`, `image_path`, `image_sha256`, `actually_viewed: true`, the five axes, and a concise image-grounded `note`. The file wrapper must declare `reviewer_kind: "AI_PROXY"` and `blind: false`; an `actual_view_source` field may document `tools.view_image`. Check the completed gallery receipt for the PNG hash; no fabricated or placeholder image hashes. Write records in small completed batches to avoid losing review work. If interrupted, leave the unviewed pages pending and report the exact remaining count rather than completing them from memory or metrics.

Assignments preserve the previous four-parent groups: `dev_a`, `dev_b`, and `history_train`. Each group contains 16 requests, 48 pages and 144 model-stage records. The task file lists every expected page ID and prospective PNG path; those paths are not claims that the images already exist or have been viewed.

Historical TRAIN means four parents seen during COMMON12000; it is a regression group. All 48 requests are outside the current 26-parent adaptation pool, but historical TRAIN is not unseen over the full training history. Historical DEV remains a separate group. Official pretraining object-level exposure is not established by this review.

Keep RAW, BASE_POST and CLEAN_FINAL_POST results separate. In particular, improvement before D0 need not survive D0 and geometry processing. Summaries should describe repeated gains and losses relative to RELATION and then USER, retaining contrary examples and uncertainty. Exact topology, legal connections, deleted-face counts and CF/FF/I remain separate numerical evidence. Any root adjudication must be an additional documented record, not a silent replacement of an original review.

This protocol authorizes no additional samples, new noise, checkpoint selection, training, model calls, scoring changes, image filtering or edits to frozen code. Only the actual review records and explicitly documented review validation may be added after gallery completion.
