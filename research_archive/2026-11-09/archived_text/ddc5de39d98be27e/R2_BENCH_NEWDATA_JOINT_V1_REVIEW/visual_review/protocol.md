# Bench new-data actual image review

This protocol retains the exact five-axis vocabulary of the preceding RELATION
and SHIFT reviews. It is a nonblind AI_PROXY review, not an independent human
study. The panel is 28 requests x 3 stages = 84 paired pages and 168 model-stage
cells. Columns are RELATION_V1_S1500 then NEWDATA_JOINT_S1500.

Inspect each full comparison PNG, both model columns, both azimuth -60/120 views
at elevation23 and both fixed front/side sections. Record both columns in the
same review round/view event. Contact sheets can aid navigation but cannot
replace full-page inspection. Blue is C, orange SEAM, gray BULK, green BRIDGE.
The source completion is not the unique correct style. Do not grade resemblance
to it, or use numerical metrics, head diagnostics, training curves, historical
ratings or column order to supply visual labels. Reused old DEV meshes still
require actual viewing in these new paired pages; old ratings are not copied.

- overall_shape: REASONABLE / INCOMPLETE / UNCERTAIN
- C_use: PLAUSIBLE / REPLACED_OR_IGNORED / UNCERTAIN
- support_retention: PRESENT / MISSING_OR_DAMAGED / UNCERTAIN
- visible_fragment_or_gap: YES / NO_OBVIOUS / UNCERTAIN
- layering_or_fold: YES / NO_OBVIOUS / UNCERTAIN

Judge axes separately. A complete-looking Bench can replace or ignore C.
UNCERTAIN covers occlusion, projection, finite-section ambiguity or ambiguous
input semantics; explain it in the note. NO_OBVIOUS is not absence certification.
Section endpoints cannot establish cracks, floating surfaces, intersections or
indexed connectivity. Visible folds/fragments are observations, not detector
results. Do not invent a joint-success score or trade C adoption against loss of
the main body or supports. Numeric topology/CF/FF/I and deleted faces remain
separate evidence.

CSV files are header-only at preparation. Append only genuinely inspected cells;
never prefill axes, image hashes, actually_viewed or observation notes. Every row
records the image hash, renderer receipt hash and gallery-definition hash, actual
view source, reviewer, UTC view timestamp, review round, view-event ID and a
concise image-grounded note. actually_viewed and views_and_sections_checked must
be true only after inspection. These are reviewer declarations tied to real
image files; hash validation alone does not prove that an image was inspected.
Save both columns of each viewed page in the same batch. Leave unviewed pages
absent; missing rows are pending, not failure, UNCERTAIN or NA. Never silently
replace original observations. Adjudication requires a separately registered
round and retained original rows; this collector does not merge duplicate cells.

CURRENT_TRAIN contains the nine selected training C (18 requests). The new
parent's R2 is SAME_PARENT_HELDOUT (2 requests), not an unseen-parent test. DEV
contains four held-out adaptation parents (8 requests), repeatedly used as a
development set; official pretraining novelty is not certified. Keep all three
groups and RAW / BASE_POST / CLEAN_FINAL_POST separate. Within a group, average
two noises within C, then conditions within parent, then weight parents equally.
Also retain raw counts and full category denominators including UNCERTAIN.

Assignments retain whole parents, both noises and all three stages: a_new_and_flat
has 24 pages/48 cells; b_train_and_backdev and c_curved_and_dev each have 30
pages/60 cells. R1 and R2 are viewed by the same reviewer. Shards can run in
parallel when reviewers are available; they do not create independent replicates.

The helper reads submitted review CSVs without editing them. It validates exact
cell identity, actual-view declarations, all five labels and real image receipts.
Incomplete reviews produce coverage/missing-cell records only. Numerical visual
summaries and paired/cross-stage transitions are emitted only after all 168
valid cells are present. No generation, rendering, model calls, rescoring, best
sample selection, extra samples or frozen-code changes are part of this review.
