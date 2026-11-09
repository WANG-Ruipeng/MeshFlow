# Actual image review instructions

This is AI_PROXY observation, not blind human review or a topology certification. Do not read new output scores, rankings or aggregates until the review CSV and receipt are frozen. Method mapping may be known; declare this honestly.

Assignments are fixed in assignments.json by parent, before any output render. Inspect both RAW and POST pages for every assigned request, each with front, side, oblique and wire views. Three methods appear in each page: A=USER_SMALL, B=USER_EXPANDED_67, C=HIST_USER_MIX_FROZEN. Blue is fixed input C; gray is generated free. Do not infer an image was viewed because a renderer completed.

Use all original AI_PROXY_review_template.csv identity columns. Per output record C_adoption, C_floating, C_replaced_by_other_surface, C_wrong_layer, whole_body_complete, support_missing, fragments_or_folded_layers, visible_join as YES / NO / UNCERTAIN (UNKNOWN for not viewed). Add concrete short notes about what is visible, not a generic success label. Both whole-shape problems and C-use problems can co-occur. A required small triangle, legitimate thickness, source-style mismatch or a different leg count is not automatically failure. A close silhouette or shared projection is not proof of a legal common edge. Mark uncertain when occlusion or resolution prevents judgment.

For each actually observed page compute its SHA256 and place it in reviewed_image_sha256 for all three corresponding method rows. review_status=REVIEWED_AI_PROXY; reviewer=AI_PROXY; blind=False. Preserve registered rows not inspected as NOT_REVIEWED with UNKNOWN fields. Do not fill unknown observations by metrics or by copying the other noise, stage or method.

Write only your chunk CSV and a small receipt under review_work. Record actual images viewed, rows reviewed, start/end UTC, and numeric_ranking_seen_before_review=False when true. Do not run models, scoring, new rendering or image editing during review. Root combines the chunks after identity/count verification, then opens numeric results.

Operational notes fixed before seeing output pages:
- C_adoption: YES when the declared seat/back C plausibly serves that role in the visible completed object, even if exact joining remains uncertain. NO for clearly isolated/replaced/misplaced C; UNCERTAIN if the views do not decide. This is not a numeric success rate or a structural certification.
- C_floating: YES only for visible separation that leaves C without a plausible connection to the object. A legitimate detached chair part or a boundary of a HALF patch is not automatically a failure.
- C_replaced_by_other_surface: YES when a distinct generated main seat/back visibly substitutes for the intended C. Do not label ordinary thickness as replacement without visible evidence.
- C_wrong_layer: YES for clear vertical/depth misplacement relative to the generated functional surface; do not use the source mesh's unique style as the target.
- whole_body_complete / support_missing: judge whether a plausible object and supporting structure are present. Different leg counts, styles, armrests, or stool forms are not automatic failures. Use UNCERTAIN for occluded or ambiguous support.
- fragments_or_folded_layers: YES for visibly unexplained fragments/folded or collapsed layers, not merely because triangles are small, skinny, or individually visible in wire view.
- visible_join: YES for visibly continuous attachment across views, NO for a clear gap, UNCERTAIN for touching projections/occlusion. Even YES is visual evidence only; legal shared-edge topology is assessed separately after review.
