# NATIVE_S12000 full image review: part03 (AI_PROXY)

Actually inspected16 existing image pages (sorted request_id indices32..47), each with RAW/POST, two surface views and fixed front/side sections. All16 requests are DEV. Saved32 stage observations and16 paired comparisons. No metric values were read to assign labels, and no generation, rescoring, rendering or model calls were run.

The images separate recognizable bench shape from visible use of C. Both 9fd5 seat-slat requests retain a blue slat well above the generated gray seat. fd355 lower-frame n0 has a separate gray bench above/offset from the blue frame; n1 has a much more plausible support arrangement. These are image observations, not topological connection judgments.

POST frequently flattens the gray seat or panels while making free supports very thin. This is particularly visible in the9fd5 requests, the curved supports of f495 leg n0, the seat strips of f5d3 back n0 and the feet of f5d3 seat n1. Plausible blue back/frame placement can coexist with this quality cost. Other images retain parallel seat/back layers or pointed gray extensions.

The source part descriptions are construction diagnostics, not semantic inputs or unique required completions. Exact shared connectivity is NOT_ASSESSED_FROM_IMAGES and topological disconnectedness is UNKNOWN_IMAGE_ONLY throughout. Section traces can be missed by the plane; a short or missing line alone does not prove a disconnected triangle or hole.

| Request | RAW to POST visual observation |
|---|---|
|9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0|Seat smoothing is accompanied by strong thinning of the free opposite legs; the blue end-frame role remains visually plausible.|
|9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n1|The frame placement is retained, while POST makes the opposite free supports substantially thinner.|
|9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n0|The seat-height mismatch is unchanged; the gray body is smoother but its legs are much thinner.|
|9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n1|No visible adoption improvement; smoothing coexists with free-leg thinning.|
|f495bd67af8d9f0cadd1cf65e8dc4de__COMBO_PART_ASSEMBLY__n0|POST simplifies the gray lower layer, while the ambiguous upper/lower assembly relation and thin protrusions remain.|
|f495bd67af8d9f0cadd1cf65e8dc4de__COMBO_PART_ASSEMBLY__n1|The input surface is more visible after POST, but nearby gray layers and thin extra supports remain.|
|f495bd67af8d9f0cadd1cf65e8dc4de__LEG_SUPPORT_1__n0|POST makes the curved free supports considerably thinner; the blue leg remains visually ambiguous within them.|
|f495bd67af8d9f0cadd1cf65e8dc4de__LEG_SUPPORT_1__n1|The blue-leg height relation is unchanged; gray rail and support surfaces become thinner.|
|f5d370e9fe5b7452dbbc9440457e303e__BACK_SURFACE_1__n0|The back-board role is retained, while POST substantially thins the free seat strips.|
|f5d370e9fe5b7452dbbc9440457e303e__BACK_SURFACE_1__n1|Gray and blue back layers remain separated; local smoothing does not resolve the ambiguous duplicated back structure.|
|f5d370e9fe5b7452dbbc9440457e303e__SEAT_SURFACE_1__n0|The macro bench is retained; the blue seat remains above a separate gray seat-layer trace.|
|f5d370e9fe5b7452dbbc9440457e303e__SEAT_SURFACE_1__n1|The layered seat relation persists and the small free feet become thinner after POST.|
|fd355d06238853c7f0d9805ad6c2059c__COMBO_SEAT_SUPPORT__n0|The plausible assembly use is retained; small smoothing does not remove the lower layer or pointed gray extensions.|
|fd355d06238853c7f0d9805ad6c2059c__COMBO_SEAT_SUPPORT__n1|Macro arrangement is stable; POST mostly smooths existing panels and leaves the visible layers/flaps.|
|fd355d06238853c7f0d9805ad6c2059c__LEG_SUPPORT_2__n0|The separate gray-bench/blue-frame arrangement persists, while POST thins the gray legs.|
|fd355d06238853c7f0d9805ad6c2059c__LEG_SUPPORT_2__n1|The plausible support arrangement is stable across POST; small gaps and pointed gray extensions remain.|

See review.jsonl for stage-specific observations and observed_label_counts.json for counts of these visual labels only. This shard is not the complete48-request result and does not select a checkpoint or change training.
