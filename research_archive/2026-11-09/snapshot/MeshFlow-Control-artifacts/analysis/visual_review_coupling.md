# P1 visual review: assigned three-parent scope

AI_VISUAL_REVIEW completed for 84 outputs: parents 866e7248434968051cc7750d0afca796, 86ec9c796a6c1789e792598ed58dd1cc and bdd29e651e5f6fb2b079317292bdc5d4; candidates 0/1/4/5; labels START/A500/B500/C500/A1000/B1000/C1000.

All 12 blind pages were actually displayed and notes saved before the 12 corresponding reference pages were opened. Every output was inspected in front three-quarter, back three-quarter, front, side and wire views. Human review: **NOT_RUN**. No usability rate or binary usability label was produced.

The blind phase hid the source completion, while model labels were visible. This is not a model-blind trial. The source is one possible completion, not the only valid chair.

## Observations

- **bdd29e... n4:** all three 1000-step outputs have visibly ragged pointed/slitted free back panels and long under-seat triangles. C1000 has a particularly conspicuous outward triangular sheet at the middle/lower back. These were already noted blind and are local deterioration relative to the coherent 500-step silhouettes; the main chair remains recognizable.
- **86ec9... n0:** B1000 has a large diagonal sheet across the opening beneath the blue crossbar; C1000 has a smaller flap. For **n5**, A1000/B1000 have clear under-seat side flaps; C1000 shows less broad flap in that view. These are local geometry observations, not proofs of disconnectedness or self-intersection.
- **866e7... n4:** 1000-step back apertures are clearer than the wedge-obstructed 500-step ones, while B1000 retains conspicuous upright triangles near the seat/back junction. **n5** includes a C500 above-seat triangle and A/B1000 under-seat flaps; C1000 looks cleaner in that specific side opening. A single image detail does not establish a global ordering.
- **86ec9... n4:** all outputs have coherent solid backs, while the source has slats. This difference is treated as style, not failure. In **866e7... n0**, START's rectangular/slatted alternative likewise is not rejected solely for differing from the round-hole source.
- **bdd29e... n0/n1/n5:** all labels retain recognizable chairs. Seat/back layers, narrow slits and orange surfaces projected within blue C can be seen, but depth/contact and small arm differences are often unresolved. The triangular blue seat-side piece belongs to C and must not be counted as a new generated fragment.

This assigned subset supports neither a universal B-over-A nor C-over-B visual ranking. Similarities and mixed local changes remain in the per-output table.

## Evidence and limitations

- `visual_review_coupling.csv`: 84 distinct request/label rows with separate C compatibility, overall organization, local-defect, style-versus-fragmentation and certainty fields; both page paths and SHA256 hashes are included.
- `visual_review_coupling_blind_notes.json`: notes retained before the reference-view phase.
- `visual_review_coupling_evidence.json`: actual page-view sequence and all 24 original PNG hashes.

The 1650x2552 blind pages and 1650x2870 reference pages contain five view columns. Individual chairs occupy only part of each tile. Silhouette and large triangular sheets are readable; tiny seams, watertightness, duplicate coplanar faces, exact intersections and topological connection cannot reliably be inferred. C appearing visually present is not a bytewise-preservation audit. Wire-line density is not itself a defect. Different crest, slat count, leg thickness or source mismatch is not automatically invalid.

No scoring, rendering, model invocation or mesh modification was performed for this review. An oversized shell write was rejected by the OS before execution; the same completed review was then saved in bounded text chunks.
