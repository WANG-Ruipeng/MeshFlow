# Actual AI_PROXY visual review

All **144 POST outputs** were actually reviewed in the fixed **48 whole-object original PNG sheets**, plus all **8 predeclared USER_MIX local original PNG sheets**. RAW, D0_OFFICIAL, D_CONT_NO_ROLE and D_ANCHOR are displayed together. Three agents inspected non-overlapping image assignments and saved their original per-output CSVs, notes and image-hash receipts. The merged receipt preserves their provenance. This is non-blind AI_PROXY review, not a human rating or topological proof.

The per-output CSV separates C use, boundary continuation, whole-shape quality and thin structure. Uncertain connection is retained as uncertain. No request was selected by its score, and no completion is ranked by resemblance to the unique source mesh. Small numerical differences can exist even where whole-object appearance is similar.

The detailed independent observations follow verbatim. Their common visual finding is that the three POST variants generally retain very similar large-scale shapes and C relationships; no repeated, unmistakable anchor-specific boundary adoption is established by these sheets. Persistent placement, layering and thin-support problems are recorded individually rather than hidden by a single aggregate label.

There are material **shared adaptation improvements**, so the outputs are not declared identical: in R2_FROZEN 33ec n1, both adapted denoisers retain more thin-leg structure than D0; in R2_CONT_USER bdd n1, both remove a thin forward projection at mid-back. The C offsets persist in both examples. These improvements support a distinction between ordinary denoiser adaptation and the incremental effect of the anchor-role embedding. USER_MIX 33ec n0 still shows marked leg thinning after all three POST variants.

# USER_MIX actual visual review

Reviewer: `/root/anchor_data`; non-blind **AI_PROXY**, not human evaluation. All 12 whole-object original PNGs and all 8 predeclared local original PNGs were actually displayed. The 36 POST rows cover D0_OFFICIAL, D_CONT_NO_ROLE and D_ANCHOR for every assigned request; RAW is the reference. Observations were written without looking at numerical scores or source GT.

The main result of this visual review is the similarity of the three POST variants at the level of C use and overall shape. I do not see an unmistakable anchor-specific adoption of the supplied boundary in these 12 requests. This does not exclude smaller numerical differences.

- **33ec seat n0:** the blue patch remains below a separate gray seat in all three POST outputs. All three thin the RAW support legs markedly, with at least one nearly line-like support. The adapted variants retain more visible width on some legs than D0 but do not cure the condition placement.
- **33ec seat n1:** a complete armchair remains; C is a tilted interior seat patch. The layered/occluded C relationship is similar across variants.
- **534b back n0/n1:** n0 retains a protruding blue panel behind a bulky layered gray back. n1 retains a large slanted gray sheet across the blue panel and a top loop. No variant visibly resolves these back relationships.
- **866e seat n0/n1:** n0 has plausible seat placement, yet local/wire views show a separate blue outline and nearby gray sheets; attachment remains uncertain. n1 largely buries the blue plane inside a layered boxy seat. The three variants retain similar arrangements.
- **86ec back n0/n1:** narrow recognizable chairs remain, with blue strips along and above a gray back frame. n1 retains broad triangular free sheets at the lower supports. Neither their legality nor topological joining is inferred from the image.
- **bdd2 back n0/n1:** n0 retains conspicuous lateral/upper C overhang. n1 retains a distinct upright blue plane separated from the main gray back in side view. The mismatch persists through all variants.
- **f596 back n0/n1:** both retain blue frame tips above and offset from the gray back. n0 retains the circular back opening and irregular faceted lower supports; n1 is simplified strongly while maintaining the large-scale chair. No anchor-specific boundary repair is apparent.

The review intentionally separates a plausible chair silhouette, plausible semantic placement of C, and actual boundary connection. Neither hard C preservation nor gray/blue proximity is counted as certified connection. Fine intersection legality and near-zero gaps require the saved numerical/topological results. Small triangle-count differences are not treated as visual quality wins.

---

# R2_CONT_OLD AI_PROXY visual review

Actually viewed all 12 whole-object PNGs, covering 36 POST outputs. This is non-blind AI_PROXY review, not a human score or proof of attachment. Four fixed views per request were inspected, including the opposite wireframe.

Across these figures, D0, D_CONT_NO_ROLE and D_ANCHOR usually have very similar silhouettes and C relationships. I do not see a convincing, repeated anchor-specific improvement in C adoption or boundary continuation at this render scale. This is a visual limitation/observation, not an assertion that numerical differences are zero.

Persistent concrete problems include the blue patch below the generated seat for 33ec n1; blue seat overhangs for 866e n1; the laterally displaced blue back patch for bdd29 n0; and blue top/side projections around a different gray back surface for f596. The 86ec n0 back is spatially more compatible, but shared topology is not certified by its appearance. The 86ec n1 outputs retain large diagonal sheets in the under-seat space. A thin under-seat needle remains for 866e n0, and a small overhead fragment remains for 33ec n0.

Most objects remain recognizable chairs, and I see no repeated gross loss of legs/back after either trained denoiser relative to D0. Some very thin structures and occluded interfaces cannot be judged conclusively at whole-object scale. Each of the 36 rows records C use, boundary continuation, overall quality, and thin structure separately; no mutually exclusive success label was imposed.

---

# R2_CONT_USER and R2_FROZEN visual inspection

Completed actual inspection of 24 original whole-object PNGs, 72 POST outputs. Each sheet was viewed in front, side, oblique and opposite-wireframe views; all four RAW/D0/NO_ROLE/ANCHOR columns were inspected. Review is non-blinded AI_PROXY, not human scoring. No model/scorer call was made. The receipt hashes the PNGs actually displayed, not JPEG previews.

Across these two generators, most denoiser differences are local. The blue seat is often buried beneath a gray generated seat, while blue back patches/frames often overhang or remain offset from gray back structure. The images do not show a clear ANCHOR-only improvement in C adoption or boundary continuation. Projected proximity never establishes shared-edge connectivity.

R2_CONT_USER: examples 33ec n1 and 866e n1 are box armchairs with C occluded inside the seat; C use therefore remains uncertain. The 534b n0 back panel and f596 frames are visibly offset; bdd n0 has conspicuous lateral overhang. In bdd n1 both adapted denoisers remove a thin forward projection present in D0 at mid-back, while keeping the main legs. The blue/gray back-plane offset remains. This is a shared adaptation change, not an anchor-specific join gain.

R2_FROZEN: the most important positive shared-adaptation case is 33ec seat-medium n1. D0 visibly loses substantial thin-leg structure; the two adapted outputs retain more of the four-leg silhouette. Some supports still look thin/open, so this is visual preservation rather than a connectivity certificate. The blue seat still lies beneath/protrudes from the gray seat. Persistent bad C relationships are visible in 534b n0, bdd n0, and f596 n0/n1. F596 n0 also retains irregular skirt/spike-like foot and under-seat geometry after all three denoisers.

No conclusion that all POST outputs are identical is warranted: the thin-leg and mid-back-projection cases are material contrary evidence to that statement. Conversely, those shared gains do not demonstrate an effect specific to the new anchor-role embedding. Keep the independent quantitative evidence and full per-output observations alongside these images.
