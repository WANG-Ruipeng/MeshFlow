# Final DEV visual consistency summary

All four candidate reviews are COMPLETE for the same 16 DEV requests (8 parents x 2 noises); all four full visual gates are FAIL. Human usability: NOT_RUN. This synthesis reads saved reviews and selections only.

Candidate identities are original TRAJECTORY_REPLAY and TRAJECTORY_REPLAY_CAPPED at exactly 500 and 1000 updates. Each is compared with explore_REPLAY_FM_CONTROL at the matching update count; START and same-step original TR remain separate comparisons.

| Candidate | Matched teacher | Full visual gate | Body regression vs teacher | Body regression vs START |
|---|---|---|---|---|
| explore_TRAJECTORY_REPLAY_500 | explore_REPLAY_FM_CONTROL_500 | FAIL | 341bca n0 | 341bca n0 |
| explore_TRAJECTORY_REPLAY_1000 | explore_REPLAY_FM_CONTROL_1000 | FAIL | 341bca n1 | 341bca n0, 341bca n1 |
| explore_TRAJECTORY_REPLAY_CAPPED_500 | explore_REPLAY_FM_CONTROL_500 | FAIL | 341bca n1 | 341bca n1 |
| explore_TRAJECTORY_REPLAY_CAPPED_1000 | explore_REPLAY_FM_CONTROL_1000 | FAIL | None flagged | 341bca n0 |

The cap1000 failure against START at 341b noise0 is shared with substantial teacher deterioration. It is not evidence that the cap alone caused that body failure. Conversely, a missing teacher-relative collapse flag does not make cap1000 pass the START constraint or establish usability.

Important local failures at the final endpoint: cap f596 noise0 adds raised side sheets and a long downward triangle; cap 86ec noise1 becomes bulky and overlaps more known structure; cap 534 noise0 adds an under-seat needle; cap bdd noise0 has larger back cuts. Both replay candidates retain c1b noise1 back/crown fragmentation relative to teacher.

Do not carry intermediate failures forward without checking the endpoint: cap341b noise1 recovers from its severe500 body failure at1000; capbdd noise1 no longer has the deep500 hanging sheet. OriginalTR86ec noise0 also partially recovers its large500 crown/back defect. These recoveries coexist with new endpoint defects and do not overturn the full gates.

| Candidate | Metric-positive / visually WORSE selected requests |
|---|---|
| explore_TRAJECTORY_REPLAY_500 | 33ec1e n1, 534b0e n0, bdd29e n0 |
| explore_TRAJECTORY_REPLAY_1000 | 341bca n1 |
| explore_TRAJECTORY_REPLAY_CAPPED_500 | 33ec1e n0, 33ec1e n1, 534b0e n0, 86ec9c n1, c1b8c2 n1 |
| explore_TRAJECTORY_REPLAY_CAPPED_1000 | bdd29e n0 |

At cap1000, bdd noise0 is both the largest nominal S gain (+0.1612875296 relative to teacher, or +16.1288 percentage points) and the retained visually WORSE counterexample because its back is more cut/fragmented. The full JSON records the frozen conservative positive criteria; some other selected counterexamples qualify through I/E improvement with S within its guard, not through S gain alone.

Both combined seven-column galleries are complete: C / START / FM_CONTROL / REPLAY_FM_CONTROL / original TRAJECTORY_REPLAY / CAPPED / GT, at the exact same500 or1000 step. Every registered DEV request is present in every model column, with four shaded views plus wireframe and S/I/E overlays. Each gallery contains480 unique mesh renders,560 laid-out mesh slots (C/GT reused between noises), and240 feedback panels. No original arm is hidden by representative selection.

Source image hashes remain in the four full reviews. This check verified manifest availability/layout and bound current review hashes into each representative manifest; it did not rerender, rescore, run a model, or read CONFIRM. Round-hole gains on866e are shared by controls and must not be described as unique to replay.

Full sources, exact identities, hashes, counterexample deltas, and defect transitions: FINAL_VISUAL_SUMMARY.json.
