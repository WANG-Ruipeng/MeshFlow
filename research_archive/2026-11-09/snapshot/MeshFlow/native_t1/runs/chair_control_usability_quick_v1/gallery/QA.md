# Gallery QA

Status: PASS. CPU only; no model/GPU calls.

- 64 main candidates and 24 nested display entries (8 reused C1 entries); 80 unique raw outputs.
- 104 PNGs, each2080x468, with four fixed orthographic views and one wireframe view; all images decode.
- 8 common parent frames cover every raw vertex. No parent required expansion beyond the original GT-based frame.
- Blinded bundle has92 local PNG assets, no GT images/model names/UIDs, and a one-to-one80-row author mapping stored outside the bundle.
- Main64-row and nested24-row human templates contain no reviewer or scores; human_usable=NA and user_study=NOT_RUN.
- All80 direct final-C byte checks and stored sampler C51 records pass; raw file hashes still match actual result records.
- Four actual real-output figures were viewed: G01-R03, G01-R07, G08-R02, G12-R02. This was rendering QA, not a human usability study or complete AI_PROXY assessment.

See qa.json for the exact image identities and checks.
