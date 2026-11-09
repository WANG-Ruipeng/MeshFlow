# Independent preparation review

PASS with interpretation limits. Zero model forwards and zero training.

- Verified all 40 frozen inference inputs and all 20 condition/query identities; N remains in 128..256.
- Independently reconstructed four sub4 conditions: child coordinates are exact FP32 source-edge midpoints; shared source edge IDs share midpoint IDs; original source winding is retained; per-parent triangle areas remain within the registered rounding tolerance.
- Verified 2048 physical query points and weights per sub4 condition are byte-identical to base. Recomputed child ownership through barycentric transforms, normals, and finite-segment boundary distances.
- Reconstructed each paired MT19937 stream on CPU and compared frozen arrays: 16 TRAIN/DEV noise pairs share exact prefixes, not necessarily complete arrays. All eight sub4 outputs use the complete unchanged free array of their own base request.
- PAIR04 is a near-duplicate-risk pair: independently recomputed mirrored C corner mean distance is 6.62407731807e-07 L. Retain its outputs, show it separately and report an exclusion sensitivity; do not present it as independent new-shape generalization.
- The registration group_uid is a parent-ID alias, not the source dataset deduplication group. Consult coverage_audit/supplement_v1 for true group metadata and source similarity.
- sub4 changes K, N and vertex-based metric denominators together. Interpret as operational representation sensitivity; it does not isolate token dilution or attention causality.

All 87 registered source/input/query hashes checked here remained unchanged. Detailed measurements and exact audit source hash are in preparation_review.json.
