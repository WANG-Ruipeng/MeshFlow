# Frozen evaluation for Direct Context Readout V1

This package is CPU-only evaluation, not a model or mesh repair. It imports no historical workspace or retired experiment at runtime. Necessary legacy numerical functions were physically migrated into this namespace; original paths, hashes and changes are in [evaluation_source_map.json](evaluation_source_map.json).

## Panels and input separation

reference.prepare_panels(legacy_root, artifact_root) makes independent arrays and immutable manifests under data/evaluation/. Inference NPZ contains only FP32 C, z, explicit N/K and masks. Full reference geometry, source IDs, Gamma, fixed GT queries and visibility arrays remain in evaluation_reference/; they are never model arguments.

P1 uses original eight A20 DEV parents with n0/n1 and new n4/n5. Original START n0/n1 is verified against actual C, Gaussian, 51-state trajectory, raw/result/score/vector/export hashes, then copied to external assets. New seeds use SHA256 of NATIVE_MESH_DIRECT_CONTEXT_READOUT_V1|EVALUATION_NOISE|20261007 plus panel/condition/candidate, first four bytes little-endian. Actual new FP32 arrays are checked against each condition's old n0..n3. These are new noises on observed DEV parents, not unseen-parent generalization.

P2 freezes a two-triangle seat, the same seat split into 32 triangles, a two-triangle back, and the four-triangle seat/back combination. N=192, four fixed noises per C. Seat width/depth=2.4 model units, y=-0.55; back z=1.2, y=-0.55..1.85. Sparse contexts are pressure tests. Two seat tessellations are not independent shape evidence. No hidden target, Gamma or visibility requirement is created.

P1 scale is original full-parent L0, evaluation-only. P2 uses one predeclared constant: diagonal of the public training-coordinate canonical cube, sqrt(3)*2/0.3762. No output bbox normalizes metrics.

## Public API and output manifest

evaluate(manifest, out, workers=2) accepts JSON with rows. Each row has request_id, panel, cohort, condition_id, parent_id, candidate, N, K, input_path/input_sha256, reference_path/reference_sha256, label, raw_path/raw_sha256, result_path. RAW supports raw[N,9] or historical output[N,9]; trajectories support trajectory[51,N,9] or historical path. Rendering uses the same completed manifest.

Results include per-output JSON/CSV, metric-registry long CSV, per-parent CSV, checkpoint profiles, paired noise/parent tables and all six within-condition candidate pairs. Missing candidates remain missing pairs; diversity is not silently averaged over survivors. Default two workers, at most four workers with two CPU threads each.

## Metrics actually computed

- Hard contracts: finite coordinates, explicit N/K, C triangle coordinate/order/winding bytes, actual Gaussian, all 51 C states when supplied; NPZ/PLY verification comes from separate exporter receipt. Missing export/trajectory proof is unknown. Exact coordinates do not certify original source vertex-ID graph preservation.
- CF/FF: original complete finite triangle scan with independent verifier. CF proper fraction uses free faces in proper C/free crossings only; positive-area coplanar pairs remain separate. FF illegal fraction is the free-face union touched by proper crossings or coplanar positive-area overlaps. Denominator=current free count N-K. Unresolved/degenerate pairs never become disjoint. This is not whole-mesh Ri because C/C is not scanned.
- Raw I: original 2048 C area samples, proper-crossing and near-parallel duplicate band union. Bands=0.005L; duplicate support requires C distance from exact boundary >=0.01L, unoriented normal angle <=15 degrees. Denominator=all sampled C weight. P2 calls the same function with only C/F and fixed L; no target or visibility. Thin walls can trigger duplicate coverage; this is not a semantic failure flag.
- P1 S: original trusted source Gamma, 64 length-weighted midpoints per edge, strict distance <tau L, unoriented tangent angle <=20 degrees, finite projection support and corresponding owner C/free CF qualification. Taus=0.005/0.01/0.02, primary=0.01. Lower and possible upper retained. P2 S is NA: no required interface.
- P1 E: original 26-direction reference-visible C retention. Main lower/upper use definite reference-visible points; global lower/upper also account for uncertain reference eligibility. Both intervals and status retained independently. P2 E is NA. These are not manifold, watertight or chair-semantic certificates.
- Reference completion: same GT512 queries for normalized MSE, p95 and <=1% coverage; original 512 bidirectional RMS in percent L; frozen GT2048 for two directed normalized MSEs and D_sym=(direction1+direction2)/2, strict PR/F-score thresholds. Predicted Q_F2048 remaps the original fixed candidate-specific seed and area sampler to each predicted surface. It does not reuse old predicted coordinate samples. C-to-itself distances excluded.
- Triangle quality: original free small-angle, p05 angle, near-degenerate and zero-area rates; predicted/GT angle references when GT exists.
- Duplicate excess: free faces are keyed by their three exact numeric coordinates, ignoring corner order/winding; this diagnostic is separate from near-duplicate I and does not remove any faces.
- Connectivity: RAW soup topology separate from diagnostic exact FP32-coordinate indexing. No tolerance weld or geometry modification. Shared C/free edges, boundaries, nonmanifold edges, same-direction two-face edges and face-connected components are descriptive. Nonmanifold vertices/full watertight certification are NOT_IMPLEMENTED; closed-two-faces-per-edge is not certification.
- Diversity: all six fixed-candidate symmetric mean squared free point-to-finite-triangle distances/L^2 using saved Q_F2048. Ordinary free pair averaging, not original TMD. Large variation may be defects, not inherently better.

Registry records version, RAW stage, domain, reference dependence, scale, thresholds, value/interval, applicability and denominator. Human usability, FID/KID, whole-mesh Ri and manifold certification are explicitly NOT_RUN/NOT_IMPLEMENTED. No source-MSE veto or scalar promotion gate is introduced.

## Aggregation and interpretation

P1 OLD, NEW and combined ALL remain separate. Noise averages within parent, then parents receive equal weight. P2 conditions stay separate; seat2/seat32 share geometric evidence. B-A measures added readout plus capacity; C-B compares equal-capacity pathways but also changes depth/state dependence and gradients. Retain parent/noise directions and tails, not just source reconstruction.

Execution validity, metric applicability and unknown detectors are separate. Errors preserve traceback/request identity. No denominator silently shrinks.

## Rendering and review

render(manifest,out) requires seven P1 labels x four noises, or four P2 labels x four noises, per condition before final rendering. An explicit allow_partial snapshot is marked PARTIAL. One camera box per condition uses all fixed outputs (plus P1 reference), includes flying geometry and is shared across models/noises. Four views plus wire use blue C/orange free with no alignment, normalization, welding, smoothing, pruning or repair.

INDEX.html lists pages without reference first, then P1 reference pages. No GT column exists for P2. Labels identify models: blind means hidden source target, not anonymous method. Rendering records AI_VISUAL_REVIEW=PENDING and human_usability=NOT_RUN. Actual review is a separate artifact.

## Migration checks

Standalone evaluation imports without torch. Compact CPU tests cover finite distances/sampling, contacts vs duplicate overlap, P2 I without GT, C and 51-state contracts, registry NA/bounds and a rendered four-view+wire page with incomplete-gallery refusal. One real START OLD RAW was successfully re-scored after import fixes: 129 common numerical scalars exactly match historical cache. Receipt: external artifacts/tests/evaluation_migration/regression.json. This is not permission to replace historical scores.
