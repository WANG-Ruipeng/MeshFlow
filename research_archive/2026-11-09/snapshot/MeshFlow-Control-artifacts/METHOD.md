# Native Mesh Direct Context Readout V1
This independent research extension starts from the retained Chair HYBRID START, not the public unconditional model. MeshFlow source: WANG-Ruipeng/MeshFlow at ae91eaa6f882a7d311233fa32053b7cd5de7ebe0. It is not an official method of the original authors.

The deployment inputs are FP32 triangles C, explicit total face count N, and Gaussian noise (or a seed). The output is the disjoint concatenation C + F. C is hard-clamped at all 51 Euler states. The objective is a compatible chair completion, not recovery of a unique source chair. Source completions and interface references are evaluation-only.

## Preserved computation
The variable-length Native/T1 model uses known time 1 and free time t. Coordinate features receive role embeddings, then Geo residuals only on known corners. Geo retains its identity propagation matrix P=I. Each original block averages corners into a face feature for joint known/free attention and broadcasts the face message back to corners. Known hidden states change in the original blocks even though known coordinates remain fixed in the sampler. The actual START forward returns velocity; no additional x1-to-velocity conversion is applied.

A uses the original computation. B and C add a corner-level readout after original blocks 3, 6, 9 and 12. Each block has independent bias-free Q/K/V/O projections, dimension 768 to 192, four heads of dimension 48, non-affine token LayerNorm with epsilon 1e-5, no dropout, gate, MLP, positional encoding or geometric attention bias. Queries are valid free corners. Sources are valid known corners, preserving corner instances and order. B reads the current block's known features. C reads a copied/index-selected H0 immediately after coordinate + role + Geo, before all blocks. C memory is recomputed each forward, reused read-only within that forward, never detached and never cached across optimizer/Euler steps.

The extra residual is scattered only into valid free hidden corners. Empty source/query sets yield zero updates in the standalone module. The public chair API requires 0 < K < N and N in 128..256.

All 16 new weight tensors total 2,359,296 parameters. A has 130,401,155 parameters; B/C each have 132,760,451. Q/K/V use one isolated seeded RNG; B/C initialization is identical; all output projections start at zero. Old 194 tensors remain exactly START. The first backward can give zero Q/K/V gradients through zero W_O; the isolated two-update CPU test checks later reachability.

## Numerical and training contract
Parameters, AdamW moments, integration and new readout math are FP32; the original backbone uses BF16. Readout LN/projections/math-SDPA/output are explicitly FP32, with conversion only at the hidden residual addition. Deterministic algorithms are strict, TF32 is off, CUBLAS_WORKSPACE_CONFIG is :4096:8, and SDPA math is fixed. Evaluation mode preserves the original dropout policy without disabling training autograd.

The retained old AdamW step5000 and moments are restored by parameter identity. New parameters have a separate equal-hyperparameter group with step0 and zero moments. All parameters train for 1000 updates per arm, batch8/micro1, eight distinct parents, four alpha1 and four independently uniform alpha in [.9,1.1]. AdamW lr1e-5, betas(.9,.95), weight_decay0, global clip1; no scheduler/EMA/auxiliary objective. The only loss is free-coordinate squared FM error divided by the effective batch's total valid free scalar count.

The 32-parent/453-READY source task distribution is imported, including original repair_v2 and 20/40 percent context. A new frozen namespace defines the shared continuation stream. Original pre-OT scaling and free-only face/corner nested OT precede HYBRID mixing sqrt(.75) Z + .5 epsilon. The matched target maps back to original Gaussian slots. No rematching occurs after mixing. Inference starts from standard Gaussian rather than the training mixture.

## Fixed evidence and interpretation
500 is a secondary observation and 1000 the primary endpoint for all arms. P1 uses eight previously observed DEV parents with original n0/n1 and new preregistered n4/n5. It does not establish unseen-parent generalization. P2 uses four frozen sparse/synthetic conditions and four noises at N192 without a source GT. Two seat tessellations represent the same geometry, not independent shape evidence.

B-A tests the extra direct-readout path together with added capacity. C-B tests memory-source choice at equal capacity, but also changes source depth, state/time dependence and gradient paths. It cannot isolate a unique known-feature pollution mechanism. Loss, branch gradients, attention entropy and branch magnitude cannot prove generation quality. Exact C preservation cannot prove attachment, manifold topology or usefulness.

Metrics separate implementation, readout effect and open-completion evidence. Reference S/E and source distances apply only to P1 and are diagnostics of that particular completion. P2 source metrics are NA. Diversity includes every candidate and is interpreted together with quality. Human usability is NOT_RUN unless humans actually rate outputs. AI visual review requires actual image inspection. No model becomes the default automatically.

See metrics.md for definitions, source_map.md for provenance and the external EXPERIMENT_REPORT.md for actual results and counts.
