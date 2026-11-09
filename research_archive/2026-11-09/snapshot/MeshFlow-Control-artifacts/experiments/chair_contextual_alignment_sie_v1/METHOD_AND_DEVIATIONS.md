# Contextual alignment S/I/E pilot: method and deviations

Task: CHAIR_CONTEXTUAL_ALIGNMENT_SIE_PILOT_V1. The task request and protocol.json are authoritative. No default checkpoint promotion, postprocessing, READOUT branch, condition dropping, CFG, geometric loss, layer search or weight search is included.

## Literature and scope

We read section 4 and appendices A.1.1/A.2 of [Learning to read the contextual tokens in Diffusion Transformers](https://arxiv.org/html/2610.06844v1). It motivates early attention-output supervision, a single-query aligner and a frozen pooled SigLIP image target. Its null-condition weighting is omitted because this mesh task keeps C in every example. The multiview renderer, alpha=1-only supervision, layer adaptation and separate learning rates are this pilot's declared choices. A negative result cannot reject the full original CoAl method. Paper/project sources: [paper](https://arxiv.org/html/2610.06844v1), [authors' project](https://omer11a.github.io/learning_to_read/).

## Frozen comparison

A_FM uses ordinary FM continuation; B_FREE_ALIGN adds alignment from free face messages; C_KNOWN_ALIGN selects known face messages. All begin from the same strict START generator and AdamW step5000 with retained moments. Each has 1000 optimizer updates. Step500 is saved for recovery/statistics only; it is never generated or selected by DEV performance. The shared stream is independently registered before training, keeping all 32 TRAINING parents and 453 READY partitions.

The generator is unchanged Native/T1 + Geo(P=I), 194 tensors and 130401155 parameters. A read-only forward hook captures backbone.layers[3].attn output [B,N,768] after its internal output projection but before the native external gate, corner broadcast, residual and MLP. The hook returns None and is released each eligible microbatch. B/C differ only in selecting valid free versus valid known face messages.

Both aligners have 3934720 parameters: one 768-dimensional learned query, 24 attention heads of width32, bias-free Q/K/V/O, affine-free LayerNorm eps1e-5, and a biased 768→1024→768 SiLU MLP. The query is normal(std=.02), linears Xavier-uniform, MLP biases zero, with isolated seed17006844. This is normal nonzero initialization, never the retired READOUT zero-output rule. Head and cosine arithmetic are FP32 and retain the gradient through a.float().

## Supervision and optimizer

Effective batch8 / microbatch1 uses eight distinct parents; four exact alpha1 examples and four original continuous U[.9,1.1] width augmentations. All eight use original free-only HYBRID-FM: nested free OT first, then sqrt(.75) Z + .5 epsilon, no rematching, original logit-normal t and original velocity parameterization. FM divides by the total valid free scalar count of the full effective batch.

Only the four alpha1 examples use an auxiliary target. Each eligible microbatch adds .1/4 × (1−cosine(prediction, frozen teacher)); one combined backward is executed. The generator group uses AdamW lr1e-5, the aligner lr1e-4, betas(.9,.95), wd0. Each group is clipped separately to norm1. Old generator moments continue, new head moments begin at zero. No adaptive cap, scheduler, EMA or auxiliary warmup is added.

Generators use eval mode with autograd, BF16 backbone, FP32 parameters/geometry/integration, strict deterministic algorithms, math SDPA, TF32 off and CUBLAS_WORKSPACE_CONFIG=:4096:8. Training-only modules are excluded from strict versioned inference exports.

## Teacher and camera

The specified google/siglip-base-patch16-256 revision is b078df89e446d623010d890864d4207fe6399f61. Official safetensors SHA256: f0cee7c815135c44a515eff72ab3040499744920442bc25567cd04efc93f8f65. Only the official vision_model.* state is loaded, strict=True, into the official vision architecture; pooler_output has768 dimensions. It is frozen/eval and used only in a separate precomputation process. An isolated teacher environment leaves the existing training environment unchanged.

There are32 unique complete clean TRAINING geometries and128 images, not4000 repeated target encodings. Each actual alpha1 C+clean-free target is reconstructed through its original source-face/corner maps and checked against the teacher geometry hash. Four equal views at45/135/225/315°, elevation20°,256×256 use common training-only camera bounds with10% margin, fixed gray/background/light and two-sided finite triangles. No per-object fitting, topology repair, C/free colors, labels or wire enter teacher input. The labeled contact sheet is a separate review artifact.

Normalize each view's pooler vector, average all four equally, then normalize again. Cache identity includes full geometry, revision, processor and rendering configuration. Pairwise teacher cosine and the constant mean-teacher baseline measure target distinguishability; they do not measure control success.

## Evaluation and interpretation

P1: eight existing DEV A20 parents, original n0/n1 plus independently registered new6/7 Gaussians, four labels,128 outputs. P2: unchanged seat2/seat32/back2/seat_back4 controls and four original fixed noises,64 outputs. Verify and reuse32 START outputs; normally generate160 new Euler50 rollouts. CONFIRM is not read.

Original chair_boundary_attachment_control_metrics_v1r1 remains unchanged. S and E on P1 are explicitly REFERENCE_INTERFACE and REFERENCE_EXPOSURE; P2 has no source S/E/distance and retains NA. I is C-area union coverage of crossing/near-duplicate bands, not a free-face affected fraction. All bounds, applicability, denominators and unresolved geometry remain visible. Prediction surface queries reuse the prescribed random stream remapped onto each new predicted mesh.

Report OLD/NEW/ALL separately, average noise within each parent and then weight eight parents equally. Retain paired output/parent differences, median, improved parent count and leave-one-parent sensitivity. Conservative differences are [candidate_lower−reference_upper, candidate_upper−reference_lower]; detector bounds are not statistical confidence intervals. The fixed screen in protocol.json applies without tuning. Local severe I/E and single-output geometric costs remain explicit even if averages improve.

All192 outputs are shown with unchanged geometry, shared condition cameras, blue C/orange free, four views and wire. Source-hidden pages precede source-reference pages. Actual AI image review is recorded separately; human usability remains NOT_RUN. Style differences alone are not defects. Source reconstruction distances and diversity are diagnostics, not a unique-GT veto or independent quality guarantee.

## Engineering changes and actual outcome

Implementation/asset errors and recoveries are separately retained in audit or teacher/engineering. No scientific formula change is authorized within this version. The final EXPERIMENT_REPORT.md and actual ledgers supersede planned counts. A complete computational experiment, a research signal and model promotion are separate judgments.


## Final execution and reporting additions

The actual run completed all three 1000-update arms, 160 new Euler50 rollouts and32 verified START caches, yielding192 scored outputs and288 diversity pairs. GPU ownership was5133.034207485 seconds; formal generator training24000F/24000B, head8000F/shared-combined-B participation, teacher128 images/8F, preflight+export12F/10B, sparse observation12 autograd calls, sampling8000NFE. The ledger has zero model-call errors, zero unfinished ownership periods and zero training replays. These categories are not additive independent backward costs.

The actual CPU scorer used two workers with two threads each; the preparation resource record allowed up to four. This scheduling choice did not change queries, scoring density, training settings or samples. All192 outputs passed all seven exact C/N/Gaussian/finite contracts. P1 exposure retains96 PARTIAL and32 PASS statuses; P2 source-interface/exposure metrics remain64 NA rows.

The original quality implementation reports two directional2048-point MSEs and a separate fixed512-query p95. To satisfy the requested two-direction distance-tail diagnosis, analysis/distance_tails adds p95 from the existing saved2048 squared-distance vectors: numpy.quantile(sqrt(d2)/L0,0.95,method=linear). It uses no new points, geometry-distance evaluation or model calls; all256 existing directional MSEs were reproduced. It does not alter any original scalar, SIE definition or screen. The report reader initially expected cached hashes instead of the new wrapper files field; that CPU-only failure was corrected before any tail CSV output. Its prior version is retained under engineering/distance_tail_summary_reader_before.py.

Source-hidden review covers all48 pages/192 outputs and was frozen before any32 P1 source-reference pages were opened. Method labels are anonymous in the images, but one reviewer had earlier exposure to the mapping while preparing schemas. Accordingly this is source-hidden-first AI visual review, not a strict method-blinded or human usability study. The original anonymous observations and subsequent reference observations remain separate. A report-side merger was updated to accept both JSON-array and pipe-separated view_scope encodings; original review CSVs were unchanged. Human usability is NOT_RUN.

Complete outcomes, local tail costs and uncertainty are in EXPERIMENT_REPORT.md. Both B and C fail the preregistered developmental screen; this is distinct from successful implementation/execution. No model promotion or automatic follow-on experiment is performed.
