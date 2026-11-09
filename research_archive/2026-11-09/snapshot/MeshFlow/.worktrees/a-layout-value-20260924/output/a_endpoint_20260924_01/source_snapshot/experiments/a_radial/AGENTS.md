# Scoped instructions for local Codex

Read the top-level handoff CODEX_PROMPT_zh.md and docs before editing this subtree.

- Only SIGGRAPH2026 qiisun/MeshFlow, pinned commit bd7d8a138465c583694f9f86083ddd2979d10f87.
- Add modules here. Do not replace EquiDiT, official OT, upstream dataset scaling/mask rules, or root train.py/flow_matching.py. A latent prior is not a replacement for EquiDiT registered under its name.
- ShapeAE:2050→1024→64; decoder64→1024→1024→2050, SiLU, PCA64 skip; output support2050/F4096. No F input to mainfield. No architecture sweep.
- A: original frozen DomainVelocity, generated-field proxy256,32 guarded steps, final accepted direction requery. It is currently CPU; do not claim GPU-parallel timings or universal formal certification.
- 16 raw parents, no official held-out objects, exact midpoint shared-edge splits to512. No Loop, fill, outer hull, extra parts, or noisy duplicates.
- All internal outlets use the identical saved sample bundle, exact AE/FM hashes and nodes; retain failures and actual F. No rejection sampling or per-row checkpoint choice.
- New data/run directories only; never overwrite original assets/checkpoints. Save complete optimizer and Python/NumPy/CPU/CUDA RNG, sampler state and config hashes.
- Upstream public post is not the paper's full denoiser chain. A missing denoiser/backend is NOT evidence of a method's failure; mark unavailable. Do not label a trimesh-only hole filler as official full post.
- Geometry correctness, distribution quality, runtime, face counts, and closed-set vs held-out qualification remain separate. No A/B/C winner from incomparable rows.
- 16 centered observations have rank<=15;64 PCA dimensions must be padded, and exact closed-set training fit is not generalization evidence.
- Tests skipped because CUDA/repo unavailable are pending, not passes. Run actual upstream-loader512 gate and one true GPU training step before long jobs.
- Never perform git push or delete checkpoints without explicit user instruction. Deliver diffs, config/data/model/noise hashes, per-attempt JSON, failure cases, and reproduction commands.
