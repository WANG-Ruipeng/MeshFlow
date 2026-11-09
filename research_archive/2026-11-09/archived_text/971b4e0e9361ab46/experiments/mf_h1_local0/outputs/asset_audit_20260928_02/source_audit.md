# MF-H1 Phase 0 asset and source audit

Generated: 2026-09-28T04:08:12.728949+00:00

- Repository: `/mnt/e/MeshFlow`; HEAD: `8cc74da013d3d2feec1853cccbb1f9ac37814b95`.
- Reviewed reference: `8cc74da013d3d2feec1853cccbb1f9ac37814b95`.
- Official chair checkpoint: **PASS**.
- Real chair data: **PASS** at `/mnt/e/MeshFlow/experiments/mf_h1_local0/assets/shapenet-03001627`.
- GPU: `NVIDIA GeForce RTX 5080`; current free/total memory are in environment.json.
- BF16 tiny environment probe: **PASS**. This is not an official model forward.
- Existing bench checkpoint and old overfit assets were explicitly excluded; no random/synthetic substitute was loaded.
- Official strict load / forward / rollout / training: **NOT_RUN** in this audit; all corresponding counters are zero.

## Source semantics checked

The chair YAML specifies version 3, width 768, 12 layers, 12 heads, max length 800,
face bin 20, QK/RMSNorm, Fourier coordinates, no face-index positional encoding,
and velocity prediction. Checkpoint configuration/shape verification is blocked
when checkpoint bytes are absent. The declared 512 evaluation face slots fit
the source configuration, but this is not a checkpoint compatibility result.

`DiT.forward` reshapes corner hidden states to [B,3N,768]. Version 3 attention
pools corners per face, attends across valid faces, and broadcasts the face
message back to corners. The original y input is face count; mask=True means
a valid face. The source attention uses PyTorch SDPA.

`ObjaverseDataset` selects `split/train.npz` or `split/test.npz`, reads UID
records from `npz_list`, and matches `objaverse_occ_v5_ids/<uid>.npz`. It keeps
20 < faces_num < max_face_length. Existing other-category or synthetic data
cannot establish the requested author chair split identity.

The original data path performs bbox centering/scaling, permutation/optional
augmentation, division by 0.3762, nested corner/face OT noise assignment,
then doubles target tokens only. `FlowMatching.sample` passes `num_steps`
linspace nodes to torchdiffeq; 51 nodes are needed for 50 Euler intervals.
Its original forward_with_cfg calls the base model's own forward, so wrapper
sampling must be verified by actual call counters separately.

## Scope and limits

Source hashes, source diff/status, checkpoint candidate paths, rejected asset
hashes, installed package versions, host/device resources and probe evidence
are preserved in adjacent JSON files. No assets were changed, downloaded,
uploaded or replaced. No Phase 3 work is authorized or started.

## Verified cross-platform Git status

Windows Git reports only the new `experiments/` directory as untracked; tracked diff is empty. The unadjusted WSL diff retained in source_audit.json reflects CRLF/LF interpretation because WSL lacks Windows global core.autocrlf=true. A command-local setting confirms the clean tracked source without modifying .git or the original source files. See git_cross_platform_verification.json.

## Actual checkpoint and author splits

Official chair bytes and SHA256 match the independently obtained HF identity. Safe weights-only mmap CPU loading found model/ema/opt/config, with 182 EMA tensors totaling 130,249,731 elements (520,998,924 bytes). Model/data/transport/sample configuration matches the reference. Training-history metadata differs: checkpoint max_steps=2,000,000, batch=72, lr=2e-5, resume=true, keep_last_ckpts=3; these are explicitly recorded rather than silently substituted. Author train/test counts are 2,822/313, UID overlap is zero, their union equals all 3,135 extracted mesh files. The archive, splits and every extracted file have SHA256 records. This audit performs no official model forward, strict model load, rollout or optimizer update.
