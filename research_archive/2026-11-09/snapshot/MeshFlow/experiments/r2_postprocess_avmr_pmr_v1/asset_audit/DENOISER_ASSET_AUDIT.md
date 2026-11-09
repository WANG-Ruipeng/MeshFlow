# Official Chair denoiser asset audit

The official Chair denoiser was found on the authors' Hugging Face dataset and recovered with its matching historical source. The earlier local absence and README omission do not establish public unavailability. No weights were installed globally, no training was run, and this audit invoked no model forward.

- Weight: `denoiser/002_500M_chair/0030000.pt` at HF revision `98d9f4a16bf5e8a187c02f12c4812000e8181614`.
- Expected size: 7,289,979,546 bytes; LFS SHA256: `6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36`.
- Matching official source: `db86bc32423c316c00bcc2f645a26243e8010f20` (2026-01-10), preserved under `../source_snapshot/official_denoiser_db86bc3/`.
- Current public `denoise` branch HEAD `20257d3` is an older/different VAE implementation; it was inspected but rejected for execution. Public `master` removed these denoiser files in later cleanup; history supplied the matching source.
- GitHub REST API initially returned unauthenticated rate limit; `git ls-remote` and a separate filtered upstream history supplied verifiable official branch and commit identities. Local repo/remotes were not changed.

## Scale and forward contract

The matching historical training dataset produces `coords * 0.95/0.5`; `vae=True` bypasses dataset-standard-deviation normalization. R2's stored coordinates are `2*unitbox/0.3762`, so use one fixed conversion:

`x_denoiser = 0.35739 * x_R2`; inverse conversion divides by 0.35739. There is no per-output bbox fitting, clipping or quantization.

The official checkpoint config describes an identity-encoder residual denoiser with hidden dimension 1024, 24 layers, 16 heads, face bin 100, regression decoder, and fixed training noise standard deviation 0.02. Its 75-wide coordinate encoding is 12-frequency NeRF encoding with the original coordinate included.

For an already generated RAW, the official postprocessor uses EMA when present, strips `module.` prefixes, and applies coordinate encoding followed by decoder plus the input residual. It does not need another encode/noise sample:

`recon = model.decode(model.pe_nerf(x.reshape(1, -1, 3)), tensor([N]), mask) + x`

Decoder output is `[1,N,9]`. C is not protected by the official network; restoring the original C bytes after this single call and protecting C during geometry consolidation is our conditional-task adapter, not an original MeshFlow capability.

## Runtime caveats

The snapshot contains optional imports for unused distributed classes (`fairscale`) and legacy flash-attention. Do not install these merely to satisfy unused code. A minimal adapter may load the exact required RMSNorm source independently and route attention to deterministic SDPA math.

The historical SDPA masked fallback inverted boolean masks. This evaluation processes each mesh separately without padding, so `mask=None` is an exact all-valid representation and avoids that branch. If an adapter introduces padding, it must correct and independently verify mask semantics.

The checkpoint and released implementation retain face-conditioned AdaLN parameters, despite the paper's description of a denoiser without AdaLN. The evaluation is a C-preserving adaptation of the official released Chair denoiser, not certification of exact paper-number reproduction.

Source hashes, coordinate evidence and download registration are in adjacent JSON files. The download completed in one attempt (743 seconds), and all 7,289,979,546 bytes passed the published SHA256 check. `weight_download_receipt.json` records the verified final asset. No `.partial` remains. Strict checkpoint key/shape loading is owned by the root execution runner, not this source audit.

Sources: [official HF asset](https://huggingface.co/datasets/qsun2001/meshflow/blob/98d9f4a16bf5e8a187c02f12c4812000e8181614/denoiser/002_500M_chair/0030000.pt), [official historical postprocessor](https://github.com/qiisun/MeshFlow/blob/db86bc32423c316c00bcc2f645a26243e8010f20/post_mesh.py), [matching dataset scale](https://github.com/qiisun/MeshFlow/blob/db86bc32423c316c00bcc2f645a26243e8010f20/datasets/mesh_dataset.py#L325), [paper](https://arxiv.org/html/2606.23489v2#S4.SS4).
