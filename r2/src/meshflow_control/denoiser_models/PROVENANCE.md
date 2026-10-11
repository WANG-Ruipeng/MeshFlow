# Official Chair denoiser components

This is an inference-only packaging of selected MeshFlow decoder definitions,
from commit `db86bc32423c316c00bcc2f645a26243e8010f20` in qiisun/MeshFlow.
It is a local compatibility layer, not a new official MeshFlow release.
`source_manifest.json` records original and packaged file hashes. No weights,
training assets, model outputs or external checkout are included.

The copied MeshFlow definitions retain the repository MIT notice in
`LICENSE_MeshFlow`. That notice was added upstream in commit
`8cc74da013d3d2feec1853cccbb1f9ac37814b95`; the historical decoder commit itself
has no root LICENSE file. Repository licensing is not asserted to override
notices on unrelated third-party files.

The historical RMSNorm source carries a Llama 2 notice and its utility file
contains unrelated IO code/notices. Neither file nor its documentation is
vendored. `rmsnorm.py` and `utils.py` are small local implementations of the
standard mathematical operations, preserving dtype, epsilon, parameter names
and the audited order of arithmetic. The unused Apache-licensed SwiGLU,
Fairscale, FlashAttention, einops, megfile and training/sample code are omitted.
Only the official SiLU decoder path is maintained.

Selected official definitions are in equivae/equidit/pe/pos_embed. Imports are
package-relative; no sys.path additions or models.* aliases exist. AutoencoderKL
exposes initialization and decode only, not the stochastic encoder/forward.
Its state layout is retained, including the unused parameter-free cross-attention
module and the extra face-count embedding row. `SelfAttention` keeps its official
projection/arithmetic layout. Its local backend supports only unmasked,
noncausal, dropout-free native SDPA. No historical masked-SDPA behavior is used.

The public loader/predictor is meshflow_control.denoise_model. It uses the exact
released Chair EMA checkpoint identity; network autocast is BF16 with FP32
parameters, residual addition and coordinate scaling. The predictor returns
all unconstrained proposal faces. C restoration, source winding and GEO are
separate explicit caller operations. This package makes no preservation or
attachment claim about the unconstrained proposal.
