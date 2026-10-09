"""Necessary MeshFlow backbone components, migrated without changing trained arithmetic.
Source and local modifications are recorded in docs/model_source_map.json.
"""
import torch


import torch.nn as nn


import torch.nn.functional as F


from .utils import RMSNorm


def _attention_sdpa(q, k, v, mask_q=None, mask_kv=None, dropout=0.0, causal=False):
    batch_size, n_target, num_heads, head_dim = q.shape
    _, m_source, _, _ = k.shape

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)

    # torch SDPA treats a boolean attn_mask as True == "attend". Build a mask that
    # is True where attention is ALLOWED: exclude padded KEYS so that no query
    # attends to padding. (The previous version passed True at *padded* positions,
    # which made SDPA attend to padding and mask out the real tokens -> corrupted
    # output whenever FlashAttention was unavailable and this fallback ran.)
    #
    # Keys only, not queries: rows for padded queries are discarded by the caller,
    # and masking their keys too would create all-False rows -> NaN after softmax.
    attn_mask = None
    key_valid = mask_kv if mask_kv is not None else mask_q
    if key_valid is not None:
        attn_mask = key_valid[:, None, None, :].expand(batch_size, 1, n_target, m_source)

    if causal:
        if not (n_target == m_source or n_target == 1):
            raise ValueError(
                f"causal=True requires n_target == m_source or n_target == 1, got n_target={n_target}, m_source={m_source}"
            )
        causal_mask = torch.tril(torch.ones(n_target, m_source, dtype=torch.bool, device=q.device))[None, None]
        attn_mask = causal_mask if attn_mask is None else (attn_mask & causal_mask)

    output = F.scaled_dot_product_attention(
        query=q_sdpa,
        key=k_sdpa,
        value=v_sdpa,
        attn_mask=attn_mask,
        dropout_p=dropout,
        is_causal=False,
    )
    return output.transpose(1, 2)


class PrecisionSafeLayerNorm(nn.Module):
    """Run RMSNorm in fp32 for stability, then cast back to input dtype."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x):
        if x.dtype in (torch.float16, torch.bfloat16):
            return self.norm(x.float()).to(dtype=x.dtype)
        return self.norm(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        input_dim=None,
        output_dim=None,
        dropout=0.0,
        causal=False,
        mixed_precision='bf16',
        qk_norm=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, 'hidden_dim must be divisible by num_heads'
        self.head_dim = hidden_dim // num_heads
        self.causal = causal
        self.dropout = dropout
        self.mixed_precision = mixed_precision

        self.qkv_proj = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self.q_norm = PrecisionSafeLayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = PrecisionSafeLayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x, mask=None):
        batch_size, n_tokens, _ = x.shape
        qkv = self.qkv_proj(x).reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        q, k, v = qkv.chunk(3, dim=0)
        q, k = self.q_norm(q[0]), self.k_norm(k[0])

        backend = 'pytorch-sdpa' if self.mixed_precision == 'fp32' else 'flash-attn'
        if q.dtype not in (torch.float16, torch.bfloat16):
            backend = 'pytorch-sdpa'
        x = attention(
            q,
            k,
            v[0],
            mask_q=mask,
            mask_kv=mask,
            dropout=self.dropout,
            causal=self.causal,
            backend=backend,
        )
        return self.out_proj(x.reshape(batch_size, n_tokens, -1))


FLASH_ATTN_AVAILABLE = False

def attention(q, k, v, mask_q=None, mask_kv=None, dropout=0.0, causal=False, backend="pytorch-sdpa"):
    """The audited SDPA fallback; True mask entries mean allowed keys."""
    if backend not in ("pytorch-sdpa", "flash-attn"):
        raise ValueError(f"Unsupported backend: {backend}")
    return _attention_sdpa(q, k, v, mask_q=mask_q, mask_kv=mask_kv, dropout=dropout, causal=causal)
