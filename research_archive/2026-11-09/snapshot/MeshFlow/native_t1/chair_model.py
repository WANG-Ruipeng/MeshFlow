"""Native T1 + Geo for the retained variable-face Chair checkpoint.

The computation preserves the trained Native arithmetic: known time is one,
free time is t, roles are added before Geo, and each original block uses the
separate valid and known masks. There is no relation-attention branch.
"""
from . import runtime as _runtime  # Establish CUBLAS policy before torch.
import torch

from .model import (
    NativeInpaintingModel, assert_native_runtime, native_block_forward,
    native_final_forward, _corners,
)
from .context_geometry import inject_context


class ChairModel(NativeInpaintingModel):
    """Native arithmetic for total N128..256, including an independent valid mask.

    Constructor and parameter registration are inherited unchanged. This keeps
    the retained checkpoint's role/backbone/Geo tensor names and order intact.
    """

    def forward(self, x, t, y, valid_mask=None, known_mask=None, *, observer=None):
        assert_native_runtime(self, coordinates=(x, t))
        if x.ndim != 3 or x.shape[-1] != 9:
            raise ValueError('Expected FP32 x[B,padded_N,9]')
        batch, faces, _ = x.shape
        if t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64:
            raise ValueError('Expected free t[B] and integer total N[B]')
        if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()):
            raise ValueError('Nonfinite coordinates/time')
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError('Free time outside [0,1]')
        for mask in (valid_mask, known_mask):
            if mask is None or mask.dtype != torch.bool or mask.shape != (batch, faces) or mask.device != x.device:
                raise ValueError('Colocated separate bool[B,padded_N] masks required')
        if bool((known_mask & ~valid_mask).any()):
            raise ValueError('Known faces must be valid')
        if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 256)).any()):
            raise ValueError('y must count real faces only, N128..256')
        if bool((known_mask.sum(1) == 0).any()) or bool(((valid_mask & ~known_mask).sum(1) == 0).any()):
            raise ValueError('Each sample requires known and free faces')
        _, hidden = self.backbone.x_embedder(x)
        hidden = hidden.view(batch, faces * 3, -1)
        role = self.role_embedding[known_mask.long()].to(hidden.dtype)
        hidden = hidden + _corners(role)
        if self.context_encoder is not None:
            delta = self.context_encoder(x, known_mask, valid_mask, observer=self.context_observer)
            hidden = inject_context(hidden, delta, known_mask, observer=self.context_observer)
        c_free = self.backbone.t_embedder(t)
        c_known = self.backbone.t_embedder(torch.ones_like(t))
        count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
        c_free, c_known = c_free + count, c_known + count
        face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
        if observer is not None:
            observer(dict(site='embedding', hidden=hidden, role=role,
                known_mask=known_mask, valid_mask=valid_mask, face_time=face_time,
                c_free=c_free, c_known=c_known))
        for index, block in enumerate(self.backbone.layers, 1):
            hidden = native_block_forward(block, hidden, c_free, c_known, known_mask,
                valid_mask, observer=observer, layer_index=index, face_time=face_time)
        output = native_final_forward(self.backbone.final_layer, hidden, c_free,
            c_known, known_mask, observer=observer, face_time=face_time)
        # Raw known velocity is unconstrained. The sampler owns exact C clamping.
        return output.view(batch, faces, 9)
