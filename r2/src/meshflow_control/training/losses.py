"""Original free-only FM loss; denominator covers all eight microbatches."""
import torch

def masked_fm_loss(velocity, target, valid_mask, known_mask, *, denominator=None):
    if velocity.shape != target.shape or velocity.ndim != 3 or velocity.shape[-1] != 9:
        raise ValueError("Velocity and target must share [B,N,9]")
    if valid_mask.shape != velocity.shape[:2] or known_mask.shape != valid_mask.shape:
        raise ValueError("Mask shapes differ")
    if valid_mask.dtype != torch.bool or known_mask.dtype != torch.bool:
        raise ValueError("Boolean masks required")
    mask = valid_mask & ~known_mask
    count = int(mask.sum()) * 9
    divisor = count if denominator is None else int(denominator)
    if count <= 0 or divisor < count:
        raise ValueError("Invalid effective-batch free-coordinate denominator")
    return (velocity.float() - target.float()).square()[mask].sum() / divisor
