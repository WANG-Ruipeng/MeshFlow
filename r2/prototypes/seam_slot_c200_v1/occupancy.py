"""C-only occupancy head for original C boundary edges.

The forward API accepts geometry features [E,D] and an edge-valid mask only:
no Y, source-parent identity, UID, or lookup table enters inference. The caller
must construct these features from C alone; this module cannot prove upstream
feature provenance. Features are detached so BCE cannot backpropagate into a
generator. The head has no shared generator parameters and needs fresh AdamW
state; generator optimizer state is independently restored from its checkpoint.
"""
from __future__ import annotations

from meshflow_control import runtime as _stable  # policy before torch import
import torch
from torch import nn
from torch.nn import functional as F


class OccupancyHead(nn.Module):
    """Small per-edge D -> 32 -> 1 MLP; fixed logit>=0 occupancy decision."""

    def __init__(self, features_dim: int, *, hidden_dim: int = 32, seed: int = 20261016):
        super().__init__()
        if features_dim < 1 or hidden_dim < 1:
            raise ValueError("Positive feature and hidden dimensions required")
        self.features_dim = int(features_dim)
        self.hidden_dim = int(hidden_dim)
        self.initialization_seed = int(seed)
        # Local CPU initialization neither consumes the caller's RNG nor
        # initializes CUDA; both matrices are nonzero (no zero-exit gradient trap).
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.initialization_seed)
            self.net = nn.Sequential(
                nn.Linear(features_dim, hidden_dim, device="cpu", dtype=torch.float32),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, device="cpu", dtype=torch.float32),
            )
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, edge_features, edge_mask):
        if edge_features.ndim != 2 or edge_features.shape[1] != self.features_dim:
            raise ValueError("Expected one C's edge_features[E,D]")
        if (edge_mask.shape != edge_features.shape[:1] or edge_mask.dtype != torch.bool
                or edge_mask.device != edge_features.device):
            raise ValueError("Colocated boolean edge_mask[E] required")
        if not bool(torch.isfinite(edge_features).all()):
            raise ValueError("Nonfinite C edge features")
        if edge_features.device != next(self.parameters()).device:
            raise ValueError("Head and edge features must be colocated")
        logits = self.net(edge_features.detach().float()).squeeze(-1)
        return torch.where(edge_mask, logits, torch.zeros_like(logits))

    @staticmethod
    def occupied(logits, edge_mask):
        """No data-tuned threshold; padded edges are always inactive."""
        if logits.shape != edge_mask.shape or edge_mask.dtype != torch.bool:
            raise ValueError("Matching logits and boolean edge mask required")
        return (logits >= 0) & edge_mask


def occupancy_bce_loss(logits, labels, edge_mask, *, class_weights):
    """Training-only binary labels; explicit [negative,positive] class weights.

    Weighted per-valid-edge mean preserves the registered BCE scale. Weights
    should be frozen from training counts (N/(2*N_class) if both classes exist),
    never silently recomputed per microbatch or inferred from evaluation labels.
    """
    if (logits.ndim != 1 or labels.shape != logits.shape or edge_mask.shape != logits.shape
            or edge_mask.dtype != torch.bool):
        raise ValueError("Expected logits, labels and boolean mask [E]")
    if any(v.device != logits.device for v in (labels, edge_mask)):
        raise ValueError("BCE tensors must be colocated")
    if not bool(edge_mask.any()):
        raise ValueError("No valid boundary edges")
    target = labels.detach().float()[edge_mask]
    if not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("Occupancy labels must be binary on valid edges")
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=logits.device).detach()
    if weights.shape != (2,) or not bool(torch.isfinite(weights).all() and (weights > 0).all()):
        raise ValueError("Explicit positive finite [negative,positive] class_weights required")
    per_edge = F.binary_cross_entropy_with_logits(logits.float()[edge_mask], target, reduction="none")
    return (per_edge * weights[target.long()]).mean()


def cpu_contract_checks():
    """Tiny head forward/backward only: no optimizer and no generator/D0 calls."""
    before_rng = torch.random.get_rng_state().clone()
    head = OccupancyHead(5)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    features = (torch.arange(20, dtype=torch.float32).reshape(4, 5) / 10).requires_grad_()
    mask = torch.tensor([True, True, True, False])
    labels = torch.tensor([1., 0., 1., 0.], requires_grad=True)
    logits = head(features, mask)
    logits.retain_grad()
    # N/(2*N_class): one negative and two positive valid edges.
    loss = occupancy_bce_loss(logits, labels, mask, class_weights=[1.5, .75])
    loss.backward()
    assert features.grad is None and labels.grad is None
    assert bool((logits.grad[~mask] == 0).all())
    norms = {name: float(parameter.grad.norm()) for name, parameter in head.named_parameters()}
    assert all(value > 0 for value in norms.values())
    assert not bool(head.occupied(logits.detach(), mask)[~mask].any())
    assert torch.equal(head.occupied(torch.tensor([-1., 0., 1., 9.]), mask),
                       torch.tensor([False, True, True, False]))
    changed_padding = features.detach().clone()
    changed_padding[-1] = -99
    assert torch.equal(head(changed_padding, mask)[mask], logits.detach()[mask])
    return dict(status="PASS", device="cpu", parameter_count=sum(p.numel() for p in head.parameters()),
                head_forward_calls=2, head_backward_calls=1, optimizer_updates=0,
                generator_forwards=0, D0_forwards=0,
                features_and_labels_gradient_detached=True, padded_logits_gradient_zero=True,
                parameter_gradient_norms=norms, initialization_preserves_caller_rng=True,
                threshold_logit=0.0, equality_at_threshold_is_occupied=True,
                optimizer_policy="fresh independent head state; restore generator AdamW separately")
