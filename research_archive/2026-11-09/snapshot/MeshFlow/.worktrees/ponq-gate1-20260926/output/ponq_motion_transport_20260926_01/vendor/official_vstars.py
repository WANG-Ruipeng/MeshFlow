# Unmodified function extracted from official PoNQ.py.
import torch


def vstars_from_quadrics(Q, P, eps=.05):
    """Compute optimal vertices positions

    Parameters
    ----------
    Q: torch.tensor 
        Nx4x4 of quadric matrices
    P: torch.tensor 
        Nx3 tensor of positions

    Returns
    -------
    vstars: torch.tensor 
        Nx3 optimal vertices 
    eigs: torch.tensor 
        Nx3 eigen values of quadric matrices
    """
    A = Q[:, :3, :3]
    b = -Q[:, 3, :3]
    u, eigs, vh = torch.linalg.svd(A)

    eigs2 = torch.zeros_like(eigs)
    mask_s = eigs / eigs[:, 0, None] > eps
    eigs2[mask_s] = 1 / eigs[mask_s]

    base_pos = P
    vstars = base_pos + (
        vh.transpose(1, 2)
        @ torch.diag_embed(eigs2)
        @ u.transpose(1, 2)
        @ (b[..., None] - A @ base_pos[..., None])
    ).squeeze(-1)
    return vstars, vh, eigs
