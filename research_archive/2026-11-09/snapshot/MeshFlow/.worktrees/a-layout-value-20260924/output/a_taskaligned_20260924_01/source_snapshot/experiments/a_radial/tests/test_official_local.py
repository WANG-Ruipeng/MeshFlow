"""Integration tests require a REAL pinned MeshFlow worktree; not stubbed in cloud."""
import os
from pathlib import Path
import pytest

def test_pinned_repo_and_loader():
    repo=os.environ.get('MESHFLOW_TEST_REPO');data=os.environ.get('A_TEST_DATA')
    if not repo or not data:pytest.skip('Local integration pending: MESHFLOW_TEST_REPO/A_TEST_DATA not provided')
    from experiments.a_radial.official_contract import audit_loader
    ds,report=audit_loader(repo,data);assert report['pass']

def test_gpu_smoke_requires_local():
    if not os.environ.get('A_RUN_CUDA_TESTS'):pytest.skip('CUDA integration not executed in cloud')
    import torch
    assert torch.cuda.is_available()
    from experiments.a_radial.models import ShapeAE
    m=ShapeAE().cuda();x=torch.randn(4,2050,device='cuda');y,z=m(x);y.square().mean().backward();torch.cuda.synchronize()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
