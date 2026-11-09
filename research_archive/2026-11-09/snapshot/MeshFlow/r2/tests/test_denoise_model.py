"""Bounded CPU/operator checks; no checkpoint load or model forward."""
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from meshflow_control import denoise_model as D
from meshflow_control.denoiser_models.attention import attention
from meshflow_control.denoiser_models.equivae import AutoencoderKL
from meshflow_control.denoiser_models.pe import NeRFEncoding
from meshflow_control.denoiser_models.rmsnorm import RMSNorm
from meshflow_control.denoiser_models import utils


def test_official_meta_architecture_without_weights_or_forward():
    # Meta tensors record sizes but allocate no 455M-parameter model storage.
    with torch.device("meta"):
        model = AutoencoderKL(**D.INIT_KWARGS)
    assert sum(p.numel() for p in model.parameters()) == 455592963
    assert len(model.state_dict()) == 296
    assert len(model.decoder.layers) == 24
    assert model.decoder.y_embedder.embedding_table.weight.shape == (9, 1024)
    assert model.decoder.layers[0].mlp.net[1].__class__ is torch.nn.SiLU
    assert model.encoder is None and model.quant_linear is None
    assert model.fixed_std == 0.02
    assert not hasattr(model, "encode")
    assert all(p.is_meta for p in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_independent_rms_formula(dtype):
    x = (torch.arange(60).reshape(3, 4, 5).float() / 17 - 0.9).to(dtype)
    norm = RMSNorm(5, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.6, 0.8, 1.0, 1.2, 1.4]))
    expected = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).type_as(x) * norm.weight
    assert torch.equal(norm(x), expected)


def test_independent_sincos_standard_coordinates():
    grid = np.arange(4, dtype=np.float32)
    frequency = 1.0 / 10000 ** (np.arange(4, dtype=np.float64) / 4.0)
    phase = grid[:, None] * frequency[None, :]
    expected = np.concatenate([np.sin(phase), np.cos(phase)], axis=1)
    np.testing.assert_array_equal(utils.get_1d_sincos_pos_embed_from_grid(8, grid), expected)
    embedded = utils.get_2d_sincos_pos_embed(16, 2, cls_token=True, extra_tokens=1)
    assert embedded.shape == (5, 16)
    np.testing.assert_array_equal(embedded[0], np.zeros(16))


def test_sdpa_unmasked_arithmetic_and_rejection():
    q = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4) / 20
    k = torch.arange(32, dtype=torch.float32).reshape(1, 4, 2, 4) / 30
    v = torch.arange(32, dtype=torch.float32).reshape(1, 4, 2, 4) / 10
    expected = (torch.softmax((q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) / 2, dim=-1) @ v.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(attention(q, k, v), expected, atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        attention(q, k, v, mask_q=torch.ones((1, 3), dtype=torch.bool))
    with pytest.raises(ValueError):
        attention(q, k, v, backend="flash-attn")


def test_nerf_has_no_noise_or_state_parameters():
    pe = NeRFEncoding(num_freqs=12)
    x = torch.tensor([[0.125, -0.375, 0.0625]], dtype=torch.float32)
    before = torch.get_rng_state().clone()
    actual = pe(x)
    expected = [x]
    for frequency in 2.0 ** torch.linspace(0.0, 11.0, 12):
        expected.extend([torch.sin(x * frequency * np.pi), torch.cos(x * frequency * np.pi)])
    assert actual.shape == (1, 75)
    assert torch.equal(actual, torch.cat(expected, dim=-1))
    assert torch.equal(before, torch.get_rng_state())
    assert len(pe.state_dict()) == 0


def test_residual_integrated_once_in_fp32():
    x = torch.tensor([1.0, 0.06251, -0.4], dtype=torch.float32)
    delta = torch.tensor([0.0039, -0.00023, 0.04], dtype=torch.bfloat16)
    actual = D._integrate_residual(x, delta)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, (delta.float() + x.float()) / D.SCALE)
    assert not torch.equal(actual, ((delta + x.to(torch.bfloat16)).float()) / D.SCALE)


def test_source_manifest_and_isolated_relative_imports():
    identity = D._source_identity()
    assert identity["official_commit"] == D.OFFICIAL_COMMIT
    folder = Path(D.__file__).with_name("denoiser_models")
    for path in folder.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "models" and not (node.module or "").startswith("models.")
            elif isinstance(node, ast.Import):
                assert all(alias.name != "models" and not alias.name.startswith("models.") for alias in node.names)
    assert not any(key == "models" or key.startswith("models.") for key in sys.modules)


def test_bad_checkpoint_rejected_without_torch_load(tmp_path, monkeypatch):
    path = tmp_path / "invalid.pt"
    path.write_bytes(b"not an official checkpoint")
    def forbidden(*args, **kwargs):
        raise AssertionError("torch.load must not be called")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="size or SHA256"):
        D.load_denoiser(path, device="cpu")
    with pytest.raises(ValueError, match="only supports"):
        D.load_denoiser(path, device="cpu", expected_sha256="0" * 64)


def test_raw_shape_precision_and_finiteness_contract():
    raw = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    saved = raw.tobytes()
    assert D._validate_raw(raw) == 2
    assert raw.tobytes() == saved
    with pytest.raises(TypeError):
        D._validate_raw(raw.astype(np.float64))
    with pytest.raises(ValueError):
        D._validate_raw(raw.reshape(2, 9))
    with pytest.raises(ValueError):
        D._validate_raw(np.full((1, 3, 3), np.nan, np.float32))


def test_runtime_is_strict_and_detects_drift():
    state = D.configure_runtime()
    assert state["math_sdp"] and not state["flash_sdp"] and not state["warn_only"]
    torch.backends.cuda.enable_flash_sdp(True)
    try:
        with pytest.raises(RuntimeError, match="runtime changed"):
            D._assert_runtime()
    finally:
        D.configure_runtime()
