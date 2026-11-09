"""Synthetic CPU contracts; never instantiate/forward the real generator."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch import nn

from meshflow_control.training.context_alignment import (
    AlignmentHead, AttentionCapture, select_messages, alignment_loss)
from meshflow_control.training.context_contract import HEAD_PARAMETERS, HEAD_SEED, validate_config
from meshflow_control.training import context_checkpoint as cp
from meshflow_control.training.losses import masked_fm_loss


def test_identical_nonzero_initialization_is_rng_isolated():
    before = torch.get_rng_state().clone()
    a, b = AlignmentHead(), AlignmentHead()
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for p in a.parameters()) == HEAD_PARAMETERS == 3934720
    assert all(torch.equal(v, b.state_dict()[k]) for k,v in a.state_dict().items())
    assert torch.count_nonzero(a.W_O.weight) == a.W_O.weight.numel()
    assert torch.count_nonzero(a.q) == 768
    assert a.source_norm.elementwise_affine is False
    assert a.W_Q.bias is None and a.linear_1.bias is not None
    assert HEAD_SEED == 17006844


def test_token_permutation_and_padding_values():
    head = AlignmentHead()
    a = torch.randn(1, 7, 768)
    known = torch.tensor([[True, True, False, False, False, False, False]])
    valid = torch.tensor([[True, True, True, True, True, False, False]])
    free, mask = select_messages(a, known, valid, "free")
    assert torch.equal(mask, torch.tensor([[False,False,True,True,True,False,False]]))
    expected = head(free[0])
    actual = head(free[0][torch.tensor([2,0,1])])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    modified = a.clone()
    modified[~valid] = float("nan")
    tokens, _ = select_messages(modified, known, valid, "free")
    assert torch.equal(head(tokens[0]), expected)
    known_tokens, known_selection = select_messages(a, known, valid, "known")
    assert torch.equal(known_tokens[0], a[0,:2])
    assert int(known_selection.sum()) == 2


def test_fp32_head_retains_bfloat16_feature_gradient_and_detaches_target():
    head = AlignmentHead()
    a = torch.randn(1, 7, 768, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn(768, requires_grad=True)
    known = torch.tensor([[True, True, False, False, False, False, False]])
    valid = torch.tensor([[True, True, True, True, True, False, False]])
    tokens, selected = select_messages(a, known, valid, "known")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = head(tokens[0])
        loss, statistics = alignment_loss(prediction, target)
    assert prediction.dtype == torch.float32 and loss.dtype == torch.float32
    loss.backward()
    assert target.grad is None
    assert a.grad is not None and float(a.grad[selected].float().norm()) > 0
    assert torch.count_nonzero(a.grad[~selected]) == 0
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
    assert all(float(p.grad.norm()) > 0 for p in head.parameters())
    assert statistics["prediction_norm"] > 0 and statistics["teacher_norm"] > 0


def test_combined_loss_uses_whole_free_scalar_and_four_sample_denominators():
    values = [torch.full((1, 4, 9), float(i + 1), requires_grad=True) for i in range(8)]
    valid = torch.ones((1, 4), dtype=torch.bool)
    known = torch.tensor([[True, False, False, False]])
    denominator = 8 * 3 * 9
    raw_aux = [torch.tensor(float(i+1), requires_grad=True) for i in range(4)]
    total = sum(masked_fm_loss(v, torch.zeros_like(v), valid, known, denominator=denominator) for v in values)
    total = total + sum(.1 / 4 * v for v in raw_aux)
    total.backward()
    assert abs(float(total.detach()) - (sum((i+1)**2 for i in range(8))/8 + .25)) < 1e-5
    for v in values:
        assert torch.count_nonzero(v.grad[:,0]) == 0
        torch.testing.assert_close(v.grad[:,1:], 2*v.detach()[:,1:]/denominator)
    for v in raw_aux:
        assert float(v.grad) == pytest.approx(.025)


def test_read_only_capture_preserves_output_and_releases_graph():
    attention = nn.Linear(768, 768)
    model = SimpleNamespace(readout_mode="none", backbone=SimpleNamespace(
        layers=[None, None, None, SimpleNamespace(attn=attention)]))
    x = torch.randn(1, 5, 768, requires_grad=True)
    baseline = attention(x)
    with AttentionCapture(model) as captured:
        observed = attention(x)
        assert torch.equal(baseline, observed)
        assert captured.get() is observed
        assert captured.get().requires_grad
        captured.get().square().sum().backward()
    assert captured.tensor is None and captured.handle is None
    assert not attention._forward_hooks
    assert float(x.grad.norm()) > 0


class TinyGenerator(nn.Module):
    def __init__(self, trainable=True):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2,3))
        self.readout_mode = "none"
        self.set_trainable(trainable).eval()
    def set_trainable(self, value):
        self.experiment_trainable = value
        self.requires_grad_(value)
        return self


def test_checkpoint_resume_and_generator_only_export_strict_roundtrip(tmp_path):
    config = validate_config({"arm": "B"})
    model, head = TinyGenerator(), AlignmentHead()
    optimizer = torch.optim.AdamW([
        dict(params=list(model.parameters()), lr=1e-5),
        dict(params=list(head.parameters()), lr=1e-4)], betas=(.9,.95), weight_decay=0.)
    for i, group in enumerate(optimizer.param_groups):
        for p in group["params"]:
            optimizer.state[p] = dict(step=torch.tensor(5000. if i == 0 else 0.),
                exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
    with patch.object(cp, "GENERATOR_TENSORS", 1), patch.object(cp, "GENERATOR_PARAMETERS", 6), patch.object(cp, "_generator_factory", TinyGenerator):
        path = tmp_path/"training.pt"
        receipt = cp.save(path, model, head, optimizer, step=0, stream_state={"batch_index":0},
            config=config, data_manifest_hash="data", teacher_manifest_hash="teacher")
        restored, aligner, opt, payload = cp.load(path, trainable=True, expected_sha256=receipt["sha256"])
        assert torch.equal(restored.weight, model.weight)
        assert all(torch.equal(v, aligner.state_dict()[n]) for n,v in head.state_dict().items())
        assert cp.validate_optimizer(restored, aligner, opt, 0)
        assert payload["source_start_state_sha256"] != payload["generator_state_sha256"]
        invalid_arm_payload = dict(payload, arm="C")
        wrong_arm = tmp_path/"wrong_arm_training.pt"
        torch.save(invalid_arm_payload, wrong_arm)
        with pytest.raises(ValueError, match="arm"):
            cp.export_generator(wrong_arm, tmp_path/"wrong_arm_export.pt")
        assert not (tmp_path/"wrong_arm_export.pt").exists()
        exported = tmp_path/"inference.pt"
        cp.export_generator(path, exported)
        exported_payload = torch.load(exported, map_location="cpu", weights_only=True)
        assert "aligner_state" not in exported_payload and "optimizer" not in exported_payload
        loaded, audit = cp.load_generator(exported)
        assert torch.equal(loaded.weight, model.weight)
        assert all(not p.requires_grad for p in loaded.parameters())
        assert audit["schema"] == cp.INFERENCE_SCHEMA
        with pytest.raises(ValueError):
            cp.load_generator(path)
        with pytest.raises(ValueError):
            cp.load(exported)
        with pytest.raises(ValueError):
            cp.load_generator(exported, expected_sha256="wrong")
        exported_payload["generator_state"]["weight"][0,0] = 99
        tampered = tmp_path/"tampered.pt"
        torch.save(exported_payload,tampered)
        with pytest.raises(ValueError, match="hash"):
            cp.load_generator(tampered)
        with pytest.raises(FileExistsError):
            cp.export_generator(path, exported)


def test_separate_clips_do_not_shrink_generator_gradient():
    g, h = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2))
    g.grad = torch.tensor([3.,4.])
    h.grad = torch.tensor([300.,400.])
    gn = torch.nn.utils.clip_grad_norm_([g],1.)
    hn = torch.nn.utils.clip_grad_norm_([h],1.)
    assert float(gn)==5. and float(hn)==500.
    torch.testing.assert_close(g.grad,torch.tensor([.6,.8]),atol=1e-6,rtol=1e-6)


def test_inference_checkpoint_import_has_no_aligner_teacher_or_capture_dependency():
    code = ("import sys; from meshflow_control.training.context_checkpoint import load_generator; "
            "assert 'meshflow_control.training.context_alignment' not in sys.modules; "
            "assert 'meshflow_control.teachers.siglip_targets' not in sys.modules; "
            "import torch; assert not torch.cuda.is_initialized()")
    p = subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True,timeout=60)
    assert p.returncode == 0, p.stdout+p.stderr


def test_configuration_rejects_unregistered_scientific_changes():
    config=validate_config({"arm":"C"})
    assert config["alignment_domain"]=="known" and config["readout_mode"]=="none"
    bad=copy.deepcopy(config)
    bad["head"]["layer_1based"]=3
    with pytest.raises(ValueError):
        validate_config(bad)
    bad=copy.deepcopy(config)
    bad["training"]["alignment_weight"]=.2
    with pytest.raises(ValueError):
        validate_config(bad)
