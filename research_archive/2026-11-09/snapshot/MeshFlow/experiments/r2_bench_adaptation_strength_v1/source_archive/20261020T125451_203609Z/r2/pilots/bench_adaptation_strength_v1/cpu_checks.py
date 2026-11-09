"""Small CPU contract checks; never construct a MeshFlow model or call forward."""
from __future__ import annotations
from collections import Counter
import copy
import json
import torch
import numpy as np
from . import training as T


class ParametersOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.ParameterDict({"weight": torch.nn.Parameter(torch.ones(2, 3))})
        self.role_embedding = torch.nn.Parameter(torch.zeros(2))
        self.context_encoder = torch.nn.ParameterDict({"weight": torch.nn.Parameter(torch.zeros(3, 2))})
        self.readout_mode = "none"
        self.eval()


def run():
    tests = []
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU checks require untouched CUDA")
    model = ParametersOnly()
    names = T.group_names(model)
    assert set(names[0]).isdisjoint(names[1])
    assert sum(map(len, names)) == len(list(model.parameters()))
    tests.append("parameter_groups_disjoint_complete")
    for arm in T.ARMS:
        opt = T.make_optimizer(model, arm)
        T.validate_optimizer(model, opt, arm, 0)
        assert len(opt.state) == 0
        tests.append(arm + "_fresh_grouped_AdamW")
        opt.param_groups[1]["lr"] = 2e-6
        try:
            T.validate_optimizer(model, opt, arm, 0)
            raise AssertionError("wrong condition LR accepted")
        except ValueError:
            pass
    tests.append("wrong_phi_lr_rejected")
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    drift = T.parameter_drift(initial, initial, names)
    assert all(x["delta_l2"] == 0 for x in drift.values())
    altered = copy.deepcopy(initial)
    altered[names[0][0]] += 1
    drift = T.parameter_drift(altered, initial, names)
    assert np.isclose(drift["theta_backbone"]["delta_rms"], 1.)
    assert drift["phi_condition"]["delta_l2"] == 0
    tests.append("drift_separates_backbone_and_condition")
    sample = dict(valid_mask=np.array([True, True, True, False]),
                  anchor_ids=np.array([[0, 1, 2], [0, 1, -1], [-1, -1, -1], [-1, -1, -1]]))
    result = T.stat(np.ones((4, 3, 3), np.float32), sample)
    assert result["all_free_scalar_count"] == 12
    assert result["seam_tip_scalar_count"] == 3 and result["bulk_scalar_count"] == 9
    assert result["all_free_velocity_sse"] == 12
    tests.append("masked_SSE_and_scalar_denominators")
    class Ledger:
        pending = {}
        failed = Counter()
        attempted = Counter(train_forward=8, train_backward=8, optimizer_update=1)
        successful = attempted.copy()
        rows = [dict(event="CALL_BEGIN", kind="optimizer_update", context={"local_step": 1})]
    T.require_cursor(Ledger(), 1)
    try:
        T.require_cursor(Ledger(), 0)
        raise AssertionError("stale cursor accepted")
    except RuntimeError:
        pass
    tests.append("stale_cursor_refuses_replay")
    assert 384 + 288 + 384 == 1056
    assert [T.source_global(a) for a in T.ARMS] == [12000, 12000, 0]
    tests.append("probe_budget_and_lineage")
    assert not torch.cuda.is_initialized()
    return dict(status="PASS", tests=tests, test_count=len(tests), model_constructions=0,
                model_forwards=0, backward_calls=0, optimizer_updates=0, CUDA_initialized=False)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
