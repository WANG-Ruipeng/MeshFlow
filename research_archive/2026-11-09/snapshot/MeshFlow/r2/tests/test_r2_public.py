"""Public R2 boundary tests; no real model or GPU dispatch."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from meshflow_control import r2_cli, r2_runs
from meshflow_control.r2_checkpoints import identify
from meshflow_control.r2_sampling import SampleBudget
from meshflow_control.r2_spec import RECIPE


def test_recipe_bootstrap_is_local_and_does_not_import_torch():
    root = Path(__file__).resolve().parents[2]
    code = ("import runpy,sys;sys.argv=['r2','recipe'];"
            "\ntry: runpy.run_module('r2',run_name='__main__')"
            "\nexcept SystemExit as e: assert e.code == 0"
            "\nimport meshflow_control;from pathlib import Path;"
            "assert Path(meshflow_control.__file__).resolve().is_relative_to(Path('r2/src').resolve());"
            "assert 'torch' not in sys.modules")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["recipe"] == RECIPE


@pytest.mark.parametrize("schema", ["chair_condition_recipe_factorial_generator_v1",
                                   "chair_recipe_confirmation_generator_v1"])
def test_checkpoint_identity_rejects_other_recipes_and_late_hybrid(schema):
    value = dict(schema=schema, recipe=RECIPE, protocol=dict(
        condition_schedule="MIX", hybrid_schedule="all", hybrid_lambda=.25))
    assert identify(value)[1] == "generator"
    with pytest.raises(ValueError):
        identify(dict(value, recipe="R4_MIX_H_LATE"))
    value["protocol"]["hybrid_schedule"] = "late"
    with pytest.raises(ValueError):
        identify(value)


def test_public_train_has_no_recipe_override_and_returns_safe_stop_code(monkeypatch, capsys):
    calls = []
    def train(**kwargs):
        calls.append(kwargs)
        return {"status": "INTERRUPTED_SAFE"}
    monkeypatch.setattr(r2_runs, "train", train)
    args = ["train", "--official", "official", "--data-manifest", "data",
            "--stream-root", "stream", "--out", "out", "--registration", "reg",
            "--attempt-id", "a01"]
    assert r2_cli.main(args) == 75
    assert len(calls) == 1 and "recipe" not in calls[0]
    assert json.loads(capsys.readouterr().out)["status"] == "INTERRUPTED_SAFE"
    with pytest.raises(SystemExit):
        r2_cli.build_parser().parse_args(args + ["--recipe", "R4_MIX_H_LATE"])


def test_sample_ledger_counts_failures_and_cannot_reset_or_exceed_budget(tmp_path):
    path = tmp_path / "ledger.jsonl"
    budget = SampleBudget(path)
    budget.reserve("rollout_attempt")
    with pytest.raises(RuntimeError):
        budget.reserve("rollout_attempt")
    def fail():
        raise ValueError("count failed forward")
    with pytest.raises(ValueError):
        budget.call("sampling_forward", fail)
    for _ in range(49):
        budget.call("sampling_forward", lambda: None)
    with pytest.raises(RuntimeError):
        budget.reserve("sampling_forward")
    with pytest.raises(FileExistsError):
        SampleBudget(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sum(r["event"] == "RESERVE" for r in rows) == 51
    assert sum(r["event"] == "ERROR" for r in rows) == 1


def test_artifacts_cannot_enter_monorepo():
    root = r2_runs.source_root()
    for path in (root, root / "outputs", root.parent / "bad-r2-run"):
        with pytest.raises(ValueError, match="outside"):
            r2_runs.external_path(path)


def test_failed_asset_check_does_not_register_or_create_run(tmp_path, monkeypatch):
    def check(*args, **kwargs):
        raise ValueError("Official identity differs")
    monkeypatch.setattr(r2_runs, "check_assets", check)
    out, reg = tmp_path / "run", tmp_path / "reg.json"
    with pytest.raises(ValueError, match="Official"):
        r2_runs.prepare_run("data", "stream", "official", out, reg)
    assert not out.exists() and not reg.exists()

def test_frozen_source_bytes_survive_checkout():
    from hashlib import sha256
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/source_manifest.json").read_text())
    for row in manifest["source_files_copied_byte_exact"]:
        assert sha256((root / row["path"]).read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.parametrize("mode", ["fresh", "safe_resume", "completed", "failed", "changed_asset"])
def test_registered_train_gates_run_before_gpu(tmp_path, monkeypatch, mode):
    from contextlib import contextmanager
    from meshflow_control import runtime, r2_checkpoints
    from meshflow_control.data.io import digest
    from meshflow_control.data.recipe_factorial import NAMESPACE
    from meshflow_control.training import recipe_factorial as backend

    official, data = tmp_path / "official.pt", tmp_path / "train_manifest.json"
    stream = tmp_path / "stream"
    stream.mkdir()
    plan, paired = stream / "recipe_plan.json", stream / "paired.json"
    for path in (official, data, plan, paired):
        path.write_text("{}")
    assets = dict(namespace=NAMESPACE, official=str(official), data_manifest=str(data),
                  stream_root=str(stream), paired_inputs=str(paired), official_sha256=digest(official),
                  data_sha256=digest(data), plan_sha256=digest(plan), paired_sha256=digest(paired))
    monkeypatch.setattr(r2_runs, "check_assets", lambda *a, **k: assets)
    out, registration = tmp_path / "out", tmp_path / "registration.json"
    r2_runs.prepare_run(data, stream, official, out, registration)
    registered = backend.check_authorization(registration, RECIPE, "all", NAMESPACE, out)
    assert registered["authorized_recipes"] == [RECIPE]
    calls, locks = [], []
    @contextmanager
    def lock(*args):
        locks.append(args)
        yield {}
    monkeypatch.setattr(runtime, "gpu_worker", lock)
    def fake_train(**kwargs):
        calls.append(kwargs)
        return {"status": "COMPLETE"}
    monkeypatch.setattr(backend, "train_recipe", fake_train)
    resume = None
    if mode in ("safe_resume", "completed", "failed"):
        folder = out / "training" / RECIPE
        folder.mkdir(parents=True)
        resume = folder / "latest.pt"
        resume.write_text("fake checkpoint")
        (folder / "status.json").write_text(json.dumps(dict(
            status="OOM" if mode == "failed" else "INTERRUPTED_SAFE",
            checkpoint={"sha256": digest(resume)})))
        monkeypatch.setattr(r2_checkpoints, "read", lambda p: (
            {"global_step": 6000 if mode == "completed" else 100}, "main", "training", digest(p)))
    if mode == "changed_asset":
        plan.write_text('{"tampered":true}')
    kwargs = dict(official=official, data_manifest=data, stream_root=stream, out=out,
                  registration=registration, attempt_id="a01", resume=resume)
    if mode in ("completed", "failed", "changed_asset"):
        with pytest.raises(ValueError):
            r2_runs.train(**kwargs)
        assert not locks and not calls
    else:
        assert r2_runs.train(**kwargs)["status"] == "COMPLETE"
        assert len(locks) == len(calls) == 1
        assert calls[0]["recipe"] == RECIPE and calls[0]["hybrid_schedule"] == "all"
        assert calls[0]["resume"] == resume
