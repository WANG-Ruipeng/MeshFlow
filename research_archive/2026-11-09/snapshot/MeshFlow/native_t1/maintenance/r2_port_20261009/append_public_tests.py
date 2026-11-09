from pathlib import Path
p=Path('/mnt/e/MeshFlow/r2/tests/test_r2_public.py')
p.write_text(p.read_text()+'''
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
''')
print('Added checkpoint/registration gates and frozen-byte tests.')
