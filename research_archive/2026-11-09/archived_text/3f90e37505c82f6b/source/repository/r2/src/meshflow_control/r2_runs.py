"""Register and execute a single R2 run using an already frozen historical stream."""
from pathlib import Path

from .r2_spec import RECIPE


def source_root():
    root = Path(__file__).resolve().parents[2]
    if not (root / "tools/train_recipe_factorial.py").is_file():
        raise RuntimeError("Training registration requires the r2 source checkout (editable install)")
    return root


def external_path(path):
    path = Path(path).resolve()
    root = source_root()
    # In the monorepo, prohibit artifacts anywhere under the parent checkout.
    protected = root.parent if (root.parent / ".git").exists() else root
    if path == protected or path.is_relative_to(protected):
        raise ValueError("Keep run artifacts outside the source repository")
    return path


def check_assets(data_manifest, stream_root, official=None):
    """Read declarations, hashes and the existing plan; never run OT or construct C."""
    from .data.io import digest, read_json
    from .data.recipe_factorial import RecipeFactorialStream
    data_manifest = Path(data_manifest).resolve()
    if data_manifest.name != "train_manifest.json":
        raise ValueError("The historical loader requires train_manifest.json")
    stream = RecipeFactorialStream(data_manifest.parent, stream_root,
                                  recipe=RECIPE, hybrid_schedule="all", require_cache=True)
    if stream.total_updates != 6000:
        raise ValueError("R2 requires the frozen 6000-update stream")
    paired = read_json(stream.paired_path)
    if (paired.get("plan_sha256") != stream.plan_sha256
            or len(paired.get("records", [])) != 48000
            or {r["sample_index"] for r in paired["records"]} != set(range(48000))):
        raise ValueError("Frozen R2 paired inputs are incomplete or have a different plan")
    result = dict(status="PASS", recipe=RECIPE, namespace=stream.namespace,
                  data_manifest=str(data_manifest), data_sha256=digest(data_manifest),
                  stream_root=str(Path(stream_root).resolve()),
                  plan_sha256=stream.plan_sha256, paired_inputs=str(stream.paired_path),
                  paired_sha256=digest(stream.paired_path), updates=6000, samples=48000,
                  cache_scope="Declarations checked; worker verifies each cache before use",
                  model_forwards=0, OT_calls=0, new_conditions=0)
    if official is not None:
        from .training.schedule40 import OFFICIAL_FILE_SHA256
        actual = digest(official)
        if actual != OFFICIAL_FILE_SHA256:
            raise ValueError("Official Chair EMA checkpoint file identity differs")
        result.update(official=str(Path(official).resolve()), official_sha256=actual)
    return result


def prepare_run(data_manifest, stream_root, official, out, registration):
    from .data.io import digest, freeze_json
    from .training.recipe_factorial import AUTH_SCHEMA, EXPERIMENT
    out, registration = external_path(out), external_path(registration)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("A new R2 registration requires an empty run directory")
    if registration.exists():
        raise FileExistsError(registration)
    assets = check_assets(data_manifest, stream_root, official)
    root = source_root()
    sources = sorted((root / "src/meshflow_control").rglob("*.py"))
    sources.append(root / "tools/train_recipe_factorial.py")
    value = dict(schema=AUTH_SCHEMA, experiment=EXPERIMENT,
                 namespace=assets["namespace"], out=str(out),
                 authorized_recipes=[RECIPE], hybrid_schedules={RECIPE: "all"},
                 updates_per_recipe=6000, effective_batch=8, microbatch=1,
                 source_hashes={str(p.relative_to(root)): digest(p) for p in sources},
                 assets=assets, preparation_model_calls=0,
                 note="Single R2 run; no sibling arms, automatic sampling or retries")
    freeze_json(registration, value)
    return dict(status="REGISTERED", registration=str(registration), out=str(out),
                recipe=RECIPE, model_calls=0, training_updates=0)


def train(*, official, data_manifest, stream_root, out, registration,
          attempt_id, resume=None):
    from . import runtime
    from .data.io import digest, read_json
    from .training.recipe_factorial import check_authorization, train_recipe
    from .data.recipe_factorial import NAMESPACE
    out = external_path(out)
    registration = Path(registration).resolve()
    registered = check_authorization(registration, RECIPE, "all", NAMESPACE, out)
    if registered["authorized_recipes"] != [RECIPE]:
        raise ValueError("The maintained R2 launcher requires a single-recipe registration")
    assets = registered.get("assets", {})
    expected_paths = dict(official=official, data_manifest=data_manifest, stream_root=stream_root)
    for key, path in expected_paths.items():
        if assets.get(key) != str(Path(path).resolve()):
            raise ValueError("Registered R2 asset path differs: " + key)
    for path, key in ((official, "official_sha256"), (data_manifest, "data_sha256"),
                      (Path(stream_root) / "recipe_plan.json", "plan_sha256"),
                      (assets["paired_inputs"], "paired_sha256")):
        if digest(path) != assets.get(key):
            raise ValueError("Registered R2 asset hash differs: " + key)
    folder = out / "training" / RECIPE
    if resume is None and folder.exists():
        raise FileExistsError("Existing training requires explicit, valid --resume")
    if resume is not None:
        from .r2_checkpoints import read
        value, stage, kind, _ = read(resume)
        if stage != "main" or kind != "training" or value["global_step"] >= 6000:
            raise ValueError("Resume is only for an interrupted main R2 run below step 6000")
        del value
        previous = read_json(folder / "status.json")
        if previous.get("status") != "INTERRUPTED_SAFE":
            raise ValueError("Unresolved failure or incomplete attempt requires explicit recovery")
        from .data.io import digest
        if digest(resume) != previous.get("checkpoint", {}).get("sha256"):
            raise ValueError("Resume must use the latest safely interrupted checkpoint")
    with runtime.gpu_worker(out / "audit/gpu_owner", "r2_training"):
        return train_recipe(recipe=RECIPE, hybrid_schedule="all", official=official,
                            data_manifest=data_manifest, stream_root=stream_root,
                            out=out, registration=registration, attempt_id=attempt_id,
                            resume=resume, device="cuda")
