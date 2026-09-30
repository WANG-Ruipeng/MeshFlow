"""Bounded isolation regression against saved T1 trajectories (no legacy import)."""
from .runtime import configure_stable_runtime
from pathlib import Path
import ast
import json
import time
import sys
import numpy as np
import torch
from .checkpoint import load_t1, REPO, T1_STATE_SHA256
from .sampling import clamped_sample, make_noise, new_counts
from .artifacts import atomic_json, file_sha256, state_sha256, array_hash

SOURCE = REPO / "experiments/mf_h1_local0/outputs/native_patch_20260929_010000"
REFERENCE = REPO / "experiments/mf_h1_local0/outputs/t1_native_confirmation_20260929_171056"
INPUT_SHA256 = "e80d52e0906bc762475f35a2526cb99ff5b52858c9d2e6a38d09018e5d5aea73"


def _definition_ast(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    return ast.dump(node, include_attributes=False)


def run_regression(out):
    """Exactly P0/P1 K12 seed9401: 2 rollouts, 100 forwards, zero updates."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    tick = time.perf_counter()
    report = dict(status="RUNNING", counts=new_counts(), rows=[], seed=9401, K=12,
                  max_rollouts=2, max_forwards=100, model_parameter_updates=0,
                  numerical_tolerance=0, historical_files_modified=False)
    def save():
        atomic_json(out / "run.json", report)
    save()
    try:
        legacy_modules = [n for n in sys.modules if n == "experiments" or n.startswith("experiments.")]
        if legacy_modules:
            raise RuntimeError("Historical Python modules were imported: " + repr(legacy_modules))
        old_source = REPO / "experiments/mf_h1_local0/native_inpainting_model.py"
        current_source = Path(__file__).with_name("model.py")
        names = ("_face_select", "_corners", "native_block_forward", "native_final_forward", "NativeInpaintingModel")
        report["model_definition_AST_equals_source"] = {name: _definition_ast(old_source, name) == _definition_ast(current_source, name) for name in names}
        if not all(report["model_definition_AST_equals_source"].values()):
            raise RuntimeError("Extracted model math differs from source")
        inputs = SOURCE / "inputs.npz"
        if file_sha256(inputs) != INPUT_SHA256:
            raise RuntimeError("Archived conditioning bank changed")
        records = json.loads((REFERENCE / "results.json").read_text())["generated"]
        refs = [next(x for x in records if x["model"] == "T1" and x["parent"] == parent and x["seed"] == 9401) for parent in (0, 1)]
        paths = [REFERENCE / "raw" / Path(row["raw_path"]).name for row in refs]
        for row, path in zip(refs, paths):
            if file_sha256(path) != row["raw_sha256"]:
                raise RuntimeError("Reference trajectory changed")
        configure_stable_runtime()
        model, report["load"] = load_t1()
        report["state_before"] = state_sha256(model)
        save()
        z = make_noise(9401)
        with np.load(inputs, allow_pickle=False) as archive:
            conditions = [archive[f"parent{parent}_K12_constraints"].copy() for parent in (0, 1)]
        for parent, C in enumerate(conditions):
            if report["counts"]["rollout_attempts"] >= 2:
                raise RuntimeError("Two-rollout regression budget exceeded")
            raw, trajectory, audit = clamped_sample(model, z.cuda(), torch.from_numpy(C).cuda(), report["counts"])
            with np.load(paths[parent], allow_pickle=False) as stored:
                expected = stored["path"]
                equal = np.array_equal(trajectory.view(np.uint32), expected.view(np.uint32))
                endpoint_equal = np.array_equal(raw.view(np.uint32), stored["output"].view(np.uint32))
            c_equal = bool((trajectory[:, :12].view(np.uint32) == C.reshape(1, 12, 9).view(np.uint32)).all())
            report["rows"].append(dict(parent=parent, seed=9401, original_raw_sha256=refs[parent]["raw_sha256"],
                reference=str(paths[parent]), all51_states_bitwise_equal=equal,
                endpoint_bitwise_equal=endpoint_equal, C_all51_bitwise_equal=c_equal,
                raw_array_sha256=array_hash(raw), path_array_sha256=array_hash(trajectory), sampling=audit))
            save()
            if not (equal and endpoint_equal and c_equal):
                raise RuntimeError("Isolated sampler trajectory differs; no tolerance relaxation or automatic retry")
        report["state_after"] = state_sha256(model)
        if report["state_after"] != report["state_before"] or report["state_after"] != T1_STATE_SHA256:
            raise RuntimeError("Frozen model changed")
        if report["counts"]["model_forward_attempts"] != 100 or report["counts"]["model_forward_returns"] != 100 or report["counts"]["complete_rollouts"] != 2:
            raise RuntimeError("Regression accounting differs")
        report["legacy_modules_loaded"] = [n for n in sys.modules if n == "experiments" or n.startswith("experiments.")]
        if report["legacy_modules_loaded"]:
            raise RuntimeError("Legacy module imported during sampling")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="STOPPED", error=repr(error), automatic_retry=False)
        raise
    finally:
        report["seconds"] = time.perf_counter() - tick
        save()
    print(json.dumps(dict(status=report["status"], counts=report["counts"], all51_states_bitwise_equal=True)))
    return report
