"""Freeze paired anchor-denoiser inputs on CPU, without torch or model calls.

``prepare_stream()`` creates/verifies one immutable shared 1000x8 training
stream plus 18 synthetic DEV inputs. ``load_batch(step, manifest=None)``
returns (arrays, manifest_row); slice arrays with offsets[j:j+2] and reshape
x_d/target_d to [1,N,9], known to [1,N]. Divide each microbatch free squared
error SUM by the one stored batch free_coordinate_denominator, then accumulate.

Faces and their three corner copies are permuted jointly. Source/interface
identifiers are audit metadata only, never model inputs. All Gaussian arrays
are saved once; resume verifies receipts/hashes rather than resampling them.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from .common import OUT, ROOT, SCALE, SIGMA, ah, checked, csvwrite, event, read, save_npz, seed, sha, write

STREAM = OUT / "stream"
WHITE = OUT / "data_audit/training_whitelist.json"
DEV = OUT / "data_audit/dev_conditions.json"
UPDATES, BATCH = 1000, 8
SCHEMA = "r2_anchor_denoiser_frozen_stream_v1"
WIDTH_SOURCE = ROOT / "r2/src/meshflow_control/data/dataset.py"


def _immutable_json(path: Path, value: dict) -> None:
    path = checked(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Immutable stream metadata changed: {path}")
    else:
        write(path, value)


def _source(row: dict, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    cid = row["condition_id"]
    if cid not in cache:
        path = Path(row["npz"])
        if sha(path) != row["sha256"]:
            raise ValueError(f"Condition source hash changed: {cid}")
        with np.load(path, allow_pickle=False) as saved:
            c, y = saved["C"].copy(), saved["Y"].copy()
        n, k = int(row["N"]), int(row["K"])
        if c.dtype != np.float32 or y.dtype != np.float32 or c.shape != (k, 3, 3) or y.shape != (n-k, 3, 3):
            raise ValueError(f"C/Y contract differs: {cid}")
        if not 0 < k < n or not np.isfinite(c).all() or not np.isfinite(y).all():
            raise ValueError(f"Invalid C/Y: {cid}")
        source_ids = np.asarray(row["source_face_ids"] + row["free_source_ids"], np.int64)
        if source_ids.shape != (n,) or not np.array_equal(np.sort(source_ids), np.arange(n)):
            raise ValueError(f"C/Y source IDs not a complete partition: {cid}")
        cache[cid] = (c, y)
    return cache[cid]


def _rng(domain: str, *identity) -> np.random.RandomState:
    return np.random.RandomState(seed(domain, *identity))


def _plan(white: dict) -> dict:
    parents = sorted(white["parent_ids"])
    if len(parents) < BATCH:
        raise ValueError("DATA_CONTRACT_BLOCKED: fewer than eight unique parents")
    pool = defaultdict(list)
    for row in white["rows"]:
        pool[row["uid"]].append(row)
    for uid in pool:
        pool[uid].sort(key=lambda row: row["condition_id"])
    if sorted(pool) != parents:
        raise ValueError("Whitelist parents and rows differ")
    rows = []
    for step in range(1, UPDATES + 1):
        chosen = _rng("training_parents", step).choice(len(parents), BATCH, replace=False)
        width_slots = set(int(i) for i in _rng("training_width_slots", step).choice(BATCH, 4, replace=False))
        for slot, parent_index in enumerate(chosen):
            index = (step - 1) * BATCH + slot
            uid = parents[int(parent_index)]
            options = pool[uid]
            condition = options[int(_rng("training_condition", index).randint(len(options)))]
            clean = slot not in width_slots
            alpha = np.float32(1) if clean else np.float32(_rng("training_width", index).uniform(.9, 1.1))
            if not clean and alpha == np.float32(1):
                raise ValueError("Registered width draw rounded to one; no silent redraw")
            rows.append(dict(sample_index=index, step=step, slot=slot, uid=uid,
                condition_id=condition["condition_id"], N=int(condition["N"]), K=int(condition["K"]),
                region_type=condition["region_type"], tier=condition["tier"],
                clean=clean, alpha=float(alpha),
                epsilon_seed=seed("training_epsilon", index),
                face_permutation_seed=seed("training_face_permutation", index),
                corner_permutation_seed=seed("training_corner_permutation", index),
                npz=condition["npz"], npz_sha256=condition["sha256"]))
    return dict(schema=SCHEMA, source_sha256=sha(Path(__file__)), whitelist_sha256=sha(WHITE),
        updates=UPDATES, batch_size=BATCH, microbatch_size=1, rows=rows,
        parent_schedule="Each batch uniformly samples 8 different parents without replacement; batches independent",
        condition_schedule="Uniform within selected parent's legal condition pool; no region/tier quotas",
        augmentation="Exactly 4 clean+4 x-axis width per batch; x*=FP32(alpha), origin fixed; alpha uniform[.9,1.1)",
        width_source=str(WIDTH_SOURCE), width_source_sha256=sha(WIDTH_SOURCE),
        RNG="Independent stateless domain seeds -> NumPy RandomState MT19937; no torch/global RNG",
        sigma_denoiser_units=SIGMA, scale=SCALE,
        noise="Independent scalar Gaussian for every free face corner coordinate; zero known noise",
        permutation="joint face and independent within-face corner permutations; inverse maps saved",
        arms_share_identical_files=True,
        free_MSE="sum over free squared scalar errors / (9*sum_batch free_faces)",
        actual_parent_exposure=dict(sorted(Counter(r["uid"] for r in rows).items())),
        actual_condition_exposure=dict(sorted(Counter(r["condition_id"] for r in rows).items())),
        actual_clean_parent_exposure=dict(sorted(Counter(r["uid"] for r in rows if r["clean"]).items())),
        actual_width_parent_exposure=dict(sorted(Counter(r["uid"] for r in rows if not r["clean"]).items())))


def _permute(array: np.ndarray, faces: np.ndarray, corners: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array[faces[:, None], corners])


def _unpermute(array: np.ndarray, inverse_faces: np.ndarray, inverse_corners: np.ndarray) -> np.ndarray:
    n = len(inverse_faces)
    return np.ascontiguousarray(array[np.arange(n)[:, None], inverse_corners][inverse_faces])


def _sample(plan_row: dict, source_row: dict, cache: dict) -> dict[str, np.ndarray]:
    c, y = _source(source_row, cache)
    n, k = plan_row["N"], plan_row["K"]
    target_model = np.concatenate((c, y), axis=0)
    alpha = np.float32(plan_row["alpha"])
    if alpha != np.float32(1):
        # Byte-matched arithmetic to the existing dataset.task width policy.
        target_model.reshape(-1, 3)[:, 0] *= alpha
    target_d = target_model * np.float32(SCALE)
    epsilon = np.zeros((n, 3, 3), np.float32)
    epsilon[k:] = np.random.RandomState(plan_row["epsilon_seed"]).standard_normal((n-k, 3, 3)).astype(np.float32)
    x_d = target_d.copy()
    x_d[k:] += np.float32(SIGMA) * epsilon[k:]
    faces = np.random.RandomState(plan_row["face_permutation_seed"]).permutation(n).astype(np.int64)
    corner_rng = np.random.RandomState(plan_row["corner_permutation_seed"])
    corners = np.asarray([corner_rng.permutation(3) for _ in range(n)], np.int64)
    inverse_faces = np.argsort(faces).astype(np.int64)
    inverse_corners = np.argsort(corners, axis=1).astype(np.int64)
    known = (np.arange(n) < k)[faces]
    result = dict(x_d=_permute(x_d, faces, corners), target_d=_permute(target_d, faces, corners),
        target_model=_permute(target_model, faces, corners), epsilon=_permute(epsilon, faces, corners),
        known=known, face_permutations=faces, inverse_face_permutations=inverse_faces,
        corner_permutations=corners, inverse_corner_permutations=inverse_corners)
    if result["x_d"][known].tobytes() != result["target_d"][known].tobytes() or np.any(result["epsilon"][known]):
        raise ValueError("Known input was perturbed")
    if _unpermute(result["target_model"], inverse_faces, inverse_corners).tobytes() != target_model.tobytes():
        raise ValueError("Joint permutation inverse failed")
    if _unpermute(result["x_d"], inverse_faces, inverse_corners).tobytes() != x_d.tobytes():
        raise ValueError("Input permutation inverse failed")
    if not np.isfinite(result["x_d"]).all() or not np.isfinite(result["target_d"]).all():
        raise ValueError("Nonfinite frozen input")
    return result


def _batch(step: int, rows: list[dict], sources: dict, cache: dict, plan_sha: str) -> dict:
    path = STREAM / "batches" / f"{step:06d}.npz"
    receipt_path = path.with_suffix(".json")
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["plan_sha256"] != plan_sha or receipt["step"] != step or receipt["rows"] != rows or sha(path) != receipt["sha256"]:
            raise ValueError("Frozen batch receipt/hash changed")
        return receipt
    if path.exists():
        raise ValueError("Frozen batch exists without complete receipt; do not overwrite")
    if len(rows) != BATCH or len({r["uid"] for r in rows}) != BATCH or sum(r["clean"] for r in rows) != 4:
        raise ValueError("Invalid paired batch population")
    samples = [_sample(row, sources[row["condition_id"]], cache) for row in rows]
    arrays = {key: np.concatenate([sample[key] for sample in samples], axis=0) for key in samples[0]}
    arrays["offsets"] = np.asarray([0] + list(np.cumsum([row["N"] for row in rows])), np.int64)
    denominator = 9 * sum(row["N"] - row["K"] for row in rows)
    arrays["free_coordinate_denominator"] = np.asarray(denominator, np.int64)
    arrays["step"] = np.asarray(step, np.int64)
    if denominator != 9 * int((~arrays["known"]).sum()):
        raise ValueError("Free MSE denominator mismatch")
    save_npz(path, **arrays)
    with np.load(path, allow_pickle=False) as reopened:
        if set(reopened.files) != set(arrays) or any(reopened[key].tobytes() != value.tobytes() for key, value in arrays.items()):
            raise ValueError("Frozen batch roundtrip differs")
    receipt = dict(status="PASS", step=step, path=str(path), sha256=sha(path), plan_sha256=plan_sha,
        rows=rows, offsets=arrays["offsets"].tolist(), free_coordinate_denominator=denominator,
        input_array_sha256=ah(arrays["x_d"]), target_array_sha256=ah(arrays["target_d"]),
        known_array_sha256=ah(arrays["known"]), epsilon_array_sha256=ah(arrays["epsilon"]),
        array_shapes={key: list(value.shape) for key, value in arrays.items()},
        array_dtypes={key: str(value.dtype) for key, value in arrays.items()})
    write(receipt_path, receipt)
    return receipt


def _synthetic(dev: dict, cache: dict) -> dict:
    rows = []
    if len(dev["rows"]) != 6:
        raise ValueError("Registered synthetic DEV requires six existing conditions")
    for source in sorted(dev["rows"], key=lambda row: row["condition_id"]):
        cid, n, k = source["condition_id"], int(source["N"]), int(source["K"])
        c, y = _source(source, cache)
        interface = read(source["source_interface_json"])
        if sha(source["source_interface_json"]) != source["source_interface_sha256"]:
            raise ValueError("DEV source interface hash changed")
        target_model = np.concatenate((c, y), axis=0)
        target_d = target_model * np.float32(SCALE)
        supported_local = np.asarray(interface["supported_Y_local_corner_indices"], np.int64).reshape(-1, 2)
        if len(supported_local) and (np.any(supported_local[:, 0] >= n-k) or np.any(supported_local[:, 0] < 0)
                                     or np.any(supported_local[:, 1] < 0) or np.any(supported_local[:, 1] >= 3)):
            raise ValueError("Synthetic interface corner indices out of range")
        for noise_index in (-1, 0, 1):
            clean = noise_index == -1
            suffix = "clean" if clean else f"noise{noise_index}"
            sid = f"{cid}__{suffix}"
            path = STREAM / "synthetic" / f"{sid}.npz"
            receipt_path = path.with_suffix(".json")
            identity = dict(synthetic_id=sid, condition_id=cid, uid=source["uid"], region_type=source["region_type"],
                tier=source["tier"], N=n, K=k, clean=clean, noise_index=noise_index,
                epsilon_seed=None if clean else seed("synthetic_dev_epsilon", cid, noise_index),
                source_npz=source["npz"], source_npz_sha256=source["sha256"],
                source_interface_json=source["source_interface_json"], source_interface_sha256=source["source_interface_sha256"],
                dev_manifest_sha256=sha(DEV))
            if receipt_path.exists():
                receipt = read(receipt_path)
                if any(receipt[key] != value for key, value in identity.items()) or sha(path) != receipt["sha256"]:
                    raise ValueError("Frozen synthetic input changed")
                rows.append(receipt)
                continue
            if path.exists():
                raise ValueError("Synthetic input exists without receipt; do not overwrite")
            epsilon = np.zeros((n, 3, 3), np.float32)
            if not clean:
                epsilon[k:] = np.random.RandomState(identity["epsilon_seed"]).standard_normal((n-k, 3, 3)).astype(np.float32)
            x_d = target_d.copy()
            x_d[k:] += np.float32(SIGMA) * epsilon[k:]
            arrays = dict(x_d=x_d, target_d=target_d, target_model=target_model, C=c,
                known=np.arange(n) < k, epsilon=epsilon,
                supported_Y_local_corner_indices=supported_local,
                supported_Y_flat_corner_indices=np.asarray(interface["supported_Y_flat_corner_indices"], np.int64),
                supported_joint_C_then_Y_flat_corner_indices=np.asarray(interface["supported_joint_C_then_Y_flat_corner_indices"], np.int64))
            if x_d[:k].tobytes() != target_d[:k].tobytes() or (clean and x_d.tobytes() != target_d.tobytes()):
                raise ValueError("Synthetic clean/known coordinates changed")
            save_npz(path, **arrays)
            with np.load(path, allow_pickle=False) as reopened:
                if any(reopened[key].tobytes() != value.tobytes() for key, value in arrays.items()):
                    raise ValueError("Synthetic input roundtrip differs")
            receipt = dict(status="PASS", **identity, path=str(path), sha256=sha(path),
                input_array_sha256=ah(x_d), target_array_sha256=ah(target_d),
                supported_free_corners=len(supported_local),
                note="Same-shape denoising diagnostic only; never source-GT ranking for generated RAW")
            write(receipt_path, receipt)
            rows.append(receipt)
    manifest = dict(schema=SCHEMA + "_synthetic", status="PASS", rows=rows, inputs=18,
        clean_inputs=6, noisy_inputs=12, sigma_denoiser_units=SIGMA, scale=SCALE,
        frozen_before_training=True, no_model_forward=True, no_DEV_training=True)
    _immutable_json(STREAM / "synthetic/manifest.json", manifest)
    return manifest


def prepare_stream() -> dict:
    start = time.perf_counter()
    white, dev = read(WHITE), read(DEV)
    if white["status"] != "PASS":
        raise ValueError("Training whitelist contract is not PASS")
    cache = {}
    sources = {row["condition_id"]: row for row in white["rows"]}
    if len(sources) != len(white["rows"]):
        raise ValueError("Duplicate training condition IDs")
    for row in white["rows"] + dev["rows"]:
        _source(row, cache)
    plan_path = STREAM / "plan.json"
    if plan_path.exists():
        plan = read(plan_path)
        if plan["schema"] != SCHEMA or plan["source_sha256"] != sha(Path(__file__)) or plan["whitelist_sha256"] != sha(WHITE):
            raise ValueError("Frozen stream plan/source identity changed")
    else:
        plan = _plan(white)
        write(plan_path, plan)
        csvwrite(STREAM / "plan_rows.csv", plan["rows"])
    plan_sha = sha(plan_path)
    batches = []
    for step in range(1, UPDATES + 1):
        rows = plan["rows"][(step-1)*BATCH:step*BATCH]
        batches.append(_batch(step, rows, sources, cache, plan_sha))
    manifest = dict(schema=SCHEMA, status="PASS", plan_path=str(plan_path), plan_sha256=plan_sha,
        batches=batches, updates=UPDATES, samples=UPDATES*BATCH,
        whitelist_sha256=sha(WHITE), data_source_sha256=sha(Path(__file__)))
    _immutable_json(STREAM / "manifest.json", manifest)
    synthetic = _synthetic(dev, cache)
    for row in white["rows"] + dev["rows"]:
        if sha(row["npz"]) != row["sha256"]:
            raise ValueError("Protected data changed during stream preparation")
    result = dict(status="PASS", batches=len(batches), samples=UPDATES*BATCH,
        training_parents=len(white["parent_ids"]), training_conditions=len(white["rows"]),
        synthetic_inputs=len(synthetic["rows"]), clean_synthetic=6, noisy_synthetic=12,
        manifest_path=str(STREAM / "manifest.json"), manifest_sha256=sha(STREAM / "manifest.json"),
        plan_sha256=plan_sha, synthetic_manifest_path=str(STREAM / "synthetic/manifest.json"),
        synthetic_manifest_sha256=sha(STREAM / "synthetic/manifest.json"),
        source_sha256=sha(Path(__file__)), model_forwards=0, GPU_calls=0,
        actual_parent_exposure=plan["actual_parent_exposure"],
        elapsed_seconds=time.perf_counter()-start)
    write(STREAM / "completion.json", result)
    event("FROZEN_PAIRED_STREAM_VERIFIED", ledger="data_stream_ledger.jsonl", **result)
    return result


def load_batch(step: int, manifest: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
    if not 1 <= step <= UPDATES:
        raise ValueError("Step outside authorized 1..1000")
    manifest = manifest if manifest is not None else read(STREAM / "manifest.json")
    row = manifest["batches"][step-1]
    if row["step"] != step or sha(row["path"]) != row["sha256"]:
        raise ValueError("Frozen training input changed")
    with np.load(row["path"], allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if ah(arrays["x_d"]) != row["input_array_sha256"] or ah(arrays["target_d"]) != row["target_array_sha256"]:
        raise ValueError("Frozen batch array identity changed")
    return arrays, row


if __name__ == "__main__":
    import json
    print(json.dumps(prepare_stream(), ensure_ascii=False))
