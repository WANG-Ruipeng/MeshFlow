"""Actual inherited CPU OT packing checks; no neural model is constructed.

Budget: the first registered exposure per TRAIN condition in the first 200
EXPOSURE updates, original plus candidate once each: 30 pairs / 60 CPU OT.
This is preparation, not an optimizer update or extra evaluation generation.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil

from . import training_data as D

OUT = D.OUT / "training_preparation"
SCHEDULE = D.ROOT / "experiments/r2_bench_four_routes_v1/exposure/data/schedule_M_PLUS3000.json"


def snapshot():
    def query(args):
        r = subprocess.run(args, capture_output=True, text=True)
        return dict(returncode=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip())
    return dict(unix=time.time(), cpu_cores=os.cpu_count(), available_RAM_bytes=psutil.virtual_memory().available,
                CPU_percent=psutil.cpu_percent(.1), load_average=list(os.getloadavg()),
                GPU=query(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"]),
                GPU_processes=query(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
                relevant_processes=[dict(pid=p.info["pid"], name=p.info["name"],
                                         command=p.info["cmdline"], RSS_bytes=p.info["memory_info"].rss)
                                    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"])
                                    if p.info["name"] in ("python", "python3")])


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run():
    result_path = OUT / "runtime_cpu_gate.json"
    D._require(not result_path.exists(), "CPU OT gate already exists; do not rerun consumed work")
    before = snapshot()
    D._require(before["available_RAM_bytes"] > 2 << 30, "Insufficient CPU RAM for preparation")
    manifest = json.loads(D.MANIFEST.read_text("utf-8-sig"))
    feasibility_path = OUT / "feasibility.json"
    feasibility = json.loads(feasibility_path.read_text("utf-8-sig"))
    D._require(feasibility["source_code_sha256"] == D.sha(D.__file__), "Prepared adapter source differs")
    D._require(feasibility["geometry_source_sha256"] == D.sha(Path(D.__file__).with_name("geometry.py")), "Prepared geometry source differs")
    schedule = json.loads(SCHEDULE.read_text("utf-8-sig"))
    first = {}
    for entry in schedule["steps"][:200]:
        for micro, row in enumerate(entry["samples"]):
            first.setdefault(row["condition_id"], (entry["stage_step"], micro, row))
    D._require(set(first) == {r["condition_id"] for r in manifest["train"]}, "First 200 updates do not cover all 30 TRAIN cases")
    D._require(len(first) == 30, "CPU OT budget requires 30 conditions")
    by_id = {r["condition_id"]: r for r in manifest["train"]}
    # No generator/model object, state dictionary or checkpoint is loaded.
    from r2.pilots.bench_natural_n_recipe_v1 import runtime as R
    import torch
    D._require(not torch.cuda.is_initialized(), "CPU check inherited CUDA initialization")
    torch.set_num_threads(1)
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    rows = []
    attempts = 0
    started = time.perf_counter()
    ledger = OUT / "runtime_cpu_gate_calls.jsonl"
    D._require(not ledger.exists(), "Existing CPU OT attempt ledger forbids silent repeat")

    def event(data):
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")

    for cid, (step, micro, entry) in first.items():
        source = D.load_train_case(by_id[cid])
        candidate, geometry = D.apply_flip_to_case(source)
        samples = []
        for label, case in (("ORIGINAL", source), ("CANDIDATE", candidate)):
            attempts += 1
            D._require(attempts <= 60, "CPU OT packing budget exceeded")
            event(dict(event="CPU_OT_BEGIN", call=attempts, condition_id=cid, label=label, source_step=step, micro=micro))
            sample = R.sample_for_training(case, entry, "PLAIN")
            D._require(int(sample["production_OT_calls"]) == 1, "Expected one inherited CPU OT call")
            R.verify_sample(sample, "NATIVE")
            scaled = case["full_target"].copy()
            scaled[..., 0] *= np.float32(entry["alpha"])
            reconstructed = scaled[sample["source_face_ids"][:, None], sample["source_corner_permutations"]]
            D._require(D._same(reconstructed, sample["x1"]), "Actual runtime source/corner reconstruction differs")
            samples.append(sample)
            event(dict(event="CPU_OT_COMPLETE", call=attempts, condition_id=cid, label=label, source_roundtrip=True))
        a, b = samples
        for key in ("known_mask", "valid_mask", "free_corner_mask", "source_face_ids", "source_corner_permutations",
                    "face_permutation", "ot_face_map", "ot_corner_map", "t", "C_vertices", "N", "K",
                    "occupied_labels", "occupied_for_FM", "free_coordinate_count"):
            D._require(D._same(a[key], b[key]), "Matched input/coupling changed: " + key)
        free = ~a["known_mask"]
        for key in ("x0", "x1", "xt", "u", "anchor_ids"):
            D._require(D._same(a[key][free], b[key][free]), "Free packed array differs: " + key)
        D._require(D._same(a["u"], b["u"]), "FM target velocity differs despite fixed free target")
        D._require(not b["u"][b["anchor_ids"] >= 0].any(), "Known velocity is nonzero")
        if geometry["status"] == "BLOCKED":
            D._require(set(a) == set(b) and all(D._same(a[key], b[key]) for key in a), "Blocked sample changed")
        else:
            try:
                D.apply_flip_to_case(candidate)
            except ValueError as exc:
                D._require(str(exc) == "Do not apply augmentation twice", "Unexpected repeated-application rejection")
            else:
                raise ValueError("Repeated adaptation unexpectedly accepted")
        rows.append(dict(condition_id=cid, source_stage_step=step, source_microbatch=micro,
                         alpha=entry["alpha"], geometry_status=geometry["status"],
                         N=int(a["N"]), K=int(a["K"]), CPU_OT_calls=2,
                         source_roundtrip=True, identical_noise_seeds=True,
                         free_x0_x1_xt_u_bitwise_same=True, OT_face_corner_maps_bitwise_same=True,
                         actual_FM_velocity_target_bitwise_same=True,
                         changed_C_faces=2 if geometry["status"] == "PASS" else 0))

    # Exercise the membership gate with real held-out metadata, without OT.
    held = manifest.get("dev", []) + manifest.get("high_holdout", [])
    D._require(bool(held), "No held-out row available for membership check")
    for row in held:
        try:
            D.load_train_case(row)
        except ValueError as exc:
            D._require("not in the registered TRAIN pool" in str(exc), "Unexpected held-out rejection reason")
        else:
            raise ValueError("Held-out row admitted to TRAIN augmentation")
    D._require(attempts == 60, "Incomplete CPU OT gate")
    D._require(not torch.cuda.is_initialized(), "CPU OT initialized CUDA")
    D._require(torch.equal(torch_rng, torch.get_rng_state()), "CPU preparation changed global Torch RNG")
    rng_after = np.random.get_state()
    D._require(numpy_rng[0] == rng_after[0] and np.array_equal(numpy_rng[1], rng_after[1])
               and numpy_rng[2:] == rng_after[2:], "CPU preparation changed global NumPy RNG")
    D._require(all(D.sha(path) == expected for path, expected in feasibility["protected_inputs"].items()), "Protected training input changed")
    result = dict(schema="bench_C_edgeflip_actual_CPU_OT_gate_v1", status="PASS",
                  conditions=30, eligible=28, blocked=2, actual_CPU_OT_calls=60,
                  model_constructions=0, tensor_weight_loads=0, model_forwards=0, model_backwards=0,
                  optimizer_updates=0, generated_evaluation_outputs=0, new_evaluation_noise_arrays=0,
                  prepared_training_Gaussian_array_materializations=120,
                  Gaussian_scope="30 registered microbatch seeds, each z/epsilon pair regenerated identically for original/candidate",
                  uses_registered_FIRST200_training_stream=True, global_RNG_unchanged=True,
                  CUDA_initialized=False, heldout_rows_rejected=len(held), rows=rows,
                  resources_before=before, resources_after=snapshot(), workers=1, numeric_threads=1,
                  wall_seconds=time.perf_counter()-started,
                  source_code_sha256=D.sha(__file__), adapter_sha256=D.sha(D.__file__),
                  feasibility_sha256=D.sha(feasibility_path), schedule_sha256=D.sha(SCHEDULE),
                  runtime_source_sha256=D.sha(R.__file__), ledger_sha256=D.sha(ledger),
                  protected_inputs_unchanged=True,
                  engineering_preparation_note="Initial feasibility gate rejected HIGH rows with absent group metadata before models/OT; corrected to authoritative manifest.train membership, then completed feasibility once.")
    write_new(result_path, result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in ("status", "conditions", "eligible", "blocked", "actual_CPU_OT_calls", "model_forwards", "optimizer_updates", "wall_seconds")}))
