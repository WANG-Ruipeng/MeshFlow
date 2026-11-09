"""Post-Phase-2 component and complete-sampling timings; no parameter updates.

Run only after training has completed and the GPU is available. CUDA event spans
include stream idle time caused by Python within a span. Nested span subtraction
is an estimate of exclusive backbone time, not a kernel profiler or a claim of
end-to-end acceleration. A separate warmed production-sampler campaign saves
eight raw timing outputs; these never enter the main evaluation table.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import adapters as adapter_module
from .adapters import ControlAdapter
from .geometry import primitive_centroids
from .official_wrapper import OfficialWrapper, load_official, state_sha256
from .prompt_dataset import collate_prompts, file_sha256
from .sample import euler_sample, save_raw


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "status": "NOT_RUN"}
    return {"count": len(values), "min": float(values.min()), "mean": float(values.mean()),
            "p05": float(np.quantile(values, .05)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max()),
            "raw": values.tolist()}


def run(checkpoint, output):
    output = Path(output)
    destination = output / "components_benchmark.json"
    if destination.exists():
        raise FileExistsError(destination)
    phase2 = json.loads((output / "phase2_execution.json").read_text(encoding="utf-8"))
    if phase2["status"] != "PASS" or phase2["optimizer_updates"] != 400:
        raise RuntimeError("Component benchmarking is gated on completed successful Phase 2")
    if any(phase2["arms"][arm]["completed_updates_total"] != 200 for arm in ("Q_center", "G_closest")):
        raise RuntimeError("Final adapters must each have exactly 200 completed updates")
    precheck = json.loads((output / "precheck_execution.json").read_text(encoding="utf-8"))
    if file_sha256(checkpoint) != precheck["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint bytes differ from the audited official EMA")
    manifest = json.loads((output / "prompt_manifest.json").read_text(encoding="utf-8"))
    source = next(record for record in manifest["records"]
                  if record["parent_index"] == 0 and record["kind"] == "mixed_2edges_1face")
    if file_sha256(source["path"]) != source["sha256"]:
        raise ValueError("Fixed prompt asset changed")
    with np.load(source["path"], allow_pickle=False) as arrays:
        prompt = {key: arrays[key] for key in ("vertices", "arity", "primitive_mask", "points",
                                              "point_primitive", "point_mask")}
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The shared CUDA BF16 measurement device is unavailable")
    root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(10)
    base, _ = load_official(checkpoint, root / "configs/snet/base-120m-ot-v-chair.yaml", "cuda")
    if state_sha256(base) != precheck["base_state_sha256"]:
        raise ValueError("Loaded backbone differs from the prechecked official EMA")
    models = {}
    for arm in ("Q_center", "G_closest"):
        saved = torch.load(output / arm / "checkpoints/adapter_step0200.pt", map_location="cpu", weights_only=True)
        if saved["step"] != 200 or saved["config"]["arm"] != arm or saved["recoverable"] is not True:
            raise ValueError("Final checkpoint is not the completed arm's update boundary")
        model = ControlAdapter(arm).cuda()
        model.load_state_dict(saved["adapter"], strict=True)
        models[arm] = model.requires_grad_(False).eval()
    adapter_hashes_before = {arm: state_sha256(model) for arm, model in models.items()}
    wrapper = OfficialWrapper(base, models["Q_center"]).eval()
    prompts = collate_prompts([prompt], "cuda")
    # Independent fixed Gaussian Xt; no target endpoint or coupled noise input.
    xt = torch.randn((1, 512, 9), generator=torch.Generator().manual_seed(4101)).cuda()
    corners = xt.reshape(1, 1536, 3)
    t = torch.full((1,), .5, device="cuda", dtype=torch.float32)
    y = torch.tensor([512], device="cuda")
    mask = torch.ones((1, 512), device="cuda", dtype=torch.bool)
    centers = primitive_centroids(prompts["vertices"], prompts["arity"])
    records = []
    active = [None]
    handles = []
    execution_stage = ["components"]
    actual = {"backbone_forward": 0, "component_backbone_forward": 0, "sampling_backbone_forward": 0,
              "geometry_helper_full_forward": 0,
              "geometry_helper_standalone": 0, "q_subtraction_standalone": 0,
              "complete_sampling_calls": 0, "complete_rollouts": 0, "optimizer_updates": 0}
    original_geometry = adapter_module.closest_points_primitives
    original_encode = wrapper.encode
    started = time.perf_counter()
    report = {"status": "RUNNING", "precision": "BF16 autocast, FP32 parameters and evolving coordinates",
              "backend": "PyTorch SDPA math", "tf32": False,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "warmup_per_arm_per_component": 3, "timed_per_arm_per_component": 10,
              "query_chunk": 256, "raw_face_slots": 512, "batch": 1,
              "time": .5, "fixed_xt_seed": 4101, "prompt_parent_uid": source["parent_uid"],
              "prompt_kind": source["kind"], "prompt_sha256": source["sha256"],
              "timing_units": "milliseconds", "phase2_updates_added": 0,
              "full_sampling_repeated": True, "sampling_warmup_per_arm": 1,
              "sampling_timed_per_arm": 3, "timing_rollouts_separate_from_main_evaluation": True,
              "counts": actual}

    def pair_start(label):
        record = active[0]
        if record is None:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        pair = [event, None]
        record["pairs"][label].append(pair)
        return pair

    def pair_end(pair):
        if pair is not None:
            pair[1] = torch.cuda.Event(enable_timing=True)
            pair[1].record()

    def hooks(module, label):
        stack = []
        def before(mod, args):
            stack.append(pair_start(label))
        def after(mod, args, value):
            pair_end(stack.pop())
        handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))

    def count_base(module, args):
        actual["backbone_forward"] += 1
        key = "component_backbone_forward" if execution_stage[0] == "components" else "sampling_backbone_forward"
        actual[key] += 1
    counter_handle = base.register_forward_pre_hook(count_base)
    hooks(base, "backbone_inclusive")
    for model in models.values():
        hooks(model.condition_encoder, "condition_mlp")
        for layer in model.layers:
            hooks(layer, "adapter_layers_inclusive")

    def instrumented_encode(value):
        pair = pair_start("condition_encoding")
        try:
            return original_encode(value)
        finally:
            pair_end(pair)

    def instrumented_geometry(points, vertices, arity):
        kind = active[0]["component"] if active[0] is not None else "outside"
        if kind == "full_forward":
            actual["geometry_helper_full_forward"] += 1
        elif kind == "geometry":
            actual["geometry_helper_standalone"] += 1
        pair = pair_start("finite_primitive_helper")
        try:
            return original_geometry(points, vertices, arity)
        finally:
            pair_end(pair)

    wrapper.encode = instrumented_encode
    adapter_module.closest_points_primitives = instrumented_geometry

    def geometry_only(arm):
        pieces = []
        for start in range(0, 1536, 256):
            points = corners[:, start:start+256]
            if arm == "G_closest":
                closest = adapter_module.closest_points_primitives(points, prompts["vertices"], prompts["arity"])
                pieces.append(points.unsqueeze(-2) - closest)
            else:
                actual["q_subtraction_standalone"] += 1
                pieces.append(points.unsqueeze(-2) - centers.unsqueeze(1))
        # Relation Fourier and dynamic MLP are excluded from this standalone
        # geometry span; both arms concatenate six identically sized chunks.
        return torch.cat(pieces, dim=1)

    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        order_rng = np.random.RandomState(20260928)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
            for component in ("encoder", "geometry", "full_forward"):
                for phase, repetitions in (("warmup", 3), ("timed", 10)):
                    jobs = [(arm, repeat) for arm in ("Q_center", "G_closest") for repeat in range(repetitions)]
                    for index in order_rng.permutation(len(jobs)):
                        arm, repeat = jobs[index]
                        wrapper.adapter = models[arm]
                        record = {"component": component, "phase": phase, "arm": arm,
                                  "repeat": repeat, "pairs": defaultdict(list)}
                        active[0] = record
                        outer = pair_start("outer")
                        if component == "encoder":
                            result = wrapper.encode(prompts).tokens
                        elif component == "geometry":
                            result = geometry_only(arm)
                        else:
                            result = wrapper(xt, t, y, mask=mask, prompts=prompts)
                        pair_end(outer)
                        records.append(record)
                        active[0] = None
                    # A total stream fence separates warmup and timed stages;
                    # individual kernel launches are not synchronously fenced.
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(result).all()):
                        raise RuntimeError(f"Nonfinite {component} benchmark result")
        torch.cuda.synchronize()
        serial = []
        summary = {arm: {} for arm in models}
        for record in records:
            spans = {label: sum(start.elapsed_time(end) for start, end in pairs)
                     for label, pairs in record["pairs"].items()}
            if record["component"] == "full_forward":
                spans["backbone_excluding_adapter_estimate"] = spans["backbone_inclusive"] - spans["adapter_layers_inclusive"]
            serial.append({key: value for key, value in record.items() if key != "pairs"} | {"milliseconds": spans})
        for arm in models:
            for component in ("encoder", "geometry", "full_forward"):
                selected = [record for record in serial if record["arm"] == arm and record["component"] == component and record["phase"] == "timed"]
                labels = sorted({label for record in selected for label in record["milliseconds"]})
                summary[arm][component] = {label: _distribution([record["milliseconds"][label] for record in selected]) for label in labels}
        # These are the already executed randomized-order complete trajectories,
        # including all 50 updates and cached encoding, not new benchmark runs.
        sampling = {arm: _distribution([1000*record["seconds"] for record in phase2["sampling"] if record["arm"] == arm]) for arm in models}
        hashes_after = {arm: state_sha256(model) for arm, model in models.items()}
        base_unchanged = state_sha256(base) == precheck["base_state_sha256"]
        actual.update(wrapper_counts=dict(wrapper.counts),
                      adapter_calls={arm: model.call_count for arm, model in models.items()},
                      encoder_calls={arm: model.encoder_call_count for arm, model in models.items()})
        count_ok = (actual["backbone_forward"] == 26 and wrapper.counts["wrapper_forward"] == 26 and
                    wrapper.counts["adapter_layer"] == 104 and wrapper.counts["condition_encoding"] == 52 and
                    actual["geometry_helper_full_forward"] == 312 and actual["geometry_helper_standalone"] == 78 and
                    actual["q_subtraction_standalone"] == 78)
        report.update(status="COMPONENTS_PASS" if count_ok and base_unchanged and hashes_after == adapter_hashes_before else "FAIL",
                      summary=summary, raw_measurements=serial,
                      existing_full_sampling_ms=sampling, existing_sampling_source="phase2_execution.json",
                      existing_sampling_rows_per_arm=48, independent_condition_inputs_per_arm=24,
                      base_hash_unchanged=base_unchanged, adapter_hashes_before=adapter_hashes_before,
                      adapter_hashes_after=hashes_after, actual_counter_checks_pass=count_ok,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                      measurement_limitations=[
                          "Only ten timed trials per arm and component; report descriptive distributions, not speedup significance.",
                          "CUDA event spans include host-induced stream idle time and instrumentation overhead.",
                          "Exclusive backbone timing subtracts nested adapter spans; condition encoding occurs before the backbone.",
                          "In-forward G helper span covers closest-point computation only; Q subtraction is measured in the separate geometry probe.",
                          "Standalone geometry covers all corners once; a full forward recomputes geometry at each of four adapter layers.",
                          "Full-forward benchmark includes encoding each call; complete sampling caches encoding once per trajectory.",
                          "Existing sampling timings are the already randomized Q/G trajectory timings, not a new warmed timing campaign.",
                          "Cyclic-mismatch inputs repeat another parent's correct prompt at the same noise, so 48 calls per arm represent 24 distinct conditional inputs."])
        if report["status"] != "COMPONENTS_PASS":
            raise RuntimeError("Component counts or frozen-state checks failed before sampling timing")
        report["component_counts"] = dict(actual, wrapper_counts=dict(wrapper.counts))
        # Remove every timing hook and timing monkeypatch. Retain only the cheap
        # actual-call counter and the normal production adapter residual hooks.
        active[0] = None
        adapter_module.closest_points_primitives = original_geometry
        wrapper.encode = original_encode
        for handle in handles:
            handle.remove()
        handles.clear()
        execution_stage[0] = "sampling"
        sampling_records = []
        repeat_outputs = {arm: [] for arm in models}
        report["fair_sampling_measurements"] = sampling_records
        components_peak_allocated = report["peak_cuda_allocated_bytes"]
        components_peak_reserved = report["peak_cuda_reserved_bytes"]
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            for phase, repetitions in (("warmup", 1), ("timed", 3)):
                jobs = [(arm, repeat) for arm in ("Q_center", "G_closest") for repeat in range(repetitions)]
                for index in order_rng.permutation(len(jobs)):
                    arm, repeat = jobs[index]
                    wrapper.adapter = models[arm]
                    # The production sampler clones this identical independent
                    # Gaussian seed. No target or training-coupled noise is used.
                    generated, stats = euler_sample(wrapper, xt, y, mask, prompts, precision="bf16")
                    actual["complete_sampling_calls"] += 1
                    actual["complete_rollouts"] += len(generated)
                    name = f"benchmark_{arm}_{phase}{repeat}_seed4101"
                    raw_path = output / "raw" / name
                    if raw_path.with_suffix(".npz").exists() or raw_path.with_suffix(".obj").exists():
                        raise FileExistsError(f"Refusing to overwrite benchmark raw output {raw_path}")
                    save_raw(generated[0], raw_path, xt[0], seed=4101,
                             benchmark_scope="complete-sampling timing only; excluded from main evaluation",
                             arm=arm, timing_phase=phase, repeat=repeat,
                             prompt_vertices=prompt["vertices"], prompt_arity=prompt["arity"],
                             prompt_points=prompt["points"], point_primitive=prompt["point_primitive"])
                    entry = dict(arm=arm, phase=phase, repeat=repeat, name=name,
                                 raw_npz=str(raw_path.with_suffix(".npz")), raw_obj=str(raw_path.with_suffix(".obj")), **stats)
                    sampling_records.append(entry)
                    repeat_outputs[arm].append(dict(phase=phase, repeat=repeat, name=name,
                        tokens=generated[0].detach().float().cpu().numpy().copy()))
                    print(json.dumps(dict(stage="sampling_timing", arm=arm, phase=phase, repeat=repeat,
                                          seconds=stats["seconds"], complete_rollouts=actual["complete_rollouts"])), flush=True)
                    delta = stats["call_delta"]
                    if stats["status"] != "PASS" or delta["wrapper_forward"] != 50 or delta["backbone_forward"] != 50 or delta["adapter_layer"] != 200 or delta["condition_encoding"] != 1:
                        raise RuntimeError("Complete-sampling timing failed numerical or actual-call checks")
        torch.cuda.synchronize()
        repeatability = {}
        for arm, outputs in repeat_outputs.items():
            pairs = []
            for left in range(len(outputs)):
                for right in range(left+1, len(outputs)):
                    a, b = outputs[left], outputs[right]
                    difference = a["tokens"].astype(np.float64) - b["tokens"].astype(np.float64)
                    rms = float(np.sqrt(np.mean(difference**2)))
                    pairs.append(dict(left=a["name"], right=b["name"],
                        both_timed=a["phase"] == "timed" and b["phase"] == "timed",
                        bitwise_equal=bool(np.array_equal(a["tokens"], b["tokens"])),
                        max_abs=float(np.max(np.abs(difference))), coordinate_rms=rms,
                        coordinate_rms_bbox_pct=100*rms/source["bbox_diagonal"]))
            timed_pairs = [pair for pair in pairs if pair["both_timed"]]
            repeatability[arm] = dict(sampling_executions=len(outputs), independent_inputs=1,
                all_pairs=pairs, all_pairs_max_abs=_distribution([pair["max_abs"] for pair in pairs]),
                all_pairs_coordinate_rms=_distribution([pair["coordinate_rms"] for pair in pairs]),
                timed_pairs_max_abs=_distribution([pair["max_abs"] for pair in timed_pairs]),
                timed_pairs_coordinate_rms=_distribution([pair["coordinate_rms"] for pair in timed_pairs]),
                all_outputs_bitwise_equal=all(pair["bitwise_equal"] for pair in pairs),
                interpretation="Identical prompt, noise, timestep grid and checkpoint. Differences measure execution/numerical repeat variability, not condition response. Pair values are dependent descriptive comparisons.")
        report["same_input_repeatability"] = repeatability
        hashes_after_sampling = {arm: state_sha256(model) for arm, model in models.items()}
        base_after_sampling = state_sha256(base)
        actual.update(wrapper_counts=dict(wrapper.counts),
                      adapter_calls={arm: model.call_count for arm, model in models.items()},
                      encoder_calls={arm: model.encoder_call_count for arm, model in models.items()},
                      sampling_geometry_helper_calls="NOT_INSTRUMENTED: timing hooks removed")
        sampling_count_ok = (actual["component_backbone_forward"] == 26 and actual["sampling_backbone_forward"] == 400 and
                             actual["backbone_forward"] == 426 and actual["complete_sampling_calls"] == 8 and
                             actual["complete_rollouts"] == 8 and wrapper.counts["wrapper_forward"] == 426 and
                             wrapper.counts["adapter_layer"] == 1704 and wrapper.counts["condition_encoding"] == 60)
        report.update(status="PASS" if sampling_count_ok and base_after_sampling == precheck["base_state_sha256"] and
                      hashes_after_sampling == adapter_hashes_before else "FAIL",
                      fair_sampling_measurements=sampling_records,
                      fair_full_sampling_ms={arm: _distribution([1000*entry["seconds"] for entry in sampling_records
                          if entry["arm"] == arm and entry["phase"] == "timed"]) for arm in models},
                      fair_sampling_configuration=dict(warmup_per_arm=1,timed_per_arm=3,seed=4101,
                          prompt_kind=source["kind"],parent_uid=source["parent_uid"],raw_faces=512,
                          batch=1,precision="bf16",backend="PyTorch SDPA math",NFE=50,
                          face_cfg=1,geometry_cfg=1,production_sampler="sample.py:euler_sample",
                          timing_hooks_removed=True,main_evaluation_rows_added=0),
                      sampling_counter_checks_pass=sampling_count_ok,
                      base_hash_after_sampling=base_after_sampling,
                      base_hash_unchanged_after_sampling=base_after_sampling == precheck["base_state_sha256"],
                      adapter_hashes_after_sampling=hashes_after_sampling,
                      peak_cuda_allocated_bytes=max([components_peak_allocated]+[entry["peak_cuda_allocated"] for entry in sampling_records]),
                      peak_cuda_reserved_bytes=max(components_peak_reserved,torch.cuda.max_memory_reserved()))
        report["measurement_limitations"].append("The fair complete-sampling campaign has only three timed repeats per arm at one fixed prompt/noise; quantiles are descriptive and cannot establish a significant speedup.")
    except BaseException as error:
        import traceback
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        active[0] = None
        adapter_module.closest_points_primitives = original_geometry
        wrapper.encode = original_encode
        for handle in handles:
            handle.remove()
        counter_handle.remove()
        wrapper.close()
        report["elapsed_seconds"] = time.perf_counter()-started
        report["cpu_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "backbone_forwards": actual["backbone_forward"],
                      "optimizer_updates": 0, "sampling_calls": actual["complete_sampling_calls"], "output": str(destination)}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.checkpoint, args.output)
    raise SystemExit(0 if result["status"] == "PASS" else 1)