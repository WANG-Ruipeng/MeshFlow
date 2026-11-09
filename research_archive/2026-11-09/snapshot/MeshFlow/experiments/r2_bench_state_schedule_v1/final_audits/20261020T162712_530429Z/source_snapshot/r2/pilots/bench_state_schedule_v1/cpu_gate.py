"""Small real-data + algebra gate. Zero neural forwards, no CUDA context."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
import io
import sys
import unittest
import numpy as np
from .common import ROOT, OUT, read, resolve, write_new
sys.path.insert(0, str(ROOT / "r2/src"))
from . import async_core as A, perturbed_core as P
from r2.pilots.bench_clean_multipart_seam_v1 import data as D
import torch


def run():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "r2.pilots.bench_state_schedule_v1.test_async_core",
        "r2.pilots.bench_state_schedule_v1.test_perturbed_core"])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(stream.getvalue())
    old = read(ROOT/"experiments/r2_bench_adaptation_strength_v1/training_registration.json")
    manifest = read(old["manifest"])
    schedule = read(old["input_schedule"])["steps"]
    cases = {r["condition_id"]: r for r in manifest["train"]}
    rows = [r for e in schedule[:4] for r in e["samples"]]
    actual, eligibility, corruption, active = [], 0, 0, 0
    for r in rows:
        original = D.sample_for_training(D.load_case(cases[r["condition_id"]]), r)
        new, audit = P.apply_perturbation(original, r)
        asynchronous = A.transform_sample(original)
        fixed = original["anchor_ids"] >= 0
        for s in (new, asynchronous):
            if not np.array_equal(s["xt"][fixed], original["xt"][fixed]):
                raise AssertionError("Real-data anchor changed")
            if not np.array_equal(s["x1"], original["x1"]):
                raise AssertionError("Real-data target changed")
        for k in ("x0", "x1", "anchor_ids", "known_mask", "valid_mask", "source_face_ids"):
            if not np.array_equal(asynchronous[k], original[k]) or not np.array_equal(new[k], original[k]):
                raise AssertionError("Frozen identity changed: "+k)
        actual.append(dict(condition_id=r["condition_id"], t=float(original["t"]),
            N=int(original["N"]), K=int(original["K"]), corruption_applied=audit["applied"],
            active_tip_coordinates=int(asynchronous["async_active_tip_scalar_count"])))
    # Full scheduling counts require no OT or model and don't alter the row stream.
    for entry in schedule:
        for row in entry["samples"]:
            t = float(np.float32(1./(1.+np.exp(-float(np.random.RandomState(int(row["time_seed"])).standard_normal())))))
            eligible = float(np.float32(.3)) <= t <= float(np.float32(.9))
            eligibility += eligible
            corruption += eligible and P.derive_row_randomness(row)["selected_by_probability"]
            active += t < float(np.float32(.7))
    if torch.cuda.is_initialized():
        raise AssertionError("CPU gate initialized CUDA")
    value = dict(status="PASS", tests_run=result.testsRun, unittest_output=stream.getvalue(),
        actual_data_fixtures=actual, full_stream_examples=12000,
        eligible_corruption_examples=eligibility, selected_corruption_examples=corruption,
        async_active_clock_examples=active, neural_forwards=0, backward=0,
        optimizer_updates=0, CUDA_initialized=False)
    write_new(OUT/"cpu_gate.json", value)
    return value


if __name__ == "__main__":
    print(run())
