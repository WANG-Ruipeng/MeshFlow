"""Bounded, independent CPU reference for eight frozen triangle/query pairs.

The reference uses adaptive Gauss--Kronrod integration of a Duffy square,
not a uniform triangular grid. Error estimates are not rigorous certificates.
No model, CUDA, RNG, optimizer, or sampling operation belongs in this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import scipy
from scipy.integrate import quad_vec


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "native_t1/runs/chair_surface_attention_v1"
OUT = ROOT / "native_t1/runs/chair_surface_attention_v1r1/reference"
METRICS = ("mu_L2", "covariance_Frobenius", "compressed_mass_abs")
PAIR_SPECS = (
    ("worst_mu", "ordinary_40", 3, 4, 60),
    ("worst_covariance", "ordinary_40", 3, 5, 70),
    ("worst_mass", "ordinary_40", 3, 2, 59),
    ("unconverged_dense", "ordinary_40", 3, 0, 41),
    ("control_ordinary20", "ordinary_20", 0, 0, 0),
    ("control_maximum20", "maximum_20", 1, 0, 0),
    ("control_ordinary40", "ordinary_40", 0, 0, 0),
    ("control_maximum40", "maximum_40", 1, 0, 0),
)
EPSILONS = (1e-8, 1e-10)
FEATURE_CONSISTENCY = 1e-5
BREAK_MULTIPLIERS = (.125, .5, 2., 8.)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_event(path, event):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(time_unix=time.time(), **event),
                                allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class IntegrationBudgetExceeded(RuntimeError):
    pass


class ReferenceBudget:
    """All actual kernel evaluations, including failed attempts, are counted."""
    def __init__(self, ledger, *, max_calls, max_total_points, total_seconds,
                 max_points_per_call=1_000_000, seconds_per_call=120.):
        self.ledger = Path(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        if self.ledger.exists():
            raise FileExistsError("Refusing to overwrite or silently replay reference ledger")
        self.max_calls = max_calls
        self.max_total_points = max_total_points
        self.total_seconds = total_seconds
        self.max_points_per_call = max_points_per_call
        self.seconds_per_call = seconds_per_call
        self.started = time.perf_counter()
        self.calls = 0
        self.points = 0
        self.active = None
        append_event(self.ledger, dict(event="BUDGET_OPEN", max_calls=max_calls,
            max_total_points=max_total_points, total_seconds=total_seconds,
            max_points_per_call=max_points_per_call, seconds_per_call=seconds_per_call))

    def begin(self, label, eps):
        if self.active is not None:
            raise RuntimeError("Concurrent/nested reference calls are forbidden")
        if self.calls >= self.max_calls or self.points >= self.max_total_points:
            raise IntegrationBudgetExceeded("Global call/point allowance exhausted")
        if time.perf_counter()-self.started >= self.total_seconds:
            raise IntegrationBudgetExceeded("Global wall allowance exhausted")
        self.calls += 1
        self.active = dict(label=label, eps=eps, call_index=self.calls,
                           points=0, start=time.perf_counter())
        append_event(self.ledger, dict(event="REFERENCE_BEGIN", label=label,
            eps=eps, call_index=self.calls, total_points=self.points))

    def tick(self):
        active = self.active
        if active is None:
            raise RuntimeError("Kernel outside registered reference call")
        if active["points"] >= self.max_points_per_call or self.points >= self.max_total_points:
            raise IntegrationBudgetExceeded("Kernel point allowance exhausted")
        # Count only evaluations which will actually execute.
        if active["points"] % 256 == 0:
            now = time.perf_counter()
            if now-active["start"] >= self.seconds_per_call or now-self.started >= self.total_seconds:
                raise IntegrationBudgetExceeded("Reference wall allowance exhausted")
        active["points"] += 1
        self.points += 1

    def end(self, status, error=None):
        active = self.active
        result = dict(label=active["label"], eps=active["eps"],
            call_index=active["call_index"], kernel_points=active["points"],
            total_kernel_points=self.points,
            elapsed_seconds=time.perf_counter()-active["start"], status=status)
        if error is not None:
            result["error"] = error
        append_event(self.ledger, dict(event="REFERENCE_RETURN" if status == "PASS"
            else "REFERENCE_FAILED", **result))
        self.active = None
        return result


def closest_on_triangle(C, x):
    """Exact linear algebra/segment minimization; also supports rank collapse."""
    candidates = []
    basis = np.column_stack((C[1]-C[0], C[2]-C[0]))
    solution, _, rank, _ = np.linalg.lstsq(basis, x-C[0], rcond=None)
    bary = np.array([1-solution.sum(), *solution])
    if rank == 2 and np.all(bary >= 0):
        candidates.append((bary @ C, bary))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = C[j]-C[i]
        norm2 = float(edge @ edge)
        u = float(np.clip((x-C[i]) @ edge / norm2, 0., 1.)) if norm2 else 0.
        weights = np.zeros(3)
        weights[i], weights[j] = 1-u, u
        candidates.append((weights @ C, weights))
    p, bary = min(candidates, key=lambda value: float(np.sum((value[0]-x)**2)))
    return p, bary


def breakpoints(center, scale):
    values = {0., 1., float(np.clip(center, 0., 1.))}
    for multiplier in BREAK_MULTIPLIERS:
        values.add(float(np.clip(center-multiplier*scale, 0., 1.)))
        values.add(float(np.clip(center+multiplier*scale, 0., 1.)))
    return sorted(values)[1:-1]


def feature_errors(first, second):
    return dict(mu_L2=float(np.linalg.norm(np.asarray(first["mu"])-second["mu"])),
        covariance_Frobenius=float(np.linalg.norm(np.asarray(first["Sigma"])-second["Sigma"])),
        compressed_mass_abs=abs(float(first["compressed_mass"])-float(second["compressed_mass"])))


def integrate_triangle(C, x, t, *, eps, budget, label):
    """Integrate normalized surface moments by independent adaptive Duffy rule.

    y=A+r(C-A)+(1-r)w(B-A), normalized measure 2(1-r) dr dw.
    Factoring exp(-distance(x,T)^2/(2s^2)) prevents far-query underflow.
    Centering moments at the closest point prevents raw-coordinate cancellation.
    Geometric split locations are fixed by inputs, never by observed results.
    """
    C = np.asarray(C, dtype=np.float64).reshape(3, 3)
    x = np.asarray(x, dtype=np.float64).reshape(3)
    if not np.isfinite(C).all() or not np.isfinite(x).all() or not 0 <= t <= 1:
        raise ValueError("Finite coordinates and t in [0,1] are required")
    if eps not in EPSILONS:
        raise ValueError("Only the two registered tolerances are allowed")
    budget.begin(label, eps)
    try:
        # Input-only canonical order makes the six vertex permutations identical.
        C = C[np.lexsort((C[:, 2], C[:, 1], C[:, 0]))]
        # Put the longest edge at r=0. Ties follow the canonical pair order.
        i, j = max(((0, 1), (0, 2), (1, 2)),
                   key=lambda ij: float(np.sum((C[ij[1]]-C[ij[0]])**2)))
        k = next(k for k in range(3) if k not in (i, j))
        C = C[[i, j, k]]
        A, B, D = C
        BA, DA = B-A, D-A
        scale = max(float(np.linalg.norm(BA)), np.finfo(float).tiny)
        # A collapsed triangle needs no artificial huge or zero coordinate scale.
        if not np.any(C != C[0]):
            scale = 1.
        s2 = (1-float(t))**2 + .1**2
        s = math.sqrt(s2)
        closest, closest_bary = closest_on_triangle(C, x)
        residual = closest-x
        distance = float(np.linalg.norm(residual))
        log_peak = -float(residual @ residual)/(2*s2)
        r_star = float(closest_bary[2])
        r_width = min(1., s/(scale*(1+distance/s)))
        r_points = breakpoints(r_star, r_width)
        inner_info = []
        exponent_max = -math.inf
        abs_tolerance = eps*1e-6

        def outer(r):
            nonlocal exponent_max
            start = A+r*DA
            direction = (1-r)*BA
            length2 = float(direction @ direction)
            if length2 > 0:
                raw_center = float((x-start) @ direction/length2)
                w_star = float(np.clip(raw_center, 0., 1.))
                excess = abs(raw_center-w_star)*math.sqrt(length2)
                w_width = min(1., s/(math.sqrt(length2)*(1+excess/s)))
            else:
                w_star, w_width = .5, 1.
            w_points = breakpoints(w_star, w_width)

            def kernel(w):
                nonlocal exponent_max
                budget.tick()
                delta = start+w*direction-closest
                exponent = -(float(delta @ delta)+2*float(delta @ residual))/(2*s2)
                exponent_max = max(exponent_max, exponent)
                if exponent > 1e-7:
                    raise ArithmeticError("Closest-point exponent shift is not a maximum")
                weight = 2*(1-r)*math.exp(exponent)
                a, b, c = delta/scale
                return weight*np.array([1., a, b, c, a*a, b*b, c*c, a*b, a*c, b*c])

            value, error, info = quad_vec(kernel, 0., 1., epsabs=abs_tolerance,
                epsrel=eps, norm="max", limit=512, points=w_points,
                quadrature="gk21", full_output=True, workers=1)
            inner_info.append(dict(error_estimate=float(error), neval=int(info.neval),
                intervals=len(info.intervals), success=bool(info.success), status=int(info.status)))
            return value

        integral, outer_error, info = quad_vec(outer, 0., 1., epsabs=abs_tolerance,
            epsrel=eps, norm="max", limit=512, points=r_points,
            quadrature="gk21", full_output=True, workers=1)
        mass = float(integral[0])
        if not math.isfinite(mass) or mass <= 0:
            raise ArithmeticError("Shifted integrated mass is not positive finite")
        normalized = integral/mass
        mean_delta = normalized[1:4]
        raw_cov = np.array([[normalized[4], normalized[7], normalized[8]],
                            [normalized[7], normalized[5], normalized[9]],
                            [normalized[8], normalized[9], normalized[6]]])
        covariance = scale**2*(raw_cov-np.outer(mean_delta, mean_delta))
        mean = closest+scale*mean_delta
        log_mass = log_peak+math.log(mass)
        compressed = math.log1p(max(-log_mass, 0.))
        worst_inner_error = max((v["error_estimate"] for v in inner_info), default=0.)
        # This is a heuristic propagated estimate: the inner maximum is sampled,
        # and QUADPACK/GK estimates themselves are not certified error bounds.
        estimated_integral_error = float(outer_error)+worst_inner_error
        relative_estimate = estimated_integral_error/mass
        if estimated_integral_error < mass:
            denominator = mass-estimated_integral_error
            max_first = float(np.max(np.abs(mean_delta)))
            max_second = float(np.max(np.abs(raw_cov)))
            first_est = estimated_integral_error*(1+max_first)/denominator
            second_est = estimated_integral_error*(1+max_second)/denominator
            estimates = dict(mu_L2=math.sqrt(3)*scale*first_est,
                covariance_Frobenius=3*scale**2*(second_est+2*max_first*first_est+first_est**2),
                compressed_mass_abs=-math.log1p(-relative_estimate))
        else:
            estimates = {key: None for key in METRICS}
        success = bool(info.success) and all(v["success"] for v in inner_info)
        result = dict(status="PASS" if success else "BLOCKED_ADAPTIVE_INTEGRATION",
            method="DUFFY_ADAPTIVE_GK21_FP64_V1", epsrel=eps, epsabs=abs_tolerance,
            mu=mean.tolist(), Sigma=covariance.tolist(), z=log_mass,
            compressed_mass=compressed, closest_point=closest.tolist(),
            normalized_shifted_mass=mass, log_peak=log_peak, moment_scale=scale,
            r_star=r_star, r_breakpoints=r_points, r_width=r_width,
            maximum_shifted_exponent=exponent_max,
            covariance_minimum_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),
            outer_error_estimate=float(outer_error),
            maximum_observed_inner_error_estimate=worst_inner_error,
            combined_integral_error_estimate=estimated_integral_error,
            combined_relative_mass_error_estimate=relative_estimate,
            propagated_feature_error_estimates=estimates,
            error_estimate_is_rigorous=False,
            error_estimate_caveat="SciPy GK estimates plus maximum sampled inner error; not a rigorous certificate.",
            outer_neval=int(info.neval), outer_intervals=len(info.intervals),
            outer_success=bool(info.success), outer_message=str(info.message),
            inner_calls=len(inner_info),
            inner_neval_total=sum(v["neval"] for v in inner_info),
            inner_failed_calls=sum(not v["success"] for v in inner_info),
            inner_max_intervals=max((v["intervals"] for v in inner_info), default=0))
        result["accounting"] = budget.end(result["status"])
        return result
    except BaseException as exc:
        accounting = budget.end("BLOCKED", repr(exc))
        return dict(status="BLOCKED", error=repr(exc), traceback=traceback.format_exc(),
                    accounting=accounting, epsrel=eps)


def register_protocol(out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out/"protocol.json"
    if protocol_path.exists():
        raise FileExistsError("Protocol already exists; never overwrite a registration")
    manifest_path = OLD/"registration/geometry_fixtures.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures = {f["fixture_id"]: f for f in manifest["fixtures"]}
    csv_path = OLD/"tests/relations/relations_real_pair_errors.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    pairs = []
    for pair_id, fixture_id, time_index, query_index, known_index in PAIR_SPECS:
        fixture = fixtures[fixture_id]
        npz_path = Path(fixture["npz_path"])
        assert sha256(npz_path) == fixture["npz_sha256"]
        with np.load(npz_path, allow_pickle=False) as data:
            t = float(data["times"][time_index])
            C = data["C"][known_index].astype(np.float64)
            x = data["queries"][time_index, query_index].astype(np.float64)
        matched = [row for row in rows if row["fixture_id"] == fixture_id
            and int(row["query_index"]) == query_index and int(row["known_index"]) == known_index
            and abs(float(row["t"])-t) < 1e-8]
        assert len(matched) == 1
        pairs.append(dict(pair_id=pair_id, fixture_id=fixture_id, time_index=time_index,
            query_index=query_index, known_index=known_index, t=t, C=C.tolist(), x=x.tolist(),
            npz_path=str(npz_path), npz_sha256=sha256(npz_path),
            source_face_id=fixture["known_source_ids"][known_index],
            saved_n8_vs_n64_errors={key: float(matched[0][key]) for key in METRICS}))
    source_files = [Path(__file__), Path(__file__).with_name("reference_tests.py")]
    protocol = dict(schema="surface_attention_r1_independent_reference_v1", status="FROZEN",
        method="DUFFY_ADAPTIVE_GK21_FP64_V1", pairs=pairs,
        fixed_pair_selection="Three prior worst ordinary components, one prior failed dense-reference pair, and four fixed t<=.5 controls; no result-dependent additions.",
        epsrel=list(EPSILONS), epsabs_rule="epsrel * 1e-6", norm="max", quad_limit=512,
        quadrature="gk21", split_multipliers=list(BREAK_MULTIPLIERS),
        geometry_splits="Closest-point barycentric r and conditional segment projection w; width min(1,s/(edge_scale*(1+distance/s))).",
        feature_consistency_each_metric_lt=FEATURE_CONSISTENCY,
        original_accuracy_each_metric_lt=.01, original_reference_each_metric_lt=.001,
        real_budget=dict(max_calls=16, max_total_points=16_000_000, total_seconds=1920,
            max_points_per_call=1_000_000, seconds_per_call=120),
        synthetic_budget=dict(max_calls=32, max_total_points=4_000_000, total_seconds=300,
            max_points_per_call=1_000_000, seconds_per_call=120),
        exact_old_relation_call_budget=24, old_relation_n=[8, 64, 128],
        old_relation_precision=["FP32", "FP64", "FP64"], model_calls=0,
        cpu_threads=2, workers=1, error_claim="Estimated integration error and two-tolerance agreement only, never rigorous certification.",
        source_sha256={str(p): sha256(p) for p in source_files},
        protected_input_sha256={str(p): sha256(p) for p in [manifest_path, csv_path,
            OLD/"numerical_failure_analysis.json",
            ROOT/"experiments/mf_h1_local0/chair_surface_attention_v1/relations.py"]})
    write_json(protocol_path, protocol)
    write_json(out/"protocol.sha256.json", dict(protocol_sha256=sha256(protocol_path),
        source_sha256=protocol["source_sha256"]))
    return protocol


def check_frozen(out):
    out = Path(out)
    protocol = json.loads((out/"protocol.json").read_text())
    record = json.loads((out/"protocol.sha256.json").read_text())
    assert sha256(out/"protocol.json") == record["protocol_sha256"]
    for group in ("source_sha256", "protected_input_sha256"):
        for path, expected in protocol[group].items():
            assert sha256(path) == expected, path
    for pair in protocol["pairs"]:
        assert sha256(pair["npz_path"]) == pair["npz_sha256"]
    return protocol


def compute_old_rules(pair, ledger, counts):
    # Import the frozen CPU relation module only here; no model loading.
    import torch
    from experiments.mf_h1_local0.chair_surface_attention_v1.relations import triangle_statistics
    torch.set_num_threads(2)
    results = {}
    for n in (8, 64, 128):
        if counts["calls"] >= 24:
            raise RuntimeError("Old-rule CPU call budget exhausted")
        counts["calls"] += 1
        append_event(ledger, dict(event="OLD_RELATION_BEGIN", call_index=counts["calls"],
            pair_id=pair["pair_id"], n=n, model_call=False))
        try:
            dtype = torch.float32 if n == 8 else torch.float64
            with torch.no_grad():
                value = triangle_statistics(torch.tensor([pair["x"]], dtype=dtype),
                    torch.tensor([pair["C"]], dtype=dtype), pair["t"], "SURFACE_XY",
                    n=n, reference=n != 8)
            z = float(value["z"][0, 0])
            results[str(n)] = dict(mu=value["mu"][0, 0].double().tolist(),
                Sigma=value["Sigma"][0, 0].double().tolist(), z=z,
                compressed_mass=math.log1p(max(-z, 0.)))
            append_event(ledger, dict(event="OLD_RELATION_RETURN", call_index=counts["calls"],
                pair_id=pair["pair_id"], n=n, model_call=False))
        except BaseException as exc:
            append_event(ledger, dict(event="OLD_RELATION_FAILED", call_index=counts["calls"],
                pair_id=pair["pair_id"], n=n, error=repr(exc), model_call=False))
            raise
    return results


def run_real(out=OUT):
    out = Path(out)
    protocol = check_frozen(out)
    budget = ReferenceBudget(out/"reference_ledger.jsonl", **protocol["real_budget"])
    old_ledger = out/"old_rule_cpu_ledger.jsonl"
    if old_ledger.exists():
        raise FileExistsError("Old rule ledger already exists")
    results, comparisons = [], []
    old_counts = dict(calls=0)
    for pair in protocol["pairs"]:
        entry = dict(pair=pair, references=[])
        for eps in EPSILONS:
            value = integrate_triangle(pair["C"], pair["x"], pair["t"],
                eps=eps, budget=budget, label=pair["pair_id"])
            entry["references"].append(value)
            write_json(out/"pairs"/(pair["pair_id"]+".json"), entry)
        entry["old_rules"] = compute_old_rules(pair, old_ledger, old_counts)
        saved = pair["saved_n8_vs_n64_errors"]
        recomputed = feature_errors(entry["old_rules"]["8"], entry["old_rules"]["64"])
        entry["saved_error_reproduction_absolute_difference"] = {
            key: abs(saved[key]-recomputed[key]) for key in METRICS}
        if all(r["status"] == "PASS" for r in entry["references"]):
            a, b = entry["references"]
            agreement = feature_errors(a, b)
            entry["two_tolerance_errors"] = agreement
            entry["reference_self_consistent"] = all(v < FEATURE_CONSISTENCY for v in agreement.values())
            for n in (8, 64, 128):
                error = feature_errors(entry["old_rules"][str(n)], b)
                comparisons.append(dict(pair_id=pair["pair_id"], fixture_id=pair["fixture_id"],
                    t=pair["t"], query_index=pair["query_index"], known_index=pair["known_index"],
                    rule_n=n, rule_points=3*n*n, **error,
                    reference_self_consistent=entry["reference_self_consistent"],
                    original_accuracy_pass=all(v < .01 for v in error.values()),
                    original_reference_tolerance_pass=all(v < .001 for v in error.values())))
        else:
            entry["reference_self_consistent"] = False
        results.append(entry)
        write_json(out/"pairs"/(pair["pair_id"]+".json"), entry)
        print(json.dumps(dict(pair=pair["pair_id"],
            reference_self_consistent=entry["reference_self_consistent"],
            total_kernel_points=budget.points)), flush=True)
    if comparisons:
        with (out/"reference_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    summary = dict(status="PASS_REFERENCE_SELF_CONSISTENCY" if all(r["reference_self_consistent"]
        for r in results) else "BLOCKED_REFERENCE_CONVERGENCE", actual_reference_calls=budget.calls,
        actual_kernel_points=budget.points, actual_old_relation_cpu_calls=old_counts["calls"],
        actual_model_calls=0, actual_OT_calls=0, actual_sampling=0,
        elapsed_seconds=time.perf_counter()-budget.started, scipy_version=scipy.__version__,
        error_estimates_are_rigorous=False, pairs=results, comparisons=comparisons)
    check_frozen(out)
    write_json(out/"reference_result.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.register == args.run:
        parser.error("Choose exactly one of --register and --run")
    if args.register:
        protocol = register_protocol(args.out)
        print(json.dumps(dict(status=protocol["status"], pairs=len(protocol["pairs"]),
            protocol_sha256=sha256(args.out/"protocol.json"))))
    else:
        result = run_real(args.out)
        print(json.dumps({k: v for k, v in result.items() if k not in ("pairs", "comparisons")}))
        raise SystemExit(0 if result["status"].startswith("PASS") else 2)


if __name__ == "__main__":
    main()
