"""Fixed synthetic checks for the independent Duffy reference; no real model."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from .reference import (OUT, EPSILONS, FEATURE_CONSISTENCY, ReferenceBudget,
    check_frozen, feature_errors, integrate_triangle, write_json)


def run_synthetic(out=OUT):
    out = Path(out)
    protocol = check_frozen(out)
    budget = ReferenceBudget(out/"synthetic_ledger.jsonl", **protocol["synthetic_budget"])
    records, checks = [], []

    def evaluate(label, C, x, t, eps=1e-10):
        result = integrate_triangle(C, x, t, eps=eps, budget=budget, label=label)
        records.append(dict(label=label, C=np.asarray(C).tolist(), x=np.asarray(x).tolist(),
                            t=t, result=result))
        write_json(out/"synthetic_partial.json", dict(records=records, checks=checks))
        return result

    def compare(name, condition, detail):
        checks.append(dict(name=name, pass_=bool(condition), detail=detail))

    C = np.array([[0., 0., 0.], [1.5, 0., 0.], [.2, .8, 0.]])
    x = np.array([.7, -.02, .05])
    permutation_results = []
    for permutation in itertools.permutations(range(3)):
        result = evaluate("permutation_"+"".join(map(str, permutation)), C[list(permutation)], x, 1.)
        permutation_results.append(result)
    base = permutation_results[0]
    for i, result in enumerate(permutation_results):
        errors = feature_errors(base, result) if "mu" in base and "mu" in result else None
        compare("permutation_"+str(i), errors is not None and max(errors.values()) < 1e-8, errors)

    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    offset = np.array([3., -2., .1])
    rotated = evaluate("rigid_transform", C@rotation.T+offset, x@rotation.T+offset, 1.)
    if "mu" in rotated and "mu" in base:
        expected = dict(mu=np.asarray(base["mu"])@rotation.T+offset,
            Sigma=rotation@np.asarray(base["Sigma"])@rotation.T,
            compressed_mass=base["compressed_mass"])
        errors = feature_errors(rotated, expected)
    else:
        errors = None
    compare("rigid_transform", errors is not None and max(errors.values()) < 1e-8, errors)

    tiny = np.array([[0., 0., 0.], [1e-4, 0., 0.], [0., 1e-4, 0.]])
    centroid = tiny.mean(axis=0)
    uniform = evaluate("uniform_small_triangle", tiny, centroid, 0.)
    uniform_cov = (tiny-centroid).T@(tiny-centroid)/12
    mean_error = float(np.linalg.norm(np.asarray(uniform["mu"])-centroid)) if "mu" in uniform else 1.
    cov_error = float(np.linalg.norm(np.asarray(uniform["Sigma"])-uniform_cov)) if "Sigma" in uniform else 1.
    compare("uniform_limit", mean_error < 1e-12 and cov_error < 1e-14,
        dict(mean_error=mean_error, covariance_error=cov_error))

    cases = [
        ("near_corner", C, [-.001, -.001, .1]),
        ("edge", C, [.7, 0., .1]),
        ("far", C, [20., -20., 20.]),
        ("slender", [[0., 0., 0.], [3., 0., 0.], [.5, 1e-7, 0.]], [1., .2, .2]),
    ]
    for label, triangle, query in cases:
        pair = [evaluate(label+"_"+str(eps), triangle, query, 1., eps) for eps in EPSILONS]
        errors = feature_errors(*pair) if all("mu" in v for v in pair) else None
        compare(label+"_two_tolerances", errors is not None and max(errors.values()) < FEATURE_CONSISTENCY, errors)
        for i, result in enumerate(pair):
            finite = "mu" in result and np.isfinite(np.r_[result["mu"],
                np.asarray(result["Sigma"]).ravel(), result["z"]]).all()
            compare(label+"_finite_"+str(i), finite, result.get("error"))
            compare(label+"_covariance_psd_"+str(i),
                result.get("covariance_minimum_eigenvalue", -1.) >= -1e-10,
                result.get("covariance_minimum_eigenvalue"))

    point = np.array([.2, .3, .4])
    query = np.array([20., -20., 20.])
    collapsed = evaluate("collapsed_exact", np.repeat(point[None], 3, axis=0), query, 1.)
    expected_z = -float(np.sum((point-query)**2))/(2*.1**2)
    collapsed_errors = dict(mu=float(np.linalg.norm(np.asarray(collapsed["mu"])-point)),
        covariance=float(np.linalg.norm(collapsed["Sigma"])), z=abs(collapsed["z"]-expected_z)) if "mu" in collapsed else None
    compare("collapsed_exact", collapsed_errors is not None and max(collapsed_errors.values()) < 1e-8, collapsed_errors)
    collinear = evaluate("collinear_finite", [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], [.5, .2, .1], 1.)
    compare("collinear_finite", "mu" in collinear and np.isfinite(collinear["z"]), collinear.get("error"))
    for record in records:
        compare("integration_"+record["label"], record["result"]["status"] == "PASS", record["result"]["status"])
    result = dict(status="PASS" if all(c["pass_"] for c in checks) else "BLOCKED_SYNTHETIC",
        checks=checks, records=records, actual_reference_calls=budget.calls,
        actual_kernel_points=budget.points, actual_model_calls=0, actual_OT_calls=0,
        elapsed_seconds=time.perf_counter()-budget.started,
        error_estimates_are_rigorous=False)
    check_frozen(out)
    write_json(out/"synthetic_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    result = run_synthetic(args.out)
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "records")}))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
