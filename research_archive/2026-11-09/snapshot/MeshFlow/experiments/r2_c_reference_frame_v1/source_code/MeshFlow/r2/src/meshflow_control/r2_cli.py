"""Lazy maintained R2 commands; help and recipe inspection need no torch or assets."""
import argparse
import json


def build_parser():
    parser = argparse.ArgumentParser(prog="meshflow-r2",
                                     description="R2: full-course balanced C20/C40 + HYBRID FM")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("recipe", help="Print the fixed recipe and its current data/resume limits")
    p = sub.add_parser("inspect", help="Validate an R2 checkpoint on CPU, with zero forwards")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--expected-sha256")
    for name in ("check-assets", "prepare-run", "train"):
        p = sub.add_parser(name)
        p.add_argument("--data-manifest", required=True)
        p.add_argument("--stream-root", required=True)
        p.add_argument("--official", required=name != "check-assets")
        if name != "check-assets":
            p.add_argument("--out", required=True)
            p.add_argument("--registration", required=True)
        if name == "train":
            p.add_argument("--attempt-id", required=True)
            p.add_argument("--resume")
    p = sub.add_parser("sample", help="One explicit R2 Euler50 sample; no automatic postprocessing")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--expected-sha256")
    p.add_argument("--condition", required=True)
    p.add_argument("--condition-key", default="C")
    p.add_argument("--num-faces", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True)
    p = sub.add_parser("prepare-condition", help="Capture user C source order before canonicalization; CPU only")
    p.add_argument("--input", required=True)
    p.add_argument("--input-key", default="C", help="Use indexed for NPZ vertices/faces")
    p.add_argument("--vertex-ids-key")
    p.add_argument("--permutation-key")
    p.add_argument("--face-ids-key")
    p.add_argument("--out", required=True)
    p = sub.add_parser("orient", help="Restore source C and orient free in an existing GEO_C NPZ; no model/GEO")
    p.add_argument("--input", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("denoise", help="One official D0 (or verified cached proposal), GEO_C and source-oriented export")
    p.add_argument("--input", required=True)
    p.add_argument("--input-key", default="raw")
    p.add_argument("--known-faces", required=True, type=int)
    p.add_argument("--condition")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--proposal")
    p.add_argument("--proposal-receipt")
    p.add_argument("--coordination", choices=("none", "rbf"), default="none")
    p.add_argument("--reference-length", type=float)
    p.add_argument("--winding", choices=("source", "canonical"), default="source")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", required=True)
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    command = args.pop("command")
    if command == "recipe":
        from .r2_spec import describe
        result = describe()
    elif command == "inspect":
        from .r2_checkpoints import inspect_checkpoint
        args["path"] = args.pop("checkpoint")
        result = inspect_checkpoint(**args)
    elif command in {"prepare-condition", "orient", "denoise"}:
        from . import denoise
        function = {"prepare-condition": denoise.prepare_condition, "orient": denoise.orient, "denoise": denoise.run}[command]
        result = function(**args)
    elif command == "sample":
        from .r2_sampling import run
        result = run(**args)
    else:
        from . import r2_runs
        function = {"check-assets": r2_runs.check_assets,
                    "prepare-run": r2_runs.prepare_run, "train": r2_runs.train}[command]
        result = function(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if command == "train" and result.get("status") != "COMPLETE":
        return 75
    return 0
