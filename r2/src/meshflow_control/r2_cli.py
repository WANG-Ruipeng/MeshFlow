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
