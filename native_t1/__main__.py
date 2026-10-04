"""Explicit sample/verify commands; importing this module does not run a model."""
from .runtime import configure_stable_runtime
import argparse
import sys
from pathlib import Path
import numpy as np
from .artifacts import atomic_json, file_sha256, state_sha256
from .checkpoint import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_t1
from .baseline import BASELINE_CHECKPOINT, load_baseline
from .sampling import make_noise, clamped_sample, new_counts


def build_parser():
    parser = argparse.ArgumentParser(description="Native T1 / Geo; Chair START N128..256 and explicit legacy N112 profiles")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="Generate from FP32 known faces only")
    sample.add_argument("--condition", type=Path, required=True, help="NPY C or NPZ containing C; no target needed")
    sample.add_argument("--condition-key", help="Required when condition is NPZ")
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--out", type=Path, required=True, help="New output directory; never overwritten")
    sample.add_argument("--profile", choices=("chair-hybrid", "fm-geo", "edge5", "a-continue", "original-t1", "trained", "trained-geo"), default="chair-hybrid",
                        help="Default: Chair HYBRID START total5000; requires --num-faces 128..256")
    sample.add_argument("--num-faces", type=int, help="Required total N128..256 for chair-hybrid; legacy profiles use N112")
    sample.add_argument("--checkpoint", type=Path,
                        help="Explicit checkpoint path; required for trained/trained-geo, optional for named local profiles")
    sample.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify = commands.add_parser("verify-historical", aliases=["verify"],
                                 help="Historical original-T1 step500 replay (2 rollouts); not the working baseline")
    verify.add_argument("--out", type=Path, required=True)
    commands.add_parser("prepare", add_help=False, help="Prepare the registered task from external official data")
    commands.add_parser("train-edge5", add_help=False, help="Optional JEdge5 training recipe, explicit budget or zero-update preflight")
    commands.add_parser("train", add_help=False, help="FM baseline or optional loss training with an explicit update budget or zero-update preflight")
    return parser


def resolve_profile(args):
    if args.profile == "chair-hybrid":
        from .working_model import load_chair_start, CHAIR_START_CHECKPOINT
        return load_chair_start, args.checkpoint or CHAIR_START_CHECKPOINT
    if args.profile == "fm-geo":
        from .working_model import load_fm_geo, FM_GEO_CHECKPOINT
        return load_fm_geo, args.checkpoint or FM_GEO_CHECKPOINT
    if args.profile == "edge5":
        from .working_model import load_edge5, EDGE5_CHECKPOINT
        return load_edge5, args.checkpoint or EDGE5_CHECKPOINT
    if args.profile == "a-continue":
        return load_baseline, args.checkpoint or BASELINE_CHECKPOINT
    if args.profile == "original-t1":
        return load_t1, args.checkpoint or DEFAULT_CHECKPOINT
    if args.profile == "trained":
        if args.checkpoint is None:
            raise ValueError("--profile trained requires --checkpoint")
        from .portable_checkpoint import load_trained_checkpoint
        return load_trained_checkpoint, args.checkpoint
    if args.profile == "trained-geo":
        if args.checkpoint is None:
            raise ValueError("--profile trained-geo requires --checkpoint")
        from .geometry_checkpoint import load_geometry_checkpoint
        return load_geometry_checkpoint, args.checkpoint
    raise ValueError("Unknown checkpoint profile")


def validate_sample_num_faces(profile, num_faces=None):
    if profile == "chair-hybrid":
        if type(num_faces) is not int or not 128 <= num_faces <= 256:
            raise ValueError("chair-hybrid requires explicit --num-faces in 128..256")
        return num_faces
    if num_faces is not None and (type(num_faces) is not int or num_faces != 112):
        raise ValueError("Legacy profiles require N112")
    return 112


def validate_sample_condition(C, profile, num_faces=None):
    N = validate_sample_num_faces(profile, num_faces)
    if C.dtype != np.float32 or C.ndim not in (2, 3) or C.shape[1:] not in ((9,), (3, 3)):
        raise ValueError("Condition must be FP32[K,9] or FP32[K,3,3]; no implicit coordinate transform")
    if not np.isfinite(C).all():
        raise ValueError("Finite known coordinates required")
    if profile == "chair-hybrid":
        if not 0 < len(C) < N:
            raise ValueError("chair-hybrid requires 0 < K < total N")
    elif len(C) not in (2, 4, 8, 12):
        raise ValueError("Legacy profiles require K2/4/8/12")
    return N


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prepare":
        from .prepare import main as prepare_main
        return prepare_main(argv[1:])
    if argv and argv[0] == "train-edge5":
        from .train_cli import main as train_main
        return train_main(argv[1:], recipe="edge5")
    if argv and argv[0] == "train":
        from .train_cli import main as train_main
        return train_main(argv[1:])
    args = build_parser().parse_args(argv)
    if args.command in ("verify", "verify-historical"):
        from .verify import run_regression
        return run_regression(args.out)
    validate_sample_num_faces(args.profile, args.num_faces)
    if args.profile == "chair-hybrid" and not 0 <= args.seed < 2**32:
        raise ValueError("START seed must be in [0, 2**32)")
    if args.out.exists():
        raise FileExistsError(args.out)
    value = np.load(args.condition, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        try:
            if not args.condition_key:
                raise ValueError("NPZ condition requires an explicit --condition-key")
            C = value[args.condition_key].copy()
        finally:
            value.close()
    else:
        C = value
    N = validate_sample_condition(C, args.profile, args.num_faces)
    args.out.mkdir(parents=True)
    report = dict(status="RUNNING", profile=args.profile, seed=args.seed, condition_file=str(args.condition.resolve()),
                  condition_key=args.condition_key, total_faces=N,
                  condition_file_sha256=file_sha256(args.condition), counts=new_counts())
    atomic_json(args.out / "run.json", report)
    try:
        import torch
        configure_stable_runtime()
        loader, checkpoint = resolve_profile(args)
        model, report["load"] = loader(checkpoint, args.config)
        if args.profile == "chair-hybrid":
            from .chair_sampling import make_noise as chair_noise, clamped_sample as chair_sample
            z = chair_noise(args.seed, N, len(C))
            sampler = chair_sample
        else:
            z = make_noise(args.seed)
            sampler = clamped_sample
        raw, path, report["sampling"] = sampler(model, z.cuda(), torch.from_numpy(C).cuda(), report["counts"])
        report["state_after"] = state_sha256(model)
        if report["state_after"] != report["load"]["state_sha256"]:
            raise RuntimeError("Frozen T1 changed")
        np.savez_compressed(args.out / "raw.npz", output=raw, path=path, initial_gaussian=z.numpy())
        report.update(status="PASS", raw_sha256=file_sha256(args.out / "raw.npz"))
    except BaseException as error:
        report.update(status="STOPPED", error=repr(error))
        raise
    finally:
        atomic_json(args.out / "run.json", report)
    print("PASS: " + str(args.out.resolve()))


if __name__ == "__main__":
    main()
