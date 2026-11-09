"""Exactly one resident official D0 pass per completed registered RAW.

GPU execution occurs only in the explicit worker entrypoint. It writes proposals
and authenticated receipts; the model-free CPU stage later restores C, applies
GEO_C and source winding. No downloads, retries, additional noise, or binding.
"""
from __future__ import annotations

from . import common as W
from . import data as D
import argparse
import json
import sys

sys.path.insert(0, str(D.ROOT / "r2/src"))

OFFICIAL_D0_SHA256 = "6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36"
OFFICIAL_D0_BYTES = 7289979546


def stage_requests(manifest, *, round_index=None, evaluation=None):
    if (round_index is None) == (evaluation is None):
        raise ValueError("Exactly one round or evaluation label is required")
    if round_index is not None:
        if not 0 <= round_index < 16:
            raise ValueError("Frozen round range is 0..15")
        label = "NFT"
        requests = [q for q in manifest["training_requests"] if q["round_index"] == round_index]
        expected = 32
        stage = D.OUT / f"training/round_{round_index:02d}"
    else:
        if evaluation not in ("BASE", "NFT", "SELF_FM"):
            raise ValueError("Unknown registered evaluation label")
        label = evaluation
        requests = list(manifest["evaluation_requests"])
        if label == "BASE":
            requests = [q for q in requests if q["evaluation_noise_partition"] == "FRESH"]
        expected = 12 if label == "BASE" else 36
        stage = D.OUT / "evaluation" / label
    if len(requests) != expected or len({q["request_id"] for q in requests}) != expected:
        raise ValueError("Registered D0 request count/identity differs")
    return stage, label, requests


def prepare_jobs(manifest, *, round_index=None, evaluation=None):
    """CPU-only: require every RAW receipt before constructing any D0 job."""
    from .sample import verify_receipt
    stage, label, requests = stage_requests(manifest, round_index=round_index, evaluation=evaluation)
    index_path = stage / "raw_index.json"
    index = D.read(index_path)
    if index.get("status") != "COMPLETE" or index.get("label") != label:
        raise ValueError("D0 requires complete RAW index for this label")
    expected = {q["request_id"]: q for q in requests}
    receipts = index["receipts"]
    if len(receipts) != len(expected) or {r["request_id"] for r in receipts} != set(expected):
        raise ValueError("All registered RAW outputs must complete before D0")
    by_id = {r["request_id"]: r for r in receipts}
    jobs = []
    for q in requests:
        receipt = by_id[q["request_id"]]
        if receipt["label"] != label or (receipt["N"], receipt["K"]) != (q["N"], q["K"]):
            raise ValueError("RAW receipt label/N/K differs")
        verify_receipt(q, receipt, model_identity=index["model_identity"])
        if D.sha(q["condition_source"]) != q["condition_source_sha256"]:
            raise ValueError("Source condition metadata changed")
        jobs.append(dict(request_id=q["request_id"], label=label,
            raw_path=receipt["raw_path"], raw_sha256=receipt["raw_sha256"],
            condition_path=q["condition_source"],
            condition_sha256=q["condition_source_sha256"], K=q["K"], N=q["N"],
            proposal_dir=str(D.checked(stage / "proposal" / q["request_id"])),
            out_dir=str(D.checked(stage / "post" / q["request_id"]))))
    return stage, label, jobs, index_path


def retained_d0_binding():
    """Read the already audited local asset; missing assets are never fetched."""
    source_registration = D.FULL / "registration.json"
    binding = D.read(source_registration)["d0_checkpoint"]
    path = D.resolve(binding["path"])
    if binding["sha256"] != OFFICIAL_D0_SHA256 or binding["bytes"] != OFFICIAL_D0_BYTES:
        raise ValueError("Historical official D0 identity differs")
    if not path.is_file() or path.stat().st_size != OFFICIAL_D0_BYTES:
        raise FileNotFoundError("Retained official Chair D0 asset is missing or incomplete; no download")
    # load_denoiser verifies the complete 7.29 GB file SHA before its strict load.
    return dict(path=str(path), sha256=OFFICIAL_D0_SHA256, bytes=OFFICIAL_D0_BYTES,
                source_registration=str(source_registration),
                source_registration_sha256=D.sha(source_registration))


def run(*, round_index=None, evaluation=None):
    manifest_path = D.OUT / "inputs/manifest.json"
    manifest = D.read(manifest_path)
    if manifest.get("status") != "FROZEN":
        raise ValueError("D0 requires frozen experiment inputs")
    stage, label, jobs, index_path = prepare_jobs(manifest, round_index=round_index, evaluation=evaluation)
    binding = retained_d0_binding()
    worker_dir = stage / "denoise_worker"
    if (stage / "proposal_index.json").exists():
        raise RuntimeError("Complete D0 proposals must be reused, not relaunched")
    identity = dict(stage="OFFICIAL_D0_PROPOSALS", label=label, round_index=round_index,
                    requests=len(jobs), manifest_sha256=D.sha(manifest_path),
                    raw_index_sha256=D.sha(index_path), checkpoint=binding,
                    proposal_only=True, D0_per_request=1)
    with W.Phase(worker_dir, identity) as phase:
        resource_path = worker_dir / "resources_initial.json"
        D.write(resource_path, phase.initial)
        D.write(worker_dir / "jobs.json", dict(identity=identity, jobs=jobs))
        from .evaluation import postprocess_batch
        result = postprocess_batch(jobs, binding["path"], worker_dir / "batch",
            resource_snapshot_path=resource_path, proposal_only=True)
        if (result["status"] != "COMPLETE" or result["jobs"] != len(jobs)
                or result["official_D0_forward_calls_in_all_jobs"] != len(jobs)):
            raise RuntimeError("Resident D0 batch did not fulfill the exact request budget")
        if result["new_D0_forwards"] != len(jobs) or result["resident_models_loaded"] != 1:
            raise RuntimeError("Expected one new official D0 per request and one resident model")
        result.update(label=label, round_index=round_index, identity=identity,
                      proposal_only=True, post_status="NOT_RUN_CPU_STAGE_FOLLOWS",
                      worker_resources=str(worker_dir / "resources.json"),
                      actual_GPU_workers=1, downloads=0)
        D.write(stage / "proposal_index.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--round", type=int)
    group.add_argument("--evaluation", choices=["BASE", "NFT", "SELF_FM"])
    args = parser.parse_args()
    result = run(round_index=args.round, evaluation=args.evaluation)
    print(json.dumps(dict(status=result["status"], jobs=result["jobs"],
                          new_D0_forwards=result["new_D0_forwards"],
                          proposal_only=result["proposal_only"]), sort_keys=True))


if __name__ == "__main__":
    main()
