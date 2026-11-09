"""Read-only MF-H1 asset/environment audit; writes only a fresh audit directory.

Run with the existing MeshFlow interpreter, for example under WSL:
  python experiments/mf_h1_local0/audit_assets.py --output <new-directory>

No download, package install, source mutation, unsafe torch.load fallback, model
generation, or training is performed. Explicit paths take precedence over the
small set of project/config/cache locations searched automatically.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time


REFERENCE = "8cc74da013d3d2feec1853cccbb1f9ac37814b95"
HF_FILE = "v1/120m-ot-v-chair/checkpoints/last.pt"
OFFICIAL_CHAIR_BYTES = 2084287208
OFFICIAL_CHAIR_SHA256 = "bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d"
SOURCE_FILES = [
    "configs/snet/base-120m-ot-v-chair.yaml", "models/equidit.py",
    "models/attention.py", "models/utils.py", "datasets/mesh_dataset.py",
    "utils/ot_utils.py", "utils/sample.py", "flow_matching.py", "train.py",
    "inference.py", "README.md", "requirements.txt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_file(path: Path, hash_bytes: bool = True) -> dict:
    result = {"path": str(path.absolute()), "exists": path.is_file()}
    if result["exists"]:
        stat = path.stat()
        result.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if hash_bytes:
            result["sha256"] = sha256(path)
    return result


def command(args: list[str], cwd: Path | None = None) -> dict:
    try:
        proc = subprocess.run(args, cwd=cwd, text=True, encoding="utf-8",
                              errors="replace", capture_output=True, timeout=30)
        return {"argv": args, "returncode": proc.returncode,
                "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"argv": args, "error": repr(exc)}


def git(repo: Path, *args: str) -> dict:
    return command(["git", "-c", f"safe.directory={repo}", "-c", "core.autocrlf=true", *args], cwd=repo)


def cache_roots() -> list[Path]:
    candidates = [Path.home() / ".cache/huggingface/hub"]
    for key in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        if os.environ.get(key):
            candidates.append(Path(os.environ[key]))
    if os.environ.get("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("XDG_CACHE_HOME"):
        candidates.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface/hub")
    return sorted(set(candidates))


def checkpoint_candidates(repo: Path, explicit: str | None) -> tuple[list[Path], list[dict]]:
    paths = ([Path(explicit)] if explicit else []) + [
        repo / "output/120m-ot-v-chair/checkpoints/last.pt",
        repo / "checkpoints/120m-ot-v-chair/last.pt",
        repo / HF_FILE,
    ]
    searched = []
    for root in cache_roots():
        cache = root / "datasets--qsun2001--meshflow"
        searched.append({"path": str(cache), "exists": cache.exists(),
                         "scope": "exact official HF dataset cache"})
        snapshots = cache / "snapshots"
        if snapshots.is_dir():
            paths.extend(snapshot / HF_FILE for snapshot in snapshots.iterdir()
                         if snapshot.is_dir())
    for path in paths:
        searched.append({"path": str(path.absolute()), "exists": path.is_file(),
                         "scope": "explicit path" if explicit and path == Path(explicit)
                         else "official chair checkpoint candidate"})
    return [path for path in paths if path.is_file()], searched


def environment(probe_gpu: bool) -> dict:
    import torch
    torch.set_num_threads(1)
    packages = {}
    for name in ("torch", "numpy", "scipy", "trimesh", "einops", "torchdiffeq",
                 "PyYAML", "psutil", "tqdm", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable, "python": sys.version,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "packages": packages, "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_arch_list": torch.cuda.get_arch_list(),
        "nvidia_smi": command(["nvidia-smi"]),
        "gpu_current": command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]),
        "gpu_compute_processes": command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]),
        "attention_backend": {
            "official_implementation": "torch.nn.functional.scaled_dot_product_attention",
            "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            "actual_official_dispatch": "NOT_RUN: no official model forward in audit",
        },
        "bf16_probe": {"status": "NOT_RUN"},
    }
    try:
        import psutil
        result["memory"] = dict(psutil.virtual_memory()._asdict())
        result["process_rss_bytes"] = psutil.Process().memory_info().rss
    except ImportError:
        result["memory"] = {"status": "NOT_RUN: psutil unavailable"}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        result["device"] = {"name": props.name, "capability": [props.major, props.minor],
                            "total_memory_bytes": total, "free_memory_bytes": free,
                            "bf16_reported_supported": torch.cuda.is_bf16_supported()}
        if probe_gpu:
            start = time.perf_counter()
            try:
                # A tiny standalone environment probe, never an official model result.
                torch.cuda.reset_peak_memory_stats()
                x = torch.linspace(-1, 1, 128 * 128, device="cuda").reshape(128, 128)
                x.requires_grad_()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = (x @ x.T).float().square().mean()
                loss.backward()
                torch.cuda.synchronize()
                okay = bool(torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item())
                result["bf16_probe"] = {
                    "status": "PASS" if okay else "FAIL", "shape": [128, 128],
                    "operation": "BF16 autocast matmul and backward",
                    "loss": loss.item(), "gradient_abs_max": x.grad.abs().max().item(),
                    "elapsed_seconds": time.perf_counter() - start,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "official_model_forward_count": 0,
                }
                del x, loss
                torch.cuda.empty_cache()
            except Exception as exc:
                result["bf16_probe"] = {"status": "FAIL", "error": repr(exc)}
    result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return result


def inspect_checkpoint(path: Path, config: dict) -> dict:
    import torch
    result = record_file(path)
    result.update(load_policy={"map_location": "cpu", "weights_only": True, "mmap": True},
                  status="NOT_RUN", strict_model_load="NOT_RUN")
    result["official_identity"] = {"expected_bytes": OFFICIAL_CHAIR_BYTES, "expected_sha256": OFFICIAL_CHAIR_SHA256, "source": "official HF API identity supplied by root asset verification", "pass": result.get("bytes") == OFFICIAL_CHAIR_BYTES and result.get("sha256") == OFFICIAL_CHAIR_SHA256}
    if not result["official_identity"]["pass"]:
        result["status"] = "BLOCKED_CHECKPOINT_IDENTITY"
        return result
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(loaded, dict):
            raise TypeError(f"checkpoint top level is {type(loaded).__name__}, expected dict")
        result["top_level_keys"] = list(loaded)
        embedded = loaded.get("config", loaded.get("args"))
        result["embedded_configuration"] = embedded if isinstance(embedded, dict) else repr(embedded)
        def differences(a, b, prefix=""):
            found = []
            for key in sorted(set(a) | set(b)):
                name = prefix + str(key)
                av, bv = a.get(key), b.get(key)
                if isinstance(av, dict) and isinstance(bv, dict):
                    found.extend(differences(av, bv, name + "."))
                elif av != bv:
                    found.append({"field": name, "reference": av, "checkpoint": bv})
            return found
        result["configuration_differences"] = differences(config, embedded) if isinstance(embedded, dict) else None
        result["critical_configuration_matches"] = isinstance(embedded, dict) and all(embedded.get(key) == config.get(key) for key in ("data", "model", "transport", "sample"))
        ema = loaded["ema"]
        if not isinstance(ema, dict) or not ema:
            raise TypeError("ema must be a nonempty state dictionary")
        prefixed = [key.startswith("module.") for key in ema]
        if any(prefixed) and not all(prefixed):
            raise ValueError("mixed module. prefixes; refusing ambiguous key normalization")
        if all(prefixed):
            ema = {key.removeprefix("module."): val for key, val in ema.items()}
        result["removed_uniform_module_prefix"] = all(prefixed)
        result["ema_tensor_shapes"] = {key: {"shape": list(val.shape), "dtype": str(val.dtype)}
                                       for key, val in ema.items() if torch.is_tensor(val)}
        result["ema_parameter_elements"] = sum(val.numel() for val in ema.values() if torch.is_tensor(val))
        result["ema_tensor_bytes"] = sum(val.numel() * val.element_size() for val in ema.values() if torch.is_tensor(val))
        layer_ids = sorted({int(key.split(".")[1]) for key in ema if key.startswith("layers.")})
        result["observed_layer_ids"] = layer_ids
        expected_shapes = {
            "x_embedder.mlp.0.weight": [768, 123],
            "y_embedder.embedding_table.weight": [41, 768],
            "final_layer.linear.weight": [3, 768],
        }
        result["shape_checks"] = {key: {"expected": shape,
            "actual": list(ema[key].shape) if key in ema else None,
            "pass": key in ema and list(ema[key].shape) == shape}
            for key, shape in expected_shapes.items()}
        result["shape_checks"]["layers"] = {"expected": list(range(12)), "actual": layer_ids,
                                               "pass": layer_ids == list(range(12))}
        result["status"] = "PASS" if all(v["pass"] for v in result["shape_checks"].values()) and result["critical_configuration_matches"] else "FAIL"
        result["note"] = "Tensor inspection only. Shape agreement cannot prove official weight identity or replace strict loading."
    except Exception as exc:
        result.update(status="BLOCKED_SAFE_LOAD" if "Weights only load failed" in str(exc) else "FAIL",
                      error=repr(exc), unsafe_fallback_used=False)
    return result


def inspect_data(path: Path) -> dict:
    result = {"path": str(path.absolute()), "exists": path.is_dir(), "status": "BLOCKED_DATA",
              "source_identity": "author train/test split identity unverified until paths and provenance are supplied"}
    if not path.is_dir():
        return result
    result["splits"] = {name: record_file(path / "split" / f"{name}.npz") for name in ("train", "test")}
    meshes = path / "objaverse_occ_v5_ids"
    result["mesh_directory"] = str(meshes)
    files = sorted(meshes.glob("*.npz")) if meshes.is_dir() else []
    result["mesh_npz_count"] = len(files)
    # Byte-level manifest avoids executing legacy object arrays during discovery.
    result["mesh_files"] = [record_file(item) for item in files]
    if files and all(item["exists"] for item in result["splits"].values()):
        result["status"] = "FILES_FOUND_IDENTITY_REQUIRES_VALIDATION"
    if result["status"] == "FILES_FOUND_IDENTITY_REQUIRES_VALIDATION":
        import numpy as np
        splits = {}
        for name in ("train", "test"):
            with np.load(path / "split" / f"{name}.npz", allow_pickle=True) as archive:
                records = archive["npz_list"].tolist()
            if not all(isinstance(item, dict) and isinstance(item.get("uid"), str) for item in records):
                raise ValueError("Unexpected author split record format")
            uids = [item["uid"] for item in records]
            splits[name] = {"record_count": len(records), "unique_uid_count": len(set(uids)),
                            "duplicate_uid_count": len(uids)-len(set(uids)), "uids": uids,
                            "first_record_schema": {key: type(val).__name__ for key,val in records[0].items()}}
        train_uids, test_uids = set(splits["train"]["uids"]), set(splits["test"]["uids"])
        file_uids = {item.stem for item in files}
        result["author_splits"] = splits
        result["train_test_uid_overlap"] = sorted(train_uids & test_uids)
        result["split_uids_without_mesh"] = sorted((train_uids | test_uids) - file_uids)
        result["mesh_uids_outside_splits"] = sorted(file_uids - (train_uids | test_uids))
        valid = not (result["train_test_uid_overlap"] or result["split_uids_without_mesh"] or result["mesh_uids_outside_splits"])
        valid = valid and not any(item["duplicate_uid_count"] for item in splits.values())
        result["status"] = "PASS" if valid else "FAIL_SPLIT_IDENTITY"
        result["source_identity"] = "Unmodified author train/test records packaged in user-provided official shapenet-cls.tar.gz; archive and extracted-file SHA256 are preserved separately."
        result["npz_object_deserialization"] = "Official legacy npz_list dictionary array read with allow_pickle=True; mesh numeric arrays use allow_pickle=False."
        with np.load(files[0], allow_pickle=False) as mesh:
            result["mesh_schema_example"] = {key: {"shape": list(mesh[key].shape), "dtype": str(mesh[key].dtype)} for key in mesh.files}
    else:
        result["npz_object_deserialization"] = "NOT_RUN by discovery audit"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", help="Explicit official chair checkpoint; no download")
    parser.add_argument("--data", type=Path, help="Explicit author-processed chair directory")
    parser.add_argument("--skip-gpu-probe", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    allowed = repo / "experiments/mf_h1_local0/outputs"
    if not output.is_relative_to(allowed):
        raise ValueError(f"Audit output must stay under {allowed}")
    output.mkdir(parents=True, exist_ok=False)

    import yaml
    config_path = repo / "configs/snet/base-120m-ot-v-chair.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = {
        "repo_path": str(repo), "windows_repo_path": "E:\\MeshFlow" if str(repo) == "/mnt/e/MeshFlow" else None,
        "reference_commit": REFERENCE,
        "head": git(repo, "rev-parse", "HEAD"), "status": git(repo, "status", "--short"),
        "diff": git(repo, "diff", "--stat"), "tracked_diff": git(repo, "diff", "--"),
        "remotes": git(repo, "remote", "-v"),
        "files": [record_file(repo / name) for name in SOURCE_FILES],
        "reference_config": config,
        "applicable_agents_files": [],
    }
    for parent in [repo, *repo.parents]:
        agent_path = parent / "AGENTS.md"
        if agent_path.is_file():
            source["applicable_agents_files"].append(record_file(agent_path))
    candidates, searched = checkpoint_candidates(repo, args.checkpoint)
    if args.checkpoint:
        checkpoint = inspect_checkpoint(Path(args.checkpoint), config) if Path(args.checkpoint).is_file() else {
            "status": "BLOCKED_CHECKPOINT_BYTES", "path": str(Path(args.checkpoint).absolute())}
    elif len(candidates) == 1:
        checkpoint = inspect_checkpoint(candidates[0], config)
    elif len(candidates) > 1:
        checkpoint = {"status": "BLOCKED_AMBIGUOUS_CHECKPOINT", "candidates": [str(p) for p in candidates]}
    else:
        checkpoint = {"status": "BLOCKED_CHECKPOINT_BYTES", "path": None, "sha256": None,
                      "top_level_keys": None, "strict_model_load": "NOT_RUN"}
    checkpoint["expected_source"] = {"repo": "qsun2001/meshflow", "type": "dataset", "file": HF_FILE}
    data_path = args.data or repo / config["data"]["data_path"]
    assets = {"checkpoint": checkpoint, "checkpoint_search": searched, "data": inspect_data(data_path),
              "excluded_assets": [], "downloaded_bytes": 0, "uploaded_bytes": 0,
              "official_model_forward_count": 0, "wrapper_forward_count": 0,
              "complete_sampling_count": 0, "optimizer_update_count": 0}
    unrelated = repo / "output/120m-ot-v-bench/checkpoints/01000000.pt"
    if unrelated.is_file():
        entry = record_file(unrelated)
        entry["exclusion_reason"] = "bench checkpoint, not required official chair EMA; never loaded or used"
        assets["excluded_assets"].append(entry)
    wrong_data = repo / "downloaded_data/ss_overfit"
    if wrong_data.is_dir():
        assets["excluded_assets"].append({"path": str(wrong_data),
            "exclusion_reason": "overfit data are not author chair train/test; never substituted"})
    env = environment(not args.skip_gpu_probe)
    env["disk"] = {"path": str(repo), **dict(zip(("total", "used", "free"), shutil.disk_usage(repo)))}
    env["execution_policy"] = "Existing environment only; no package/driver changes; no other process killed."
    for name, obj in (("environment.json", env), ("asset_manifest.json", assets), ("source_audit.json", source)):
        (output / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = f"""# MF-H1 Phase 0 asset and source audit

Generated: {env['timestamp_utc']}

- Repository: `{repo}`; HEAD: `{source['head'].get('stdout')}`.
- Reviewed reference: `{REFERENCE}`.
- Official chair checkpoint: **{checkpoint['status']}**.
- Real chair data: **{assets['data']['status']}** at `{data_path}`.
- GPU: `{env.get('device', {}).get('name', 'unavailable')}`; current free/total memory are in environment.json.
- BF16 tiny environment probe: **{env['bf16_probe']['status']}**. This is not an official model forward.
- Existing bench checkpoint and old overfit assets were explicitly excluded; no random/synthetic substitute was loaded.
- Official strict load / forward / rollout / training: **NOT_RUN** in this audit; all corresponding counters are zero.

## Source semantics checked

The chair YAML specifies version 3, width 768, 12 layers, 12 heads, max length 800,
face bin 20, QK/RMSNorm, Fourier coordinates, no face-index positional encoding,
and velocity prediction. Checkpoint configuration/shape verification is blocked
when checkpoint bytes are absent. The declared 512 evaluation face slots fit
the source configuration, but this is not a checkpoint compatibility result.

`DiT.forward` reshapes corner hidden states to [B,3N,768]. Version 3 attention
pools corners per face, attends across valid faces, and broadcasts the face
message back to corners. The original y input is face count; mask=True means
a valid face. The source attention uses PyTorch SDPA.

`ObjaverseDataset` selects `split/train.npz` or `split/test.npz`, reads UID
records from `npz_list`, and matches `objaverse_occ_v5_ids/<uid>.npz`. It keeps
20 < faces_num < max_face_length. Existing other-category or synthetic data
cannot establish the requested author chair split identity.

The original data path performs bbox centering/scaling, permutation/optional
augmentation, division by 0.3762, nested corner/face OT noise assignment,
then doubles target tokens only. `FlowMatching.sample` passes `num_steps`
linspace nodes to torchdiffeq; 51 nodes are needed for 50 Euler intervals.
Its original forward_with_cfg calls the base model's own forward, so wrapper
sampling must be verified by actual call counters separately.

## Scope and limits

Source hashes, source diff/status, checkpoint candidate paths, rejected asset
hashes, installed package versions, host/device resources and probe evidence
are preserved in adjacent JSON files. No assets were changed, downloaded,
uploaded or replaced. No Phase 3 work is authorized or started.
"""
    (output / "source_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output), "checkpoint_status": checkpoint["status"],
                      "data_status": assets["data"]["status"], "bf16_probe": env["bf16_probe"]["status"]}))


if __name__ == "__main__":
    main()
