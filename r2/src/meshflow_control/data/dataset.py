"""Frozen repaired meshes and C partitions, with an explicit model boundary."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .io import read_json, digest, bits

CYCLIC = np.asarray([[0, 1, 2], [1, 2, 0], [2, 0, 1]], np.int64)

def gaussian(seed, count, fp32=False):
    a = np.random.RandomState(int(seed)).randn(int(count), 3, 3)
    return a.astype(np.float32) if fp32 else a

def training_time(seed):
    z = np.random.RandomState(int(seed)).randn()
    return np.float32(1.0 / (1.0 + np.exp(-z)))

def permutations(seed, n, k):
    rng = np.random.RandomState(int(seed))
    return rng.permutation(n-k), CYCLIC[rng.randint(0, 3, size=n)], rng.permutation(n)

class TrainingDataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "train_manifest.json"
        self.manifest = read_json(self.manifest_path)
        if self.manifest["schema"] != "meshflow_control_training_data_v1":
            raise ValueError("Unknown training data schema")
        self.manifest_sha256 = digest(self.manifest_path)
        self.tasks = {r["task_id"]: r for r in self.manifest["tasks"]}
        self.parents = {r["uid"]: r for r in self.manifest["parents"]}
        if len(self.parents) != 32 or len(self.tasks) != 453:
            raise ValueError("Expected original 32 parents and 453 READY tasks")

    @lru_cache(maxsize=32)
    def parent(self, uid):
        row = self.parents[uid]
        p = self.root / row["npz"]
        if digest(p) != row["sha256"]:
            raise ValueError("Frozen parent data changed: " + uid)
        with np.load(p, allow_pickle=False) as f:
            result = {k: f[k].copy() for k in ("full_target", "pre_ot_full", "model_vertices", "source_face_vertex_ids")}
        n = row["N"]
        if not bits(result["pre_ot_full"] * np.float32(2), result["full_target"].reshape(n, 3, 3)):
            raise ValueError("Saved pre-OT x2 identity failed")
        if not bits(result["model_vertices"][result["source_face_vertex_ids"]], result["full_target"].reshape(n,3,3)):
            raise ValueError("Source vertex mapping differs")
        return result

    def task(self, task_id, alpha=1.0):
        task = self.tasks[task_id]
        a = np.float32(alpha)
        if not np.isfinite(a) or not .8999999 <= a <= 1.1000001:
            raise ValueError("Alpha outside frozen interval")
        parent = {k: v.copy() for k, v in self.parent(task["uid"]).items()}
        if a != np.float32(1):
            for key in ("full_target", "pre_ot_full", "model_vertices"):
                parent[key].reshape(-1, 3)[:, 0] *= a
        n, k = task["N"], task["K"]
        known = np.asarray(task["source_face_ids"], np.int64)
        free = np.asarray(task["free_source_ids"], np.int64)
        if not np.array_equal(np.sort(np.r_[known, free]), np.arange(n)) or len(known) != k:
            raise ValueError("Invalid C/free partition")
        if not bits(parent["pre_ot_full"] * np.float32(2), parent["full_target"].reshape(n,3,3)):
            raise ValueError("Augmented pre-OT identity differs")
        parent.update(N=n, K=k, source_face_ids=known, free_source_ids=free)
        return parent

def collate(samples, device="cpu", pad_to=None):
    import torch
    if not samples:
        raise ValueError("Empty batch")
    nmax = max(int(s["y"]) for s in samples)
    nmax = nmax if pad_to is None else int(pad_to)
    if nmax < max(int(s["y"]) for s in samples):
        raise ValueError("Padding shorter than input")
    result = {}
    for key in ("xt", "u", "context"):
        a = np.zeros((len(samples), nmax, 9), np.float32)
        for i, s in enumerate(samples):
            a[i, :int(s["y"])] = s[key]
        result[key] = torch.from_numpy(a).to(device)
    for key in ("known_mask", "valid_mask"):
        a = np.zeros((len(samples), nmax), bool)
        for i, s in enumerate(samples):
            a[i, :int(s["y"])] = s[key]
        result[key] = torch.from_numpy(a).to(device)
    result["t"] = torch.from_numpy(np.asarray([s["t"] for s in samples], np.float32)).to(device)
    result["y"] = torch.from_numpy(np.asarray([s["y"] for s in samples], np.int64)).to(device)
    return result

def model_inputs(batch):
    """Explicit positional allowlist; no GT, UID, Gamma, bbox or source IDs."""
    return tuple(batch[k] for k in ("xt", "t", "y", "valid_mask", "known_mask"))
