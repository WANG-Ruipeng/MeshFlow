"""Real author-split parents, original data/OT path, and surface prompts.

No generated substitutes. The dataset shim calls the upstream __getitem__ without
altering normalization, permutation, Gaussian noise, nested OT, or target scale.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

PROMPT_TYPES = ("single_edge", "single_face", "mixed_2edges_1face")


@contextmanager
def numpy_stream(generator):
    """Isolate upstream NumPy legacy RNG without changing its sampling semantics."""
    old = np.random.get_state()
    np.random.set_state(generator.get_state())
    try:
        yield
    finally:
        generator.set_state(np.random.get_state())
        np.random.set_state(old)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_real_parents(root, output, seed=20260928):
    """Freeze 16 train + 4 disjoint author-train diagnostic parents before output.

    Phase 3 splits/test evaluation are deliberately not started here. The four
    diagnostics are adapter-unseen only, never claimed base-pretraining-unseen.
    """
    root, output = Path(root), Path(output)
    split = root / "split/train.npz"
    folder = root / "objaverse_occ_v5_ids"
    if not split.is_file() or not folder.is_dir():
        raise FileNotFoundError("BLOCKED_DATA: real author chair train split/meshes required")
    rows = np.load(split, allow_pickle=True)["npz_list"].tolist()
    uids = sorted(set(str(row["uid"]) for row in rows))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(uids))
    selected, rejected = [], []
    for i in order:
        uid = uids[i]
        path = folder / f"{uid}.npz"
        try:
            with np.load(path, allow_pickle=False) as data:
                vertices, faces = data["vertices"], data["faces"]
                n = int(data["faces_num"])
                if not 20 < n < 800 or n != len(faces):
                    raise ValueError("outside official 20 < N < 800 or inconsistent N")
                if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.shape != (n, 3):
                    raise ValueError("invalid mesh array shape")
                if not np.isfinite(vertices).all() or np.ptp(vertices, axis=0).max() <= 0:
                    raise ValueError("invalid coordinates/bbox")
                if not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices):
                    raise ValueError("invalid face indices")
                tri = vertices[faces]
                area2 = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
                if (area2 > np.ptp(vertices, axis=0).max() ** 2 * 1e-12).sum() < 3:
                    raise ValueError("fewer than three nondegenerate prompt faces")
            selected.append(dict(uid=uid, parent_uid=uid, path=str(path.resolve()),
                                 sha256=file_sha256(path), faces=n))
        except (ValueError, KeyError, OSError) as error:
            rejected.append(dict(uid=uid, reason=str(error)))
        if len(selected) == 20:
            break
    manifest = dict(author_split="train", split_path=str(split.resolve()),
                    split_sha256=file_sha256(split), seed=seed, available_unique_uids=len(uids),
                    train=selected[:16], diagnostic=selected[16:20], rejected=rejected,
                    attempted=len(selected)+len(rejected),
                    parent_identity="author uid; no independent ancestry metadata available",
                    base_pretraining_unseen_verified=False,
                    phase3="NOT_AUTHORIZED", status="PASS" if len(selected) == 20 else "BLOCKED_DATA")
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if manifest["status"] != "PASS":
        raise RuntimeError("BLOCKED_DATA: insufficient distinct real author-train parents")
    return manifest


class RealChairStream:
    def __init__(self, parents, seed=10):
        from datasets.mesh_dataset import ObjaverseDataset
        self.parents = parents
        self.data_rng = np.random.RandomState(seed + 10000)
        self.prompt_rng = np.random.RandomState(seed + 20000)
        self.time_rng = np.random.RandomState(seed + 30000)
        self.dataset = ObjaverseDataset.__new__(ObjaverseDataset)
        self.dataset.__dict__.update(training=True, use_custom_prior=False,
            do_dataset_normalize=True, max_face_length=800, use_rot_aug=False,
            use_scale_aug=False, noise_sort="ot", use_permut_aug=True, std=0.3762)
        self.dataset.data = []
        for parent in parents:
            if file_sha256(parent["path"]) != parent["sha256"]:
                raise ValueError("Source data hash changed")
            with np.load(parent["path"], allow_pickle=False) as f:
                self.dataset.data.append({key: f[key] for key in ("vertices", "faces", "faces_num", "uid")})

    def next_effective_batch(self, size=8):
        samples = []
        with numpy_stream(self.data_rng):
            for _ in range(size):
                index = np.random.randint(len(self.parents))
                item = self.dataset[index]
                item["parent_uid"] = self.parents[index]["uid"]
                samples.append(item)
        for item in samples:
            kind = int(self.prompt_rng.randint(3))
            item["prompt"] = make_prompt(item["coords"], kind, self.prompt_rng)
            item["t"] = float(1 / (1 + np.exp(-self.time_rng.randn())))
        return samples

    def state_dict(self):
        return {key: getattr(self, key).get_state() for key in ("data_rng", "prompt_rng", "time_rng")}

    def load_state_dict(self, state):
        for key, value in state.items():
            getattr(self, key).set_state(value)


def sample_surface_primitive(vertices, arity, count, rng, include_vertices=True):
    if arity == 2:
        # Equally spaced context includes endpoints. Independent check points use
        # fresh uniform internal coordinates, never just the vertices.
        u = np.linspace(0, 1, count) if include_vertices else rng.uniform(0, 1, count)
        return (1-u[:, None])*vertices[0] + u[:, None]*vertices[1]
    n = count - 3 if include_vertices else count
    uv = rng.uniform(0, 1, (n, 2))
    root = np.sqrt(uv[:, 0:1])
    points = (1-root)*vertices[0] + root*(1-uv[:, 1:2])*vertices[1] + root*uv[:, 1:2]*vertices[2]
    return np.concatenate([vertices[:3], points]) if include_vertices else points


def make_prompt(tokens, kind, rng):
    tri = np.asarray(tokens).reshape(-1, 3, 3)
    scale = np.ptp(tri.reshape(-1, 3), axis=0).max()
    area2 = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    legal = np.flatnonzero(area2 > scale**2 * 1e-12)
    arities = [2] if kind == 0 else [3] if kind == 1 else [2, 2, 3]
    counts = [32] if kind < 2 else [8, 8, 16]
    face_ids = rng.choice(legal, len(arities), replace=False)
    vertices, points, groups = [], [], []
    for j, (face, arity, count) in enumerate(zip(face_ids, arities, counts)):
        v = tri[face].copy()
        if arity == 2:
            chosen = rng.choice(3, 2, replace=False)
            v = np.stack([v[chosen[0]], v[chosen[1]], v[chosen[1]]])
        vertices.append(v)
        points.append(sample_surface_primitive(v, arity, count, rng))
        groups.extend([j] * count)
    return dict(vertices=np.asarray(vertices, dtype=np.float32), arity=np.asarray(arities, dtype=np.int64),
                primitive_mask=np.ones(len(arities), dtype=bool), points=np.concatenate(points).astype(np.float32),
                point_primitive=np.asarray(groups, dtype=np.int64), point_mask=np.ones(32, dtype=bool))


def collate_prompts(prompts, device="cpu"):
    k = max(len(p["arity"]) for p in prompts)
    m = max(len(p["points"]) for p in prompts)
    arrays = dict(vertices=np.zeros((len(prompts), k, 3, 3), np.float32),
                  arity=np.full((len(prompts), k), 2, np.int64),
                  primitive_mask=np.zeros((len(prompts), k), bool),
                  points=np.zeros((len(prompts), m, 3), np.float32),
                  point_primitive=np.zeros((len(prompts), m), np.int64),
                  point_mask=np.zeros((len(prompts), m), bool))
    for i, prompt in enumerate(prompts):
        for key in arrays:
            arrays[key][i, :len(prompt[key])] = prompt[key]
    return {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}


def collate_samples(samples, device="cpu"):
    from datasets.mesh_dataset import collate_fn
    if any(int(item["num_faces"]) >= 800 for item in samples):
        raise ValueError("No truncation of source faces is permitted")
    batch = {k: v.to(device) for k, v in collate_fn(samples).items()}
    batch["prompts"] = collate_prompts([item["prompt"] for item in samples], device)
    batch["t"] = torch.tensor([item["t"] for item in samples], device=device, dtype=torch.float32)
    return batch


def sample_input_hash(samples):
    h = hashlib.sha256()
    for item in samples:
        h.update(item["parent_uid"].encode())
        for value in (item["coords"], item["noise"], item["num_faces"], item["t"], *item["prompt"].values()):
            array = np.ascontiguousarray(value)
            h.update(str((array.dtype, array.shape)).encode())
            h.update(array.tobytes())
    # Masks are uniquely determined by N and microbatch grouping; actual padded
    # model tensors are separately hashed by the training loop.
    return h.hexdigest()
