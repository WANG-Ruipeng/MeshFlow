"""Small IO and hashing helpers, independent of experiment orchestration."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def array_hash(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        dtype, shape = str(value.dtype), tuple(value.shape)
        raw = value.view(torch.uint8).numpy().tobytes()
    else:
        value = np.ascontiguousarray(value)
        dtype, shape, raw = str(value.dtype), tuple(value.shape), value.tobytes()
    return hashlib.sha256(str((dtype, shape)).encode() + raw).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
