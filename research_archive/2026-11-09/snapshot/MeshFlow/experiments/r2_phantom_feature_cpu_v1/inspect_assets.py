"""Read-only CPU environment and saved asset schema inspection."""
import json, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import scipy, psutil

ROOT = Path("/mnt/e/MeshFlow")
OUT = ROOT / "experiments/r2_phantom_feature_cpu_v1"
resource = dict(python=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
                polygon_union="local_convex_difference_no_install", cpu_count=psutil.cpu_count(),
                memory=psutil.virtual_memory()._asdict(),
                cpu_percent=psutil.cpu_percent(interval=.3),
                load=os.getloadavg(), processes=[])
for p in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
    if any(s in (p.info["name"] or "").lower() for s in ("python", "audit", "ot")):
        d = p.info.copy()
        if d["memory_info"]:
            d["rss"] = d.pop("memory_info").rss
        resource["processes"].append(d)
(OUT / "resources.json").write_text(json.dumps(resource, indent=2), encoding="utf-8")
print(json.dumps(resource))
paths = [
"experiments/r2_bench_clean_multipart_seam_v1/evaluation/NEW_HEAD/1dd7409c95d59d0da192483aa282f8e5__SEAT_SURFACE_1__n0/raw/raw.npz",
"experiments/r2_bench_clean_multipart_seam_v1/evaluation/NEW_HEAD/1dd7409c95d59d0da192483aa282f8e5__SEAT_SURFACE_1__n0/post/post.npz",
"experiments/r2_bench_clean_multipart_seam_v1/cpu/NEW_HEAD/1dd7409c95d59d0da192483aa282f8e5__SEAT_SURFACE_1__n0/CLEAN_FINAL_POST/post.npz",
]
for path in paths:
    with np.load(ROOT/path, allow_pickle=False) as z:
        d = {k: dict(shape=z[k].shape, dtype=str(z[k].dtype),
                    value=z[k].tolist() if z[k].size < 12 else None) for k in z.files}
    print(path, json.dumps(d))
for name in ("experiments/r2_hinge_v2_cpu_v1/protocol.json",
             "experiments/r2_chair_back_transfer_v1/eval_manifest.json"):
    x = json.loads((ROOT/name).read_text())
    print(name, list(x), "FIRST", json.dumps(x["requests"][0]))
