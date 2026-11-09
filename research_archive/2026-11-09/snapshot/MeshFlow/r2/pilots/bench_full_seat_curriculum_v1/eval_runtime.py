"""GPU-safe evaluation runtime; C-only inputs, no CPU scorer imports."""
from pathlib import Path
import numpy as np
from .common import ROOT, OUT, HEAD, read, resolve, sha
from .panel import LABELS, requests, arrays, verify, P
HEAD_ROOT=HEAD.parent
BASELINE_LABEL=LABELS[0]
ALL_LABELS=FINAL_LABELS=LABELS

def label_spec(label):
    if label not in LABELS:raise ValueError('Unknown label')
    reg=read(OUT/'registration.json');s=dict(reg['evaluation_labels'][label]);s.setdefault('subset','full')
    wanted={LABELS[0]:('NATIVE',0,12000),LABELS[1]:('NATIVE',reg['stage_updates']['NATIVE'],12000+reg['stage_updates']['NATIVE']),LABELS[2]:('SEAM',reg['stage_updates']['SEAM'],12000+sum(reg['stage_updates'].values()))}[label]
    if (s['arm'],s['local_step'],s['global_step'])!=wanted or s['subset']!='full':raise ValueError('Unexpected endpoint')
    expected=OUT/'training'/s['arm']/('generator_step'+str(s['local_step'])+'.pt')
    if resolve(s['generator_path'])!=expected:raise ValueError('Unexpected generator export')
    if s.get('mode') != ('SEAM' if label==LABELS[2] else 'NATIVE'):raise ValueError('Registered representation differs')
    s['mode']='SEAM' if label==LABELS[2] else 'PLAIN'
    return s

def new_requests(label):return requests(label)
def reuse_spec(q,label):return None

def copy_reference(*args):raise RuntimeError('This panel has no old model output reuse')

def root_for(label):return OUT


def source_assets(q):
    for key in ("inference_npz", "metadata_npz", "source_npz", "noise_npz"):
        digest = q.get(key + "_sha256")
        if digest is None and key == "noise_npz":
            digest = q["noise_sha256"]
        if digest is None or sha(q[key]) != digest:
            raise ValueError("Frozen input identity differs: " + key)
    a = arrays(q["inference_npz"])
    if set(a) != {"C_vertices", "C_faces_original", "C_faces_model", "N", "K"}:
        raise ValueError("Inference bundle contains non-C information")
    core = {k: a[k] for k in ("C_vertices", "C_faces_original", "C_faces_model")}
    core["total_N"] = int(a["N"])
    if core["total_N"] != q["N"] or len(core["C_faces_model"]) != q["K"]:
        raise ValueError("Input N/K differs")
    noise = arrays(q["noise_npz"])["z"]
    if noise.shape != (q["N"] - q["K"], 3, 3) or noise.dtype != np.float32:
        raise ValueError("Frozen standard Gaussian shape or precision differs")
    from meshflow_control.source_winding import validate_source
    source = arrays(q["source_npz"])
    validate_source(source, core["C_vertices"][core["C_faces_model"]])
    return core, source

def raw_completion(label):
    label_spec(label)
    return OUT / "evaluation" / label / "raw_complete_full.json"

def verify_raw(label):
    qs = requests(label)
    done = read(raw_completion(label))
    if done["status"] != "COMPLETE" or done["outputs"] != 24:
        raise ValueError("Incomplete RAW wave")
    verify(done["files"])
    receipts = []
    for q in qs:
        path = OUT / "evaluation" / label / q["request_id"] / "raw/receipt.json"
        r = read(path)
        if (r["request_id"] != q["request_id"] or r["label"] != label
                or r["generator_forwards"] != 50 or r["D0_forwards"] != 0
                or r["raw_sha256"] != sha(path.parent / "raw.npz")
                or r["noise_sha256"] != q["noise_sha256"]):
            raise ValueError("Invalid RAW receipt: " + str(path))
        if bool(reuse_spec(q, label)) != (r.get("execution") == "REUSED"):
            raise ValueError("Reference reuse provenance differs")
        verify(r["files"])
        receipts.append(path)
    return qs, receipts

def all_raw_barrier():
    receipts = []
    for label in LABELS:
        _, rr = verify_raw(label)
        receipts.extend(rr)
    if len(receipts) != 72:
        raise ValueError("All 72 paired RAW products must exist before any new D0")
    return receipts