"""Lossless triangle soup IO, without welding, quantization or winding repair."""
from pathlib import Path
import json
import struct
import numpy as np

def validate_C(C,N):
    a=np.asarray(C)
    if a.dtype!=np.float32 or a.ndim not in (2,3) or a.shape[1:] not in ((9,),(3,3)):
        raise ValueError("C must be FP32[K,9] or FP32[K,3,3]")
    a=np.ascontiguousarray(a.reshape(-1,3,3))
    if type(N) is not int or not 128<=N<=256 or not 0<len(a)<N or not np.isfinite(a).all():
        raise ValueError("Require finite C and explicit N128..256, 0<K<N")
    return a
def load_condition(path,key="C"):
    path=Path(path)
    if path.suffix.lower()==".npy": return np.load(path,allow_pickle=False)
    with np.load(path,allow_pickle=False) as data:
        if key not in data: raise KeyError("Missing condition array "+key)
        return data[key].copy()
def write_ply(path,triangles):
    t=np.ascontiguousarray(triangles,dtype=np.float32).reshape(-1,3,3)
    if not np.isfinite(t).all(): raise ValueError("Nonfinite mesh")
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    header=("ply\nformat binary_little_endian 1.0\ncomment RAW unmodified triangle soup\n"
        f"element vertex {3*len(t)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(t)}\nproperty list uchar int vertex_indices\nend_header\n").encode("ascii")
    with path.open("wb") as handle:
        handle.write(header);handle.write(t.astype("<f4",copy=False).tobytes())
        face=np.empty(len(t),dtype=np.dtype([("n","u1"),("v","<i4",(3,))],align=False))
        face["n"]=3;face["v"]=np.arange(3*len(t)).reshape(-1,3);handle.write(face.tobytes())
def read_ply(path):
    with Path(path).open("rb") as handle:
        header=[]
        while True:
            line=handle.readline()
            if not line: raise ValueError("Missing PLY header terminator")
            header.append(line.decode("ascii").strip())
            if header[-1]=="end_header": break
        if "format binary_little_endian 1.0" not in header: raise ValueError("Expected our binary RAW PLY")
        nv=int(next(x for x in header if x.startswith("element vertex ")).split()[-1])
        nf=int(next(x for x in header if x.startswith("element face ")).split()[-1])
        vertices=np.frombuffer(handle.read(nv*12),dtype="<f4").copy().reshape(nv,3)
        faces=np.frombuffer(handle.read(nf*13),dtype=np.dtype([("n","u1"),("v","<i4",(3,))],align=False))
        if len(faces)!=nf or not (faces["n"]==3).all() or handle.read(1): raise ValueError("Malformed RAW PLY")
        return vertices[faces["v"]],faces["v"].copy()
def write_raw(out,raw,C,z,trajectory,audit):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=np.ascontiguousarray(raw,dtype=np.float32).reshape(-1,9)
    C=np.ascontiguousarray(C,dtype=np.float32).reshape(-1,3,3);K=len(C)
    if raw[:K].tobytes()!=C.tobytes(): raise ValueError("C bytes changed before export")
    np.savez_compressed(out/"raw.npz",raw=raw,free=raw[K:].reshape(-1,3,3),C=C,
        z=np.asarray(z,dtype=np.float32),trajectory=np.asarray(trajectory,dtype=np.float32),
        known_face_indices=np.arange(K,dtype=np.int64),
        source_corner_to_output_vertex=np.arange(3*K,dtype=np.int64))
    write_ply(out/"raw.ply",raw)
    actual,indices=read_ply(out/"raw.ply")
    with np.load(out/"raw.npz",allow_pickle=False) as saved:
        if saved["raw"].tobytes()!=raw.tobytes() or actual.reshape(-1,9).tobytes()!=raw.tobytes():
            raise RuntimeError("RAW export did not preserve bit patterns/multiplicity/winding")
    audit.update(export_npz_ply_bitwise=True,NPZ_C_bitwise=True,PLY_C_bitwise=True,known_coordinates_bitwise=True,
        original_vertex_id_graph="NOT_PROVIDED",output_stage="RAW",postprocessing=None)
    (out/"run.json").write_text(json.dumps(audit,indent=2,allow_nan=False),encoding="utf-8")
    return out/"raw.npz"
