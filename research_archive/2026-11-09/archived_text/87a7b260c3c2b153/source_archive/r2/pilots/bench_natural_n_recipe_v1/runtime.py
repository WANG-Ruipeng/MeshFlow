"""Independent Native N128..799 runtime for the frozen natural-N recipe.

The Native forward is a literal local copy with only the explicit N guard
expanded; inherited helpers preserve arithmetic and parameter registration.
No old module, function, or state is patched. No head/SEAM model is constructed.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "r2/src"))
import numpy as np
import torch
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.models.native import (
    NativeModel as _OriginalNative, assert_native_runtime, native_execution,
    native_block_forward, native_final_forward, gather_known_corners,
    apply_readout, readout_execution_options,
)
from meshflow_control.models.backbone.equidit import DiT
from meshflow_control.training.coupling import nested_ot_map, to_noise_slots
from r2.pilots.bench_parts_convergence_v1.data import boundary_from_C
from meshflow_control.seam.slot_data import L, FEATURE_DIM

VERSION = "bench_natural_n_native128_799_v1"
N_MIN, N_MAX = 128, 799


class NativeModel(_OriginalNative):
    """Same named parameters and arithmetic; this experiment permits N128..799."""
    runtime_version = VERSION

    def forward(self, x, t, y, valid_mask=None, known_mask=None, *, observer=None):
        assert_native_runtime(self, coordinates=(x, t))
        if x.ndim != 3 or x.shape[-1] != 9:
            raise ValueError("Expected FP32 x[B,padded_N,9]")
        batch, faces, _ = x.shape
        if t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64:
            raise ValueError("Expected free t[B] and integer total N[B]")
        if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()):
            raise ValueError("Nonfinite coordinates/time")
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError("Free time outside [0,1]")
        for mask in (valid_mask, known_mask):
            if mask is None or mask.dtype != torch.bool or mask.shape != (batch, faces) or mask.device != x.device:
                raise ValueError("Colocated separate bool[B,padded_N] masks required")
        if bool((known_mask & ~valid_mask).any()):
            raise ValueError("Known faces must be valid")
        if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 799)).any()):
            raise ValueError("y must count real faces only, N128..799")
        if bool((known_mask.sum(1) == 0).any()) or bool(((valid_mask & ~known_mask).sum(1) == 0).any()):
            raise ValueError("Each sample requires known and free faces")
        hidden, role = self._embed(x, known_mask, valid_mask)
        memory = gather_known_corners(hidden, known_mask, valid_mask) if self.readout_mode == "clean" else None
        c_free = self.backbone.t_embedder(t)
        c_known = self.backbone.t_embedder(torch.ones_like(t))
        count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
        c_free, c_known = c_free + count, c_known + count
        face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
        if observer is not None:
            observer(dict(site="embedding", hidden=hidden, role=role,
                known_mask=known_mask, valid_mask=valid_mask, face_time=face_time,
                c_free=c_free, c_known=c_known))
            if memory is not None:
                observer(dict(site="clean_memory", memory=memory))
        for index, block in enumerate(self.backbone.layers, 1):
            hidden = native_block_forward(block, hidden, c_free, c_known, known_mask,
                valid_mask, observer=observer, layer_index=index, face_time=face_time)
            if str(index) in self.readouts:
                hidden = apply_readout(hidden, known_mask, valid_mask, self.readouts[str(index)],
                    memory=memory, observer=self.readout_observer, layer=index,
                    source_mode=self.readout_mode, **readout_execution_options())
        output = native_final_forward(self.backbone.final_layer, hidden, c_free,
            c_known, known_mask, observer=observer, face_time=face_time)
        return output.view(batch, faces, 9)

def build_model(model_config, readout_mode="none", readout_seed=20261007, trainable=True):
    """Construct the identical parameter structure, with this version's guard."""
    if dict(model_config) != MODEL_CONFIG or readout_mode != "none":
        raise ValueError("Only the unchanged Native no-readout architecture is registered")
    config = dict(model_config)
    config.pop("model_type")
    with torch.random.fork_rng(devices=[]):
        return NativeModel(DiT(**config), mode="none",
            readout_seed=readout_seed, trainable=trainable)


def edge_features(C_vertices, C_faces, total_N):
    """Legacy 24 fields for metadata only; no head consumes these features."""
    v, f = np.asarray(C_vertices, np.float32), np.asarray(C_faces, np.int64)
    edges = boundary_from_C(v, f)
    n, k = int(total_N), len(f)
    if not N_MIN <= n <= N_MAX or not 2 <= k < n:
        raise ValueError("Native N128..799 and 2<=K<N required")
    if not len(edges):
        return edges, np.empty((0, FEATURE_DIM), np.float32)
    center = (v.min(0) + v.max(0)) * .5
    span = np.ptp(v, axis=0)
    degree = np.bincount(edges[:, 1:].reshape(-1), minlength=len(v))
    features = []
    for fi, u, w in edges:
        a, b = v[u], v[w]
        delta = b-a
        normal = np.cross(v[f[fi,1]]-v[f[fi,0]], v[f[fi,2]]-v[f[fi,0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            raise ValueError("Zero-area C face")
        feature = np.r_[(a+b)*.5/L, delta/L, np.abs(delta)/L,
                        normal/norm, span/L, center/L,
                        np.linalg.norm(delta)/L, k/256., n/256.,
                        degree[u]/8., degree[w]/8., 1.]
        if feature.shape != (FEATURE_DIM,):
            raise ValueError("Unchanged metadata feature shape required")
        features.append(feature)
    return edges, np.asarray(features, np.float32)


def inference_layout(C_vertices, C_faces_original, C_faces_model, total_N, occupied=None):
    """Dynamic all-free Native layout; C is the only source of fixed corners."""
    v = np.asarray(C_vertices, np.float32)
    original, canonical = np.asarray(C_faces_original, np.int64), np.asarray(C_faces_model, np.int64)
    if original.shape != canonical.shape or not np.array_equal(np.sort(original, 1), np.sort(canonical, 1)):
        raise ValueError("Model C changes original faces")
    edges, features = edge_features(v, original, total_N)
    occupancy = np.zeros(len(edges), bool) if occupied is None else np.asarray(occupied)
    if occupancy.dtype != np.bool_ or occupancy.shape != (len(edges),) or occupancy.any():
        raise ValueError("Native all-free layout requires zero activated boundary slots")
    n, k = int(total_N), len(original)
    anchors = np.full((n, 3), -1, np.int64)
    anchors[:k] = canonical
    known = np.arange(n) < k
    return dict(anchor_ids=anchors, known_mask=known, valid_mask=np.ones(n, bool),
                free_corner_mask=anchors < 0, occupied=occupancy.copy(), edge_records=edges,
                features=features, free_coordinate_count=9 * (n-k),
                seam_slots=np.empty(0, np.int64), bulk_slots=np.arange(k, n, dtype=np.int64),
                C_vertices=v, C_faces_original=original, C_faces_model=canonical, N=n, K=k)


def supervised_sample(case, raw_noise, epsilon, t, permutation_seed, arm="PLAIN"):
    """Original Native bulk OT/HYBRID arithmetic at the case's natural N."""
    if arm != "PLAIN":
        raise ValueError("This experiment has no SEAM branch")
    full = np.asarray(case["full_target"], np.float32).reshape(-1,3,3)
    pre = np.asarray(case["pre_ot_full"], np.float32)
    known_ids, free_ids = np.asarray(case["known_ids"], np.int64), np.asarray(case["free_ids"], np.int64)
    n, k = len(full), len(known_ids)
    if not np.array_equal(pre*np.float32(2), full):
        raise ValueError("Original pre-OT half-scale contract differs")
    if not np.array_equal(np.sort(np.r_[known_ids, free_ids]), np.arange(n)):
        raise ValueError("Known and free source IDs must partition the complete target")
    z, eps = np.asarray(raw_noise), np.asarray(epsilon)
    if z.shape != (n-k,3,3) or eps.shape != z.shape or not np.isfinite(z).all() or not np.isfinite(eps).all():
        raise ValueError("Common underlying random arrays differ")
    if not 0 <= float(t) <= 1:
        raise ValueError("Invalid time")
    labels = np.asarray(case["occupied_labels"], bool)
    layout = inference_layout(case["C_vertices"], case["C_faces_original"], case["C_faces_model"], n)
    if labels.shape != layout["occupied"].shape:
        raise ValueError("Source labels must describe the same C boundary; never used as anchors")
    anchor = layout["anchor_ids"]
    target = np.zeros((n,3,3), np.float32)
    ids, corner = np.full(n,-1,np.int64), np.full((n,3),-1,np.int64)
    target[:k], ids[:k], corner[:k] = full[known_ids], known_ids, np.arange(3)
    bulk_slots = layout["bulk_slots"]
    bulk_faces = np.asarray([fi for fi in free_ids], np.int64)
    rng = np.random.RandomState(int(permutation_seed))
    bulk_faces = bulk_faces[rng.permutation(len(bulk_faces))]
    cp0 = np.asarray([[0,1,2],[1,2,0],[2,0,1]],np.int64)[rng.randint(0,3,len(bulk_faces))]
    face_map, corner_map = nested_ot_map(pre[bulk_faces[:,None],cp0],z[bulk_slots-k])
    ids[bulk_slots] = bulk_faces[np.argsort(face_map)]
    corner[bulk_slots] = to_noise_slots(cp0,face_map,corner_map)
    target[bulk_slots] = to_noise_slots(full[bulk_faces[:,None],cp0],face_map,corner_map)
    mixed = np.float32(np.sqrt(.75))*z.astype(np.float32) + np.float32(.5)*eps.astype(np.float32)
    x0 = np.zeros_like(target); x0[k:] = mixed
    fixed = anchor >= 0
    x0[fixed] = layout["C_vertices"][anchor[fixed]]
    if not np.array_equal(target[fixed], x0[fixed]):
        raise ValueError("Known C changed")
    velocity = target-x0
    xt = (np.float32(1)-np.float32(t))*x0+np.float32(t)*target
    xt[fixed] = x0[fixed]
    order = np.random.RandomState(int(permutation_seed)^0xD00D).permutation(n)
    result = dict(x0=x0[order],x1=target[order],xt=xt[order],u=velocity[order],
                anchor_ids=anchor[order],known_mask=layout["known_mask"][order],
                valid_mask=layout["valid_mask"][order],free_corner_mask=layout["free_corner_mask"][order],
                source_face_ids=ids[order],source_corner_permutations=corner[order],
                face_permutation=order,ot_face_map=face_map,ot_corner_map=corner_map,
                t=np.asarray(t,np.float32),C_vertices=layout["C_vertices"],
                free_coordinate_count=np.asarray(layout["free_coordinate_count"]),
                occupied_labels=labels,occupied_for_FM=np.zeros_like(labels),
                edge_features=layout["features"],N=np.asarray(n),K=np.asarray(k),
                arm=np.asarray("FULL4_C200"),production_OT_calls=np.asarray(1))
    if not np.array_equal(np.sort(ids),np.arange(n)) or not np.array_equal(
            full[result["source_face_ids"][:,None],result["source_corner_permutations"]], result["x1"]):
        raise ValueError("Complete source face/corner identity roundtrip failed")
    if result["u"][result["anchor_ids"]>=0].any():
        raise ValueError("C velocity must remain exactly zero")
    return result


def sample_for_training(case, row, arm="PLAIN"):
    copied = {key: value.copy() for key, value in case.items()}
    alpha = np.float32(row["alpha"])
    for key in ("full_target", "pre_ot_full", "C_vertices"):
        copied[key][...,0] *= alpha
    n, k = len(copied["full_target"]), len(copied["known_ids"])
    z = np.random.RandomState(row["noise_seed"]).standard_normal((n-k,3,3)).astype(np.float32)
    eps = np.random.RandomState(row["epsilon_seed"]).standard_normal(z.shape).astype(np.float32)
    normal = float(np.random.RandomState(row["time_seed"]).standard_normal())
    t = float(1/(1+np.exp(-normal)))
    return supervised_sample(copied,z,eps,t,row["permutation_seed"],arm)


def verify_sample(sample, stage="NATIVE"):
    n, k = int(sample["N"]), int(sample["K"])
    if stage != "NATIVE" or not N_MIN <= n <= N_MAX or not 2 <= k < n:
        raise ValueError("Only Native N128..799/K>=2 completion is registered")
    anchor = np.asarray(sample["anchor_ids"])
    fixed = anchor >= 0
    known = np.asarray(sample["known_mask"], bool)
    labels = np.asarray(sample["occupied_labels"], bool)
    active = np.asarray(sample["occupied_for_FM"], bool)
    if active.shape != labels.shape or active.any():
        raise ValueError("Source owners must not constrain Native free corners")
    if known.sum() != k or not np.array_equal(known, fixed.all(1)) or not np.asarray(sample["valid_mask"]).all():
        raise ValueError("Known/valid role masks differ")
    expected = 9*(n-k)
    if fixed[~known].any() or int(sample["free_coordinate_count"]) != expected or int((~fixed).sum())*3 != expected:
        raise ValueError("Native whole-batch free scalar denominator differs")
    return dict(source_owner_count=int(labels.sum()), active_seam_slots=0,
                C_boundary_edges=len(labels), closed_C=not len(labels),
                free_coordinate_count=expected, policy="NATIVE_ALL_FREE_OT")


def cpu_gate():
    """Finite CPU structural checks only: zero model calls and zero OT."""
    import ast, hashlib, inspect, json, textwrap
    from . import common, stream
    from r2.pilots.bench_parts_convergence_v1 import data as old_data
    from meshflow_control.models import native as old_native
    from meshflow_control.seam.model_adapter import corner_masked_fm_loss
    dest = common.OUT / "runtime_cpu_gate.json"
    source_paths = [Path(__file__),Path(__file__).with_name("training.py"),
        Path(__file__).with_name("stream.py"),Path(old_native.__file__),
        Path(old_data.__file__)]
    hashes = {str(p):common.sha(p) for p in source_paths}
    if dest.exists():
        prior = common.read(dest)
        if prior["source_hashes"] != hashes or prior["manifest_sha256"] != common.sha(common.OUT/"data/manifest.json"):
            raise ValueError("Prior CPU gate source/input changed; explicit gate revision required")
        return prior
    cuda_before = torch.cuda.is_initialized()
    if cuda_before:
        raise RuntimeError("CPU gate must not inherit an initialized CUDA context")
    torch.set_num_threads(2)
    rng_before = torch.get_rng_state().clone()
    old_text = textwrap.dedent(inspect.getsource(old_native.NativeModel.forward))
    new_text = textwrap.dedent(inspect.getsource(NativeModel.forward))
    expected = old_text.replace("(y > 256)", "(y > 799)").replace("N128..256", "N128..799")
    if ast.dump(ast.parse(expected)) != ast.dump(ast.parse(new_text)):
        raise ValueError("Native forward changed beyond the versioned N guard")
    with torch.device("meta"):
        old_model = old_native.build_model(dict(MODEL_CONFIG),trainable=False)
        new_model = build_model(dict(MODEL_CONFIG),trainable=False)
    describe = lambda model: [(n,tuple(v.shape),str(v.dtype)) for n,v in model.state_dict().items()]
    old_shapes, new_shapes = describe(old_model), describe(new_model)
    if old_shapes != new_shapes or sum(p.numel() for p in new_model.parameters()) != 130401155:
        raise ValueError("Native parameter names/shapes/capacity changed")
    if tuple(new_model.backbone.y_embedder.embedding_table.weight.shape) != (41,768):
        raise ValueError("Original face-count embedding shape changed")
    del old_model,new_model
    manifest = common.read(common.OUT/"data/manifest.json")
    schedules = {a:stream.make_schedule(manifest["train"],a) for a in stream.ARMS}
    stream_audit = stream.validate_pair(*(schedules[a] for a in stream.ARMS),manifest["train"])
    layouts = []
    protected = {}
    for row in manifest["train"] + manifest.get("high_holdout",[]):
        case = old_data.load_case(row)
        protected[str(common.resolve(row["case_path"]))] = common.sha(row["case_path"])
        layout = inference_layout(case["C_vertices"],case["C_faces_original"],case["C_faces_model"],int(row["N"]))
        if not np.array_equal(layout["C_vertices"][layout["C_faces_model"]],case["full_target"][case["known_ids"]]):
            raise ValueError("C layout coordinates differ from the source target")
        if layout["anchor_ids"][int(row["K"]):].max(initial=-1) != -1 or len(layout["seam_slots"]):
            raise ValueError("Native free faces acquired anchors")
        compared = False
        if int(row["N"]) <= 256:
            old = old_data.inference_layout(case["C_vertices"],case["C_faces_original"],case["C_faces_model"],
                int(row["N"]),np.zeros(len(case["occupied_labels"]),bool))
            if set(old) != set(layout) or any(not np.array_equal(old[k],layout[k]) for k in old):
                raise ValueError("Old-range layout differs: "+row["condition_id"])
            compared = True
        synthetic_sample = dict(N=layout["N"],K=layout["K"],anchor_ids=layout["anchor_ids"],
            known_mask=layout["known_mask"],valid_mask=layout["valid_mask"],
            occupied_labels=case["occupied_labels"],occupied_for_FM=layout["occupied"],
            free_coordinate_count=layout["free_coordinate_count"])
        verify_sample(synthetic_sample)
        layouts.append(dict(condition_id=row["condition_id"],N=int(row["N"]),K=int(row["K"]),
            free_scalars=layout["free_coordinate_count"],old_layout_exact=compared,model_calls=0))
    for bad_n in (127,800):
        try:
            inference_layout(case["C_vertices"],case["C_faces_original"],case["C_faces_model"],bad_n)
        except ValueError:
            pass
        else:
            raise ValueError("Out-of-contract N was accepted")
    pred = torch.ones((2,6,9),dtype=torch.float32)
    pred[1] *= 2
    pred.requires_grad_()
    target = torch.zeros_like(pred)
    valid = torch.tensor([[True,True,True,True,False,False],[True]*6])
    fixed = torch.zeros((2,6,3),dtype=torch.bool)
    fixed[0,:1] = True; fixed[1,:2] = True
    loss = corner_masked_fm_loss(pred,target,valid,fixed)
    parts = sum(corner_masked_fm_loss(pred[i:i+1],target[i:i+1],valid[i:i+1],fixed[i:i+1],denominator=63) for i in range(2))
    torch.testing.assert_close(loss,torch.tensor(171/63),rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(loss,parts)
    loss.backward()
    mask = (valid[...,None] & ~fixed)[...,None].expand(2,6,3,3).reshape(2,6,9)
    torch.testing.assert_close(pred.grad,torch.where(mask,2*pred.detach()/63,0.))
    if torch.cuda.is_initialized() or not torch.equal(torch.get_rng_state(),rng_before):
        raise ValueError("CPU structural gate changed CUDA initialization or global Torch RNG")
    result = dict(status="PASS",runtime_version=VERSION,N_contract=[N_MIN,N_MAX],
        forward_change="Only upper-N guard 256->799 and error string; normalized AST equal",
        same_parameter_names_shapes=True,generator_parameters=130401155,generator_tensors=len(new_shapes),
        count_embedding_shape=[41,768],meta_model_constructions=2,loaded_model_weights=0,
        model_forwards=0,production_OT_calls=0,new_scientific_noise=0,
        synthetic_loss_backward=1,synthetic_loss_scope="CPU tensor-only formula and microbatch-gradient test; no model",
        loss_normalization="whole_effective_batch_free_coordinates",stream=stream_audit,layouts=layouts,
        source_hashes=hashes,manifest_sha256=common.sha(common.OUT/"data/manifest.json"),protected=protected)
    common.write_new(dest,result)
    return result


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-gate",action="store_true",required=True)
    parser.parse_args()
    result = cpu_gate()
    print(json.dumps({k:result[k] for k in ("status","runtime_version","generator_parameters","model_forwards","production_OT_calls")}))


