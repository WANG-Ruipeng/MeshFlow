from pathlib import Path
import json, hashlib
root=Path("/mnt/e/MeshFlow")
p=root/"native_t1/geometry_checkpoint.py"
t=p.read_text(encoding="utf-8")
t=t.replace("import time\n", "import time\nimport re\nfrom collections.abc import Mapping\n",1)
needle='    return context\n'
insert='''    for key in ("optimizer","stream_state"):
        if not isinstance(payload.get(key),Mapping):
            raise ValueError("Geometry checkpoint requires "+key)
    if (payload.get("stream_seed") != 1010 or payload["stream_state"].get("seed") != 1010
            or payload["stream_state"].get("arm") != "Native_correct"
            or payload["stream_state"].get("batch_index") != payload["completed_updates"]
            or payload["stream_state"].get("sample_index") != 8*payload["completed_updates"]):
        raise ValueError("Registered new input stream identity/progress differs")
    hashes=[payload.get(k) for k in ("input_sha256","config_sha256","model_state_sha256")]
    hashes += [payload["base_identity"].get(k) for k in ("checkpoint_sha256","state_sha256")]
    if any(not isinstance(v,str) or re.fullmatch(r"[0-9a-f]{64}",v) is None for v in hashes):
        raise ValueError("Checkpoint hash identity is incomplete")
    return context
'''
assert needle in t
t=t.replace(needle,insert,1);p.write_text(t,encoding="utf-8")
p=root/"native_t1/geometry_generation.py"
t=p.read_text(encoding="utf-8")
t=t.replace('    registration=read_json(args.out/"registration.json")\n',
'''    registration=read_json(args.out/"registration.json")
    root=Path(__file__).resolve().parents[1]
    if any(file_sha256(root/name)!=digest for name,digest in registration["execution_sources"].items()):
        raise RuntimeError("Execution code changed after registration")
    from .baseline import BASELINE_CHECKPOINT
    if file_sha256(args.checkpoint or BASELINE_CHECKPOINT)!=registration["initialization_file_sha256"]:
        raise RuntimeError("Generation initialization changed after registration")
''',1)
t=t.replace('            model,audit=_load(args,name)\n',
'''            model,audit=_load(args,name)
            if name=="T0":
                if audit["checkpoint_sha256"]!=registration["initialization_file_sha256"]:
                    raise RuntimeError("T0 is not the registered common initialization")
            elif audit["cumulative_updates"]!=2000:
                raise RuntimeError("Generation endpoint is not cumulative2000")
            if name in ("T_geo","T_graph"):
                if (audit["context"]["context_mode"]!=name.split("_",1)[1]
                        or audit["base_identity"]["checkpoint_sha256"]!=registration["initialization_file_sha256"]):
                    raise RuntimeError("Geometry endpoint architecture/common initialization differs")
''',1)
p.write_text(t,encoding="utf-8")
reg=root/"native_t1/runs/geometry_context_20260930_142000/registration.json"
data=json.loads(reg.read_text())
changes=[]
for name,old in data["execution_sources"].items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=old:
        changes.append(dict(path=name,before=old,after=actual))
        data["execution_sources"][name]=actual
data["pre_gpu_code_amendments"]=[dict(reason="Tighten checkpoint stream/hash and generation identity guards after independent review, before any GPU model call or update; mathematical choices/data/seeds/budgets unchanged.",files=changes)]
reg.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({"status":"PASS","identity_guard_files":changes},indent=2))
