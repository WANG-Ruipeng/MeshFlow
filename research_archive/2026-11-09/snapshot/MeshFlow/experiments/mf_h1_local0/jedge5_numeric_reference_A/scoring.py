"""Window A: read-only frozen evaluator, isolated CPU score outputs."""
from pathlib import Path
import argparse
import json
import os
import time
import numpy as np
from .assets import ROOT, EVALUATION, EVALUATION_SHA, read_json, write_json, file_sha256
from experiments.mf_h1_local0.jedge5_diagnosis import scoring as archived


def own_out(value):
    out = Path(value).resolve()
    if out.parent != ROOT/'native_t1/runs' or not out.name.startswith('jedge5_numeric_reference_A_'):
        raise ValueError('Writes must remain inside the isolated window A run')
    return out


def initialize(out):
    """Copy the verified, immutable CC audit; do not launch an extra geometry matrix."""
    import fcntl
    out = own_out(out)
    folder = out/'registration'
    with (folder/'scoring_init.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        protocol = read_json(folder/'protocol.json')
        source = Path(protocol['source_archive'])/'registration/static_context_audit.json'
        cache = folder/'static_context_audit.json'
        if not cache.exists():
            cache.write_bytes(source.read_bytes())
        if file_sha256(cache) != file_sha256(source):
            raise ValueError('The copied static CC cache differs from the archive')
        banks = archived.banks_with_context_audit(out)
        hashes = {}
        for path in (Path(archived.__file__).resolve(), archived.REFERENCE/'evaluation.py',
                     archived.REFERENCE/'edge_metric.py', source, EVALUATION):
            hashes[str(path)] = file_sha256(path)
        record = dict(status='PASS',static_context_audit_source=str(source),
                      static_context_audit_sha256=file_sha256(source),
                      frozen_evaluation_sha256=EVALUATION_SHA,read_source_sha256=hashes,
                      copied_static_audit=True,new_CC_audits=0,model_forwards=0,
                      note='Same registered surface/Gamma/coverage points and unchanged independent evaluator')
        if not (folder/'scoring_identity.json').exists():
            write_json(folder/'scoring_identity.json', record)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return banks


def score_one_cli(out, trajectory):
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Offline score worker must be CPU-only')
    out = own_out(out)
    trajectory = Path(trajectory)
    if not trajectory.is_absolute():
        trajectory = out/trajectory
    trajectory = trajectory.resolve()
    if out not in trajectory.parents:
        raise ValueError('Trajectory result must be in this run')
    row = read_json(trajectory)
    ident = trajectory.parent.name
    dest = out/'scores'/ident
    result_path = dest/'result.json'
    if result_path.exists():
        raise FileExistsError(result_path)
    if not row.get('complete') or row.get('repeat') or row.get('is_repeat') or 'repeat' in ident.lower():
        write_json(result_path,dict(status='NOT_RUN',reason='Only complete formal endpoints are scored',
                                   trajectory=str(trajectory.relative_to(out)),row=row,model_forwards=0))
        return read_json(result_path)
    banks = initialize(out)
    raw_path = Path(row['path'])
    if not raw_path.is_absolute():
        raw_path = out/raw_path
    if out not in raw_path.resolve().parents:
        raise ValueError('Raw output is outside window A run')
    if file_sha256(raw_path) != row['sha256']:
        raise ValueError('Generated raw SHA differs')
    with np.load(raw_path,allow_pickle=False) as z:
        raw = z['raw'].copy().reshape(112,3,3)
        C = z['C'].copy().reshape(12,3,3)
        states = z['states'].copy().reshape(-1,112,3,3)
    if raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('Endpoint is not finite FP32')
    if not np.array_equal(raw[:12].view(np.uint32), C.view(np.uint32)):
        raise ValueError('Endpoint changed C')
    if not np.array_equal(states[:,:12].view(np.uint32),np.broadcast_to(C,(len(states),12,3,3)).copy().view(np.uint32)):
        raise ValueError('Saved fixed-time state changed C')
    tick = time.perf_counter(); ledger = {}
    metrics, scalars = archived.score_one(raw,banks[int(row['patch'])],int(row['parent']),int(row['patch']),ledger)
    detail = dest/'metrics.json'
    write_json(detail, metrics)
    result = dict(status='PASS',trajectory=str(trajectory.relative_to(out)),
                  row=row,scalars=scalars,cross_remainder=metrics['remainder']['cross'],
                  metric_path=str(detail.relative_to(out)),metric_sha256=file_sha256(detail),
                  seconds=time.perf_counter()-tick,ledger=ledger,
                  C_saved_states_bitwise=True,full_faces=112,free_faces=100,
                  model_forwards=0,backward_calls=0,OT_calls=0,repairs=0,
                  scoring_version_sha256=file_sha256(archived.REFERENCE/'evaluation.py'),
                  evaluation_inputs_sha256=EVALUATION_SHA)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU score unexpectedly initialized CUDA')
    write_json(result_path,result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--trajectory',type=Path)
    p.add_argument('--initialize',action='store_true')
    a=p.parse_args(argv)
    if a.initialize:
        initialize(a.out)
        print(json.dumps(dict(status='PASS',action='initialize',model_forwards=0)))
    else:
        if a.trajectory is None: p.error('--trajectory is required unless --initialize')
        r=score_one_cli(a.out,a.trajectory)
        print(json.dumps(dict(status=r['status'],trajectory=str(a.trajectory),seconds=r.get('seconds')),allow_nan=False),flush=True)

if __name__=='__main__':main()
