"""Prepare finite manifests without model calls or asset mutation."""
from __future__ import annotations
import json
from .common import *
from . import stream


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    source=SOURCE_ROOT/'data/manifest.json'
    m=read(source)
    target=OUT/'data/manifest.json'
    if target.exists():
        if read(target)!=m:raise ValueError('Existing branch data differs')
    else:write_new(target,m)
    audits={}
    for arm in ARMS:
        value=stream.make_schedule(m['train'],arm)
        dest=OUT/'data'/('schedule_'+arm+'.json')
        if dest.exists():
            if read(dest)!=value:raise ValueError('Existing schedule differs')
        else:write_new(dest,value)
        audits[arm]=stream.validate(value,m['train'],1500,arm)
    gate=dict(status='PASS',source_manifest=str(source),source_manifest_sha256=sha(source),
        copied_manifest_sha256=sha(target),same_cases_and_coordinates=True,arms=audits,
        CPU_only=True,model_calls=0)
    for r in m.get('rows',m['train']):
        if sha(r['case_path'])!=r['case_sha256']:raise ValueError('Source case changed')
    dest=OUT/'data/cpu_gate.json'
    if not dest.exists():write_new(dest,gate)
    elif read(dest)!=gate:raise ValueError('Data preparation changed')
    return gate


if __name__=='__main__':
    r=prepare();print(json.dumps({'status':r['status'],'arms':r['arms']}))
