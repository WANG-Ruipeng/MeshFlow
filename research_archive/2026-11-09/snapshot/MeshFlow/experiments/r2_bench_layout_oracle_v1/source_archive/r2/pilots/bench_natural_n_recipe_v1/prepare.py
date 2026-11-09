"""Freeze paired data streams without model calls."""
import json
from .common import *
from . import stream


def prepare():
    m = read(OUT/'data/manifest.json')
    schedules = []
    for arm in ARMS:
        value = stream.make_schedule(m['train'],arm)
        dest = OUT/'data'/('schedule_'+arm+'.json')
        if dest.exists():
            if read(dest) != value: raise ValueError('Existing stream differs')
        else: write_new(dest,value)
        schedules.append(value)
    gate = stream.validate_pair(*schedules,m['train'])
    gate.update(CPU_only=True,model_calls=0,manifest_sha256=sha(OUT/'data/manifest.json'))
    dest = OUT/'data/stream_gate.json'
    if dest.exists():
        if read(dest) != gate: raise ValueError('Existing stream gate differs')
    else: write_new(dest,gate)
    return gate


if __name__ == '__main__':
    r=prepare(); print(json.dumps(r))
