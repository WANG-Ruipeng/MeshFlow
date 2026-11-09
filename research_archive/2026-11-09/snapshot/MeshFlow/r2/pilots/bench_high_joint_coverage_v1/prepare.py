"""Freeze one real wide-pool stream after data admission, without model calls."""
import json
from .common import *
from . import stream


def prepare():
    m=read(OUT/'data/manifest.json');pool=stream.validate_manifest(m)
    value=stream.make_schedule(m['train'],'WIDE_HIGH')
    gate=stream.validate_frozen_narrow(value,m['train'])
    for path,obj in ((OUT/'data/schedule_WIDE_HIGH.json',value),):
        if path.exists():
            if read(path)!=obj:raise ValueError('Existing stream differs')
        else:write_new(path,obj)
    gate.update(pool=pool,CPU_only=True,model_calls=0,OT_calls=0,manifest_sha256=sha(OUT/'data/manifest.json'),
                schedule_sha256=sha(OUT/'data/schedule_WIDE_HIGH.json'))
    dest=OUT/'data/stream_gate.json'
    if dest.exists():
        if read(dest)!=gate:raise ValueError('Existing stream gate differs')
    else:write_new(dest,gate)
    return gate


if __name__=='__main__':print(json.dumps(prepare()))
