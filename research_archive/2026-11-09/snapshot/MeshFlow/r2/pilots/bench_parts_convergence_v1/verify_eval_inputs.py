"""CPU layout/decode roundtrip for actual frozen C-only requests, no forward."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import time
import numpy as np
from . import panel as P, data as D
from r2.pilots.seam_slot_c200_v1.generate import raw_arrays


def run(root=P.OUT):
    root=P.resolve(root);start=time.perf_counter();m=P.freeze(root);probe=P.read(root/'fixed_loss_probe/manifest.json')
    for name in ('panel.py','evaluation.py','sample.py','postprocess.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    unique={q['condition_id']:q for q in m['requests']};records=[]
    for q in unique.values():
        inp=D.arrays(q['inference_npz']);source=D.arrays(q['source_npz']);noise=D.arrays(q['noise_npz'])['z']
        edges,features=D.edge_features(inp['C_vertices'],inp['C_faces_original'],int(inp['N']))
        if len(edges)!=q['boundary_edges'] or features.shape!=(len(edges),24):raise AssertionError('C-only features differ')
        for activation in (False,True):
            occupied=np.full(len(edges),activation,bool)
            layout=D.inference_layout(inp['C_vertices'],inp['C_faces_original'],inp['C_faces_model'],int(inp['N']),occupied)
            n,k=layout['N'],layout['K'];fixed=layout['anchor_ids']>=0
            state=np.zeros((n,3,3),np.float32);state[k:]=noise;state[fixed]=layout['C_vertices'][layout['anchor_ids'][fixed]]
            saved=raw_arrays(state,layout,np.zeros(len(edges),np.float32),source)
            original=inp['C_vertices'][inp['C_faces_original']]
            if saved['output'][:k].tobytes()!=original.tobytes():raise AssertionError('Original C restoration changed bytes')
            if saved['vertices'][saved['faces']].tobytes()!=saved['output'].tobytes():raise AssertionError('Indexed roundtrip differs')
            if len(edges)==0 and (layout['seam_slots'].size or saved['raw_seam_slot_mask'].any()):raise AssertionError('Closed C created seams')
            records.append(dict(condition_id=q['condition_id'],closed_C=len(edges)==0,activation='ALL_ON' if activation else 'ALL_OFF',
                edges=len(edges),N=n,K=k,C_exact=True,indexed_roundtrip=True,seam_slots=len(layout['seam_slots'])))
    if len(records)!=48 or len(probe['rows'])!=96:raise AssertionError('Budget identity differs')
    report=dict(status='PASS',scope='Actual input-only layout and shared-index serialization; no generated model result',
        layouts=records,condition_count=24,closed_conditions=sum(q['closed_C'] for q in unique.values()),
        loss_probe_rows=96,model_constructions=0,model_forwards=0,GPU_work=0,training_updates=0,
        code_sha256=P.sha(__file__),elapsed_seconds=time.perf_counter()-start,
        protected={str(root/'evaluation_inputs/manifest.json'):P.sha(root/'evaluation_inputs/manifest.json'),
                   str(root/'fixed_loss_probe/manifest.json'):P.sha(root/'fixed_loss_probe/manifest.json')})
    path=root/'audits/evaluation_inputs_cpu_roundtrip.json'
    if path.exists():
        old=P.read(path)
        if old['layouts']!=records or old['protected']!=report['protected']:raise AssertionError('Prior input roundtrip differs')
        return old
    P.immutable(path,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(P.OUT));a=p.parse_args();r=run(a.root)
    print(r['status'],r['condition_count'],'C; no model calls')
