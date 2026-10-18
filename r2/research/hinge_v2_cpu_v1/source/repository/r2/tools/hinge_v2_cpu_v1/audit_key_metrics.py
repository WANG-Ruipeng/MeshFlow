"""Independent saved-graph reach and fixed-parent conflict accounting.

No triangle contact detection, scoring API, model, or new sample is invoked.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import json
from pathlib import Path
import time
import numpy as np
from .diagnose import rows,read,arrays,sha,write_csv

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_hinge_v2_cpu_v1'
BASE=ROOT/'experiments/r2_hinge_opposite_v1/cells/HINGE_OPP015'
STAGES=('BASELINE','MIDPOINT_REPLAY','ALPHA','HOLE','ALPHA_HOLE')
ILLEGAL={'proper_transverse_penetration','coplanar_positive_area_overlap'}


def run():
    tick=time.perf_counter();source=OUT/'scores/per_output.csv';table=rows(source)
    lookup={(r['stage'],r['request_id']):r for r in table};assert len(lookup)==len(table)==140
    ids=sorted({r['request_id'] for r in table});assert len(ids)==28
    checks=0;face_ids={};records=[]
    def eq(a,b,context):
        nonlocal checks
        assert np.isclose(float(a),float(b),rtol=1e-12,atol=1e-12),(context,a,b)
        checks+=1
    for stage in STAGES:
        for rid in ids:
            row=lookup[stage,rid];cell=BASE/rid if stage=='BASELINE' else OUT/'cells'/stage/rid
            m=arrays(cell/'post.npz');n=len(m['faces']);K=int(m['known_faces'])
            roles=m.get('face_role')
            if roles is None:
                roles=np.asarray(['C' if i<K else 'SEAM' if m['raw_seam_slot_mask'][m['source_face_ids'][i]] else 'BULK' for i in range(n)])
            roles=roles.astype(str);bridge=roles=='BRIDGE';bulk=roles=='BULK'
            parent=m.get('parent_post_face_id',np.arange(n));P=int(row['original_POST_free_faces'])
            assert np.sum(~bridge)-K==P
            edges=rows(OUT/'scores'/stage/rid/'valid_edges.csv');adj=collections.defaultdict(set);seeds=set()
            for e in edges:
                if e['valid_edge'].lower()!='true':continue
                i,j=map(int,json.loads(e['incident_face_ids']))
                if min(i,j)<K:
                    fi=max(i,j)
                    if roles[fi]=='SEAM':seeds.add(fi)
                else:adj[i].add(j);adj[j].add(i)
            reached=set(seeds);todo=list(seeds)
            while todo:
                i=todo.pop()
                for j in adj[i]-reached:reached.add(j);todo.append(j)
            rb=sorted(i for i in reached if bulk[i]);rs=sorted(i for i in reached if roles[i]=='SEAM')
            rrbridge=sorted(i for i in reached if bridge[i]);tri=m['output'].astype(float)
            area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
            numerator=float(area[rb].sum());denominator=float(area[bulk].sum());fraction=numerator/denominator
            for key,value in (('free_only_reached_bulk_faces',len(rb)),('free_only_reached_seam_faces',len(rs)),
                ('free_only_reached_bulk_area',numerator),('retained_bulk_area',denominator),
                ('free_only_reached_bulk_area_fraction',fraction),('retained_bulk_faces',int(bulk.sum())),
                ('bulk_reach_ge_0_30_descriptive',int(fraction>=.30)),
                ('free_only_reached_bridge_faces',len(rrbridge)),('bridge_face_count',int(bridge.sum())),
                ('bridge_area',float(area[bridge].sum()))):eq(row[key],value,(stage,rid,key))
            face_ids[stage,rid]=set(rb)
            pairs=read(OUT/'scores'/stage/rid/'pairs.json')
            for kind in ('CF','FF'):
                affected=set();original_affected=set();original_pairs=set();bridge_pairs=[];count=0
                for p in pairs[kind]['contacts']:
                    if p['classification'] not in ILLEGAL:continue
                    count+=1;i,j=map(int,p['pair']);j+=K
                    if kind=='FF':i+=K;affected.add(i)
                    affected.add(j)
                    for z in ([j] if kind=='CF' else [i,j]):
                        if not bridge[z]:original_affected.add(int(parent[z]))
                    if bridge[i] or bridge[j]:bridge_pairs.append((i,j))
                    else:original_pairs.add(tuple(sorted((int(parent[i]),int(parent[j])))))
                assert not bridge_pairs,(stage,rid,'illegal bridge')
                for key,value in ((kind+'_illegal_pair_count',count),
                    (kind+'_illegal_affected_free_fraction',len(affected)/(n-K)),
                    ('fixed_parent_'+kind+'_illegal_affected_count',len(original_affected)),
                    ('fixed_parent_'+kind+'_illegal_affected_fraction',len(original_affected)/P),
                    ('fixed_parent_'+kind+'_illegal_unique_pair_count',len(original_pairs)),
                    ('bridge_'+kind+'_illegal_pair_count',len(bridge_pairs))):eq(row[key],value,(stage,rid,key))
            records.append(dict(stage=stage,request_id=rid,split=row['split'],reached_original_bulk_face_ids=rb,
                reached_original_bulk_count=len(rb),original_bulk_area=numerator,original_bulk_denominator=denominator,
                bulk_reach_fraction=fraction,bulk_reach_ge30=fraction>=.30,bridge_faces=int(bridge.sum()),bridge_area=float(area[bridge].sum())))
    summaries=[];comparisons=[]
    for split in ('TRAIN','DEV','ALL'):
        group=[rid for rid in ids if split=='ALL' or lookup['BASELINE',rid]['split']==split]
        reference={rid for rid in group if float(lookup['BASELINE',rid]['free_only_reached_bulk_area_fraction'])>=.30}
        for stage in STAGES:
            data=[lookup[stage,rid] for rid in group]
            positives={rid for rid in group if float(lookup[stage,rid]['free_only_reached_bulk_area_fraction'])>=.30}
            result=dict(split=split,stage=stage,outputs=len(group),bulk_reach_ge30=len(positives),positive_identities=sorted(positives),
                gained_positive_identities=sorted(positives-reference),lost_positive_identities=sorted(reference-positives),
                bridge_faces=sum(int(r['bridge_face_count']) for r in data))
            for key in ('FF_illegal_affected_free_fraction','fixed_parent_FF_illegal_affected_fraction','CF_illegal_affected_free_fraction',
                'fixed_parent_CF_illegal_affected_fraction','ConflictCov_C','free_only_reached_bulk_area_fraction','free_minimum_angle_lt5_fraction'):
                result[key+'_macro_mean']=float(np.mean([float(r[key]) for r in data]))
            result['FF_illegal_pairs_total']=sum(int(r['FF_illegal_pair_count']) for r in data)
            result['FF_fixed_affected_total']=sum(int(r['fixed_parent_FF_illegal_affected_count']) for r in data)
            summaries.append(result)
    for stage in STAGES[1:]:
        for rid in ids:
            a,b=lookup['BASELINE',rid],lookup[stage,rid]
            comparisons.append(dict(stage=stage,request_id=rid,split=b['split'],
                new_reached_original_bulk_faces=sorted(face_ids[stage,rid]-face_ids['BASELINE',rid]),
                lost_reached_original_bulk_faces=sorted(face_ids['BASELINE',rid]-face_ids[stage,rid]),
                actual_FF_delta=float(b['FF_illegal_affected_free_fraction'])-float(a['FF_illegal_affected_free_fraction']),
                fixed_parent_FF_delta=float(b['fixed_parent_FF_illegal_affected_fraction'])-float(a['fixed_parent_FF_illegal_affected_fraction']),
                bridge_faces=int(b['bridge_face_count'])))
    dest=OUT/'independent_audit';dest.mkdir(parents=True,exist_ok=True)
    for name,data in (('key_metrics_summary',summaries),('original_bulk_reached_faces',records),('key_metrics_paired',comparisons)):
        write_csv(dest/(name+'.csv'),data)
    result=dict(status='PASS_INDEPENDENT_KEY_METRICS',outputs=140,numeric_checks=checks,
        original_bulk_graph_reconstructed_outputs=140,bridge_illegal_pairs=0,
        summary=summaries,source_sha256=sha(source),code_sha256=sha(__file__),
        seconds=time.perf_counter()-tick,new_model_calls=0,new_contact_scans=0,
        distinction='Reach uses original BULK face identities/areas only; bridge faces are traversable but excluded. Actual-face FF and original-parent FF independently reconstructed.')
    (OUT/'independent_key_metrics_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='summary'},indent=2));return result


def main():return run()
if __name__=='__main__':main()
