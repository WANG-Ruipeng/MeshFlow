"""Explicit-manifest, all-attempt evaluation. No filename globs or hidden truncation.
Distribution metrics are computed ONLY when a full predeclared group is present.
"""
from pathlib import Path
import argparse,json,csv
import numpy as np
from scipy.spatial import cKDTree
from .registry import load_attempts,validate_internal
from .io_utils import new_dir,write_json,read_mesh,file_hash
from .mesh_ops import topology
from .legacy.exact_accel import audit
from .legacy.common import sample_surface,face_quality

def normalized_points(v,f,n=2048,seed=90123):
    centre=(v.min(0)+v.max(0))/2;scale=np.linalg.norm(np.ptp(v,axis=0))
    if not np.isfinite(v).all() or scale<=0:raise ValueError('invalid reference/sample scale')
    return sample_surface((v-centre)/scale,f,n,np.random.default_rng(seed))[0]

def cd(a,b):
    # Squared symmetric sum of mean nearest-point distances; no x1000 scaling here.
    return float(np.mean(cKDTree(b).query(a)[0]**2)+np.mean(cKDTree(a).query(b)[0]**2))

def distribution(real,generated):
    if len(real)!=len(generated):raise ValueError('Closed-set table uses equal16-object blocks, not silent truncation')
    n=len(real);C=np.array([[cd(a,b) for b in real] for a in generated])
    points=generated+real;D=np.zeros((2*n,2*n))
    for i in range(2*n):
        for j in range(i):D[i,j]=D[j,i]=cd(points[i],points[j])
    np.fill_diagonal(D,np.inf);lab=np.r_[np.zeros(n,dtype=int),np.ones(n,dtype=int)];nn=D.argmin(1)
    return {'MMD_CD':float(C.min(0).mean()),'coverage':float(len(np.unique(C.argmin(1)))/n),'one_NNA':float((lab[nn]==lab).mean()),'n_each':n,'nearest_train_distance_mean':float(C.min(1).mean()),'CD_definition':'sum of both squared mean NN distances, bbox diagonal normalization','closed_set_not_holdout':True},C,D

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--ours',required=True);p.add_argument('--meshflow');p.add_argument('--out',required=True);p.add_argument('--exact',action='store_true');p.add_argument('--points',type=int,default=2048);a=p.parse_args()
    out=new_dir(a.out);ours=Path(a.ours);pair=validate_internal({k:ours/f'{k}_MANIFEST.json' for k in ['fixed','a','qem']});write_json(out/'PAIRING_GATE.json',pair)
    data=Path(a.data);meta=json.loads((data/'DATASET.json').read_text());ref=[]
    for r in meta['rows']:
        v,f=read_mesh(data/'raw'/f'{r["uid"]}.npz');ref.append(normalized_points(v,f,a.points))
    paths=[ours/f'{k}_MANIFEST.json' for k in ['fixed','a','qem']]
    if a.meshflow:paths=[Path(a.meshflow)/'raw_MANIFEST.json',Path(a.meshflow)/'post_MANIFEST.json']+paths
    summary=[];per=[];groups=[]
    for path in paths:
        m=load_attempts(path)
        if any(r.get('dataset_manifest_sha256')!=file_hash(data/'DATASET.json') for r in m['records']):raise ValueError('A row was trained/exported on a different dataset manifest')
        clouds=[];gates=[];times=[]
        for r in m['records']:
            row={'row':m['row'],'sample_id':r['sample_id'],'status':r['status']};cl=None
            try:
                if r['status']!='ok':raise ValueError(r.get('error','upstream failed'))
                v,f=read_mesh(path.parent/r['mesh_path']);tp=topology(v,f);ex=audit(v,f) if a.exact else None
                row.update(actual_faces=len(f),budget_match=len(f)==512,topology=tp,quality=face_quality(v,f),exact=ex)
                valid=tp['closed_oriented_g0'] and (ex['embedded_endpoint_pass'] if ex else False)
                row['joint_valid_exact']=bool(valid);cl=normalized_points(v,f,a.points);times.append(r.get('seconds_noise_to_mesh'))
            except Exception as e:row.update(metric_status='failed',metric_error=repr(e))
            clouds.append(cl);per.append(row);gates.append(row)
        for offset in range(0,len(clouds),16):
            block=clouds[offset:offset+16]
            if len(block)!=16 or any(x is None for x in block):groups.append({'row':m['row'],'offset':offset,'status':'unavailable_incomplete_or_failed_group'});continue
            metrics,C,D=distribution(ref,block);groups.append({'row':m['row'],'offset':offset,'status':'ok',**metrics});np.savez_compressed(out/f'{m["row"]}_group{offset:04d}.npz',cross_distance=C,all_distance=D)
        gg=[g for g in groups if g['row']==m['row'] and g['status']=='ok']
        summary.append({'row':m['row'],'attempts':len(clouds),'metrics_readable':sum(c is not None for c in clouds),'budget512_count':sum(g.get('budget_match',False) for g in gates),'joint_valid_exact_count':sum(g.get('joint_valid_exact',False) for g in gates) if a.exact else None,'MMD_CD':float(np.mean([g['MMD_CD'] for g in gg])) if gg else None,'coverage':float(np.mean([g['coverage'] for g in gg])) if gg else None,'one_NNA':float(np.mean([g['one_NNA'] for g in gg])) if gg else None,'valid_metric_groups':len(gg),'not_heldout':True,'paper_full_post':False if 'public_post' in m['row'] else None,'full_publication_ranking_qualified':False})
    write_json(out/'PER_SAMPLE.json',per);write_json(out/'DISTRIBUTION_GROUPS.json',groups);write_json(out/'MATRIX.json',summary)
    with (out/'MATRIX.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    write_json(out/'SCOPE.json',{'closed_set_training_reference':True,'no_test_generalization':True,'sampling_groups_not_independent_training_runs':True,'shared_mainfield_error_and_fwhm_not_computed_here':'Codex task extends aligned-layout metrics; not a complete paper evaluator','exact_scope':'all supplied output meshes' if a.exact else 'not run','missing_external_rows':not bool(a.meshflow)})
if __name__=='__main__':main()
