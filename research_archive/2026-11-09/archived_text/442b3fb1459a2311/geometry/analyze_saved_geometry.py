"""Read-only attribution of saved CAPPED500 distances and control witnesses.

No generated mesh modification, query resampling, model import or metric rewrite.
Only new distances are original frozen GT queries to original C, used for fixed
descriptive spatial bins. Original model-error distances are read verbatim.
"""
from pathlib import Path
import csv,hashlib,json,os,time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
import numpy as np
from experiments.mf_h1_local0.evaluate import surface_distance_squared

ROOT=Path('/mnt/e/MeshFlow')
SRC=ROOT/'native_t1/runs/chair_control_feedback_auto_v1'
OUT=ROOT/'native_t1/runs/chair_capped500_offline_v1/geometry'
LABELS={'START':'START','teacher500':'explore_REPLAY_FM_CONTROL_500','TR500':'explore_TRAJECTORY_REPLAY_500','cap500':'explore_TRAJECTORY_REPLAY_CAPPED_500','cap1000':'explore_TRAJECTORY_REPLAY_CAPPED_1000'}
PAIRS=[('cap500','START'),('cap500','teacher500'),('cap500','TR500'),('cap1000','cap500')]
sources={};checks=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):
    p=Path(p);h=sha(p)
    if str(p) in sources:assert sources[str(p)]==h
    else:sources[str(p)]=h
    return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def npz(p):
    with np.load(track(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def check(name,a,b):
    assert np.allclose(a,b,atol=2e-12,rtol=1e-10), (name,a,b)
    checks.append(name)
def bins(points,condition):
    d=np.sqrt(surface_distance_squared(points,condition['C']))/float(condition['L0'])
    verts=condition['GT_full'].reshape(-1,3);lo=verts[:,1].min();span=np.ptp(verts[:,1])
    height=(points[:,1]-lo)/span
    return d,np.where(d<=.01,'C_0_1pct',np.where(d<=.05,'C_1_5pct','C_over5pct')),np.where(height<1/3,'Y_lower_third',np.where(height<2/3,'Y_middle_third','Y_upper_third'))
def summarize(arr):
    dist=np.sqrt(arr);order=np.argsort(arr);n=len(arr)
    return dict(mse=float(arr.mean()),mean_distance=float(dist.mean()),median_distance=float(np.median(dist)),p95_distance=float(np.quantile(dist,.95)),max_distance=float(dist.max()),tail_top5_mse_contribution=float(arr[order[-int(np.ceil(.05*n)):]].sum()/n),tail_top10_mse_contribution=float(arr[order[-int(np.ceil(.1*n)):]].sum()/n),coverage_1pct=float(np.mean(dist<=.01)))

def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    requests=js(SRC/'manifests/DEV.json')['rows']
    scored={k:{r['request_id']:r for r in js(SRC/'evaluation'/v/'DEV/per_output.json')} for k,v in LABELS.items()}
    distance_rows=[];region_rows=[];pair_rows=[];points_rows=[];control_rows=[];edge_rows=[];face_rows=[];raw_rows=[]
    fixed={}
    for req in requests:
        rid= req['id'];ident=dict(request_id=rid,uid=req['uid'],noise_index=req['noise_index'])
        cid=req['condition_id']
        if cid not in fixed:
            c=npz(req['condition_npz']);q=npz(SRC/'references'/cid/'quality.npz');ref=npz(SRC/'references'/cid/'control.npz')
            bank={512:c['legacy_query_points'],2048:q['Q_Y2048']}
            fixed[cid]=(c,ref,bank,{n:bins(x,c) for n,x in bank.items()})
        c,ref,bank,partitions=fixed[cid];L=float(c['L0'])
        quality={k:npz(SRC/'evaluation'/v/'DEV/scoring/vectors'/(rid+'.npz')) for k,v in LABELS.items()}
        controls={k:npz(SRC/'evaluation'/v/'DEV/scoring/control'/(rid+'.npz')) for k,v in LABELS.items()}
        for label,v in LABELS.items():
            metric=scored[label][rid];qv=quality[label]
            check(rid+label+'MSE512',qv['legacy_query_d2'].mean()/L**2,metric['fixed_query_MSE'])
            check(rid+label+'GT2048',qv['GT_to_F_d2_2048'].mean()/L**2,metric['GT_to_F_MSE2048_L0'])
            check(rid+label+'F2048',qv['F_to_GT_d2_2048'].mean()/L**2,metric['F_to_GT_MSE2048_L0'])
            check(rid+label+'Dsym',(qv['GT_to_F_d2_2048'].mean()+qv['F_to_GT_d2_2048'].mean())/(2*L**2),metric['D_sym2048'])
            for direction,key in [('GT512','legacy_query_d2'),('GT2048','GT_to_F_d2_2048'),('F2048','F_to_GT_d2_2048')]:
                distance_rows.append(dict(ident,label=label,direction=direction,**summarize(qv[key]/L**2)))
            raw=npz(SRC/'runtime'/v/rid/'raw.npz');m=raw['known_mask'];t=raw['output'].reshape(-1,3,3);f=t[~m]
            assert np.array_equal(t[m],c['C']) and len(t)==req['N']
            verts=f.reshape(-1,3);gt=c['GT_full'].reshape(-1,3);lo=gt.min(0);hi=gt.max(0)
            area=.5*np.linalg.norm(np.cross(f[:,1]-f[:,0],f[:,2]-f[:,0]),axis=1)
            raw_rows.append(dict(ident,label=label,free_faces=len(f),free_area=float(area.sum()),area_over_GT=float(metric['free_area_over_GT_free_area']),free_vertex_fraction_outside_GT_AABB=float(np.mean(np.any((verts<lo-.01*L)|(verts>hi+.01*L),axis=1))),minimum_y=float(verts[:,1].min()),maximum_y=float(verts[:,1].max()),condition_preserved=True))
        for n,pts in bank.items():
            qkey='legacy_query_d2' if n==512 else 'GT_to_F_d2_2048'
            values={label:q[qkey]/L**2 for label,q in quality.items()};cd,cb,yb=partitions[n]
            for i,point in enumerate(pts):
                points_rows.append(dict(ident,query_count=n,point_index=i,x=float(point[0]),y=float(point[1]),z=float(point[2]),distance_to_C_L=float(cd[i]),C_band=cb[i],Y_bin=yb[i],**{label+'_d2_L2':float(x[i]) for label,x in values.items()}))
            for partition,tags in [('C_distance',cb),('GT_Y_thirds',yb)]:
                for region in sorted(set(tags)):
                    mask=tags==region
                    for label,val in values.items():
                        region_rows.append(dict(ident,query_count=n,partition=partition,region=region,label=label,points=int(mask.sum()),point_fraction=float(mask.mean()),conditional_MSE=float(val[mask].mean()),global_MSE_contribution=float(val[mask].sum()/n)))
            for ca,re in PAIRS:
                delta=values[ca]-values[re];up=delta>0;down=delta<0
                row=dict(ident,query_count=n,candidate=ca,reference=re,MSE_delta=float(delta.mean()),points_improved_fraction=float(down.mean()),points_worse_fraction=float(up.mean()),positive_error_contribution=float(delta[up].sum()/n),negative_error_contribution=float(delta[down].sum()/n))
                top=np.argsort(delta)[-max(1,int(np.ceil(.05*n))):]
                row['top5_positive_delta_contribution']=float(np.maximum(delta[top],0).sum()/n)
                pair_rows.append(row)
        # Reconstruct S/I from saved witnesses, not from a new scorer execution.
        gw=ref['gamma_weights'];cw=ref['C_weights'];edges=ref['gamma_point_edge'];cf=ref['C_point_faces']
        for label,cv in controls.items():
            for tau in [.005,.01,.02]:check(rid+label+'S'+str(tau),np.dot(gw,cv['seam_covered_'+str(tau)])/gw.sum(),scored[label][rid]['SeamCov_'+str(tau)])
            check(rid+label+'I',np.dot(cw,cv['conflict_union'])/cw.sum(),scored[label][rid]['ConflictCov_C'])
        for ca,re in PAIRS:
            a,b=controls[ca],controls[re];sa,sb=a['seam_covered_0.01'],b['seam_covered_0.01']
            new=sa&~sb;lost=sb&~sa;ia,ib=a['conflict_union'],b['conflict_union']
            cr=dict(ident,candidate=ca,reference=re,S_gained_weight=float(gw[new].sum()/gw.sum()),S_lost_weight=float(gw[lost].sum()/gw.sum()),I_added_area=float(cw[ia&~ib].sum()/cw.sum()),I_removed_area=float(cw[ib&~ia].sum()/cw.sum()))
            for key in ['conflict_cross','conflict_dup']:
                cr[key+'_delta']=float(np.dot(cw,a[key].astype(float)-b[key].astype(float))/cw.sum())
            cr['cross_and_dup_delta']=float(np.dot(cw,(a['conflict_cross']&a['conflict_dup']).astype(float)-(b['conflict_cross']&b['conflict_dup']).astype(float))/cw.sum())
            control_rows.append(cr)
            check(rid+ca+re+'Sdecomp',cr['S_gained_weight']-cr['S_lost_weight'],scored[ca][rid]['SeamCov']-scored[re][rid]['SeamCov'])
            check(rid+ca+re+'Idecomp',cr['I_added_area']-cr['I_removed_area'],scored[ca][rid]['ConflictCov_C']-scored[re][rid]['ConflictCov_C'])
            for edge in np.unique(edges):
                mask=edges==edge;endpoint=ref['gamma_edges'][edge];delta=(sa.astype(float)-sb.astype(float))*gw
                edge_rows.append(dict(ident,candidate=ca,reference=re,gamma_edge_index=int(edge),known_face_index=int(ref['gamma_known_face_indices'][edge]),length_fraction=float(gw[mask].sum()/gw.sum()),S_delta_global=float(delta[mask].sum()/gw.sum()),mid_x=float(endpoint[:,0].mean()),mid_y=float(endpoint[:,1].mean()),mid_z=float(endpoint[:,2].mean())))
            for face in np.unique(cf):
                mask=cf==face;cent=c['C'][face].mean(0)
                face_rows.append(dict(ident,candidate=ca,reference=re,known_face_index=int(face),source_face_id=int(c['known_source_face_ids'][face]),C_weight_fraction=float(cw[mask].sum()/cw.sum()),I_delta_global=float(np.dot(cw[mask],ia[mask].astype(float)-ib[mask].astype(float))/cw.sum()),dup_delta_global=float(np.dot(cw[mask],a['conflict_dup'][mask].astype(float)-b['conflict_dup'][mask].astype(float))/cw.sum()),centroid_x=float(cent[0]),centroid_y=float(cent[1]),centroid_z=float(cent[2])))
        print('COMPLETE',rid,flush=True)
    for name,rows in [('distance_summary.csv',distance_rows),('region_distance_contributions.csv',region_rows),('paired_point_changes.csv',pair_rows),('query_per_point.csv',points_rows),('control_witness_changes.csv',control_rows),('seam_per_edge.csv',edge_rows),('conflict_per_known_face.csv',face_rows),('raw_geometry_summary.csv',raw_rows)]:write(name,rows)
    for p,h in sources.items():assert sha(p)==h
    receipt=dict(status='PASS',requests=len(requests),labels=LABELS,source_hashes=sources,checks=len(checks),model_calls=0,training_updates=0,generated_meshes=0,resampled_query_points=0,original_scoring_calls=0,new_descriptive_geometry='8 conditions: existing 512/2048 GT queries to original C, finite closed triangle distance only; fixed 1%/5% L0 and GT Y thirds; no semantic segmentation',raw_files_read=80,source_files_unchanged=True,seconds=time.monotonic()-started,script_sha256=sha(Path(__file__)))
    (OUT/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_hashes'}))

if __name__=='__main__':main()
