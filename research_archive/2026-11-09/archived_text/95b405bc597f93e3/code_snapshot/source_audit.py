"""CPU-only source-target localization; old meshes, scorers and reports stay read-only."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/e/MeshFlow')
OLD = ROOT / 'experiments/r2_user_data_scaling_32_67_v1'
AUDIT = ROOT / 'experiments/r2_source_interface_audit_v1'
OUT = ROOT / 'experiments/r2_seat_failure_diagnosis_v1/source_audit'
TAUS = (0.005, 0.01, 0.02)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def js(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def rows(path):
    return list(csv.DictReader(Path(path).open(encoding='utf-8-sig', newline='')))

def csvout(path, data):
    keys=list(dict.fromkeys(k for row in data for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)

def compact(v):
    return json.dumps(v,separators=(',',':'))

def triangle_key(t):
    # Exact numeric FP32 equality, unordered corners; +0 and -0 normalized.
    z=np.array(t,dtype=np.float32,copy=True);z[z==0]=0
    return tuple(sorted(tuple(float(q) for q in p) for p in z))

def frame(triangles):
    t=np.asarray(triangles,dtype=np.float64)
    cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); areas=np.linalg.norm(cross,axis=1)/2
    total=areas.sum(); center=np.sum(areas[:,None]*t.mean(1),axis=0)/total
    second=np.zeros((3,3))
    for a,tt in zip(areas,t):
        s=tt.sum(0);second+=a*(np.outer(s,s)+tt.T@tt)/12
    cov=second/total-np.outer(center,center)
    vals,vecs=np.linalg.eigh(cov); n=vecs[:,0]
    # Model Y is the inherited Chair vertical axis; do not use source winding.
    if n[1]<0:n=-n
    axis=np.eye(3)[np.argmin(np.abs(n))];u=np.cross(n,axis);u/=np.linalg.norm(u);v=np.cross(n,u)
    basis=np.stack([u,v],axis=1)
    return center,n,basis,areas,vals

def point_surface_distance(points,triangles):
    p=np.asarray(points,dtype=np.float64)[:,None,:];t=np.asarray(triangles,dtype=np.float64)
    a=t[None,:,0,:];b=t[None,:,1,:];c=t[None,:,2,:]
    ab=b-a;ac=c-a;n=np.cross(ab,ac);nn=(n*n).sum(-1)
    signed=((p-a)*n).sum(-1);proj=p-n*(signed/np.maximum(nn,1e-30))[...,None]
    v=proj-a;d00=(ab*ab).sum(-1);d01=(ab*ac).sum(-1);d11=(ac*ac).sum(-1)
    d20=(v*ab).sum(-1);d21=(v*ac).sum(-1);den=d00*d11-d01*d01
    s=(d11*d20-d01*d21)/np.maximum(den,1e-30);q=(d00*d21-d01*d20)/np.maximum(den,1e-30)
    inside=(s>=-1e-12)&(q>=-1e-12)&(s+q<=1+1e-12)&(nn>0)
    best=np.where(inside,signed*signed/np.maximum(nn,1e-30),np.inf)
    for x,y in ((a,b),(b,c),(c,a)):
        e=y-x;frac=np.clip(((p-x)*e).sum(-1)/np.maximum((e*e).sum(-1),1e-30),0,1)
        d=((p-(x+frac[...,None]*e))**2).sum(-1);best=np.minimum(best,d)
    return np.sqrt(np.min(best,axis=1))

def cross2(a,b):return float(a[0]*b[1]-a[1]*b[0])

def area2(poly):
    if len(poly)<3:return 0.0
    x=np.asarray(poly);return abs(float(np.sum(x[:,0]*np.roll(x[:,1],-1)-x[:,1]*np.roll(x[:,0],-1))))/2

def clip_area(subject,clip):
    if np.any(np.max(subject,axis=0)<np.min(clip,axis=0)) or np.any(np.max(clip,axis=0)<np.min(subject,axis=0)):return 0.0
    c=np.array(clip);sgn=cross2(c[1]-c[0],c[2]-c[0]);
    if abs(sgn)<1e-20:return 0.0
    if sgn<0:c=c[::-1]
    p=list(np.array(subject))
    for a,b in zip(c,np.roll(c,-1,axis=0)):
        if not p:break
        new=[];last=p[-1];dl=cross2(b-a,last-a)
        for cur in p:
            dc=cross2(b-a,cur-a)
            if (dc>=0)!=(dl>=0):
                new.append(last+(cur-last)*(dl/(dl-dc)))
            if dc>=0:new.append(cur)
            last=cur;dl=dc
        p=new
    return area2(p)

def projection_stats(tri,region,center,basis,L):
    p=(tri-center)@basis;rr=(region-center)@basis
    ar=area2(p);over=[clip_area(p,r) for r in rr]
    # Sum is intentionally pairwise, not a union; curved projections may overlap.
    return {'projected_area':ar,'projection_pairwise_overlap_sum':sum(over),
            'projection_max_single_triangle_overlap':max(over,default=0),
            'projection_any_positive':max(over,default=0)>1e-12*L*L,
            'projection_pairwise_overlap_over_Y_projection':sum(over)/ar if ar>1e-16*L*L else 'NA_ZERO_PROJECTED_AREA'}

def resources():
    import psutil,subprocess
    r={'cpu_count':os.cpu_count(),'memory_available':psutil.virtual_memory().available,'loadavg':os.getloadavg(),
       'workers':1,'threads':1,'GPU_calls':0,'dispatch':'One CPU analysis worker; independent of controller GPU tasks.'}
    for key,args in [('gpu',['--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
                     ('processes',['--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])]:
        try:r[key]=subprocess.check_output(['nvidia-smi',*args],text=True,stderr=subprocess.STDOUT)
        except Exception as e:r[key]='NA_'+str(e)
    return r

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    m=json.loads((OLD/'data/train_manifest.json').read_text())
    tasks={k:v for k,v in m['tasks'].items() if v['condition_type'].startswith('SEAT') and v['uid'] in m['pools']['USER_EXPANDED_67']}
    assert len(tasks)==98
    L=json.loads((OLD/'evaluation/registration.json').read_text())['definition']['L']
    paths={Path(__file__),ROOT/'AGENTS.md',ROOT/'WORKSPACE.md',ROOT/'native_t1/docs/control_metrics.md',
           OLD/'data/train_manifest.json',OLD/'evaluation/registration.json',AUDIT/'statistics/per_condition.csv',
           AUDIT/'statistics/exposure_association.csv',AUDIT/'source_interfaces.csv',AUDIT/'C_Y_edge_details.csv'}
    for t in tasks.values():paths.update([Path(t['npz']),Path(t['quality_path']),Path(m['parents'][t['uid']]['npz'])])
    hashes=[{'path':str(p),'exists':p.exists(),'sha256':sha(p) if p.exists() else None,'bytes':p.stat().st_size if p.exists() else None} for p in sorted(paths)]
    rule={'version':'seat_source_localization_v1','created_unix':time.time(),'conditions':sorted(tasks),'count':98,'L':L,'tau_over_L':list(TAUS),
          'target':'EXP67 frozen 50 SEAT_FULL + 48 SEAT_HALF; source geometry diagnostic, not generated success score.',
          'identity':'Read original indexed model_vertices/source_faces plus condition source_face_ids and free_source_ids; require exact C_original match and saved file hashes.',
          'duplicate':'Exact unordered FP32 numeric coordinate triangle groups wholly inside Y; independently recomputed and compared with saved audit. Coincident winding is ignored, +/-0 numerically identical.',
          'seat_region':'R is frozen region_source_face_ids, C is frozen source_face_ids. HALF C differs from R. Never infer a new seat label.',
          'local_frame':'Area-integrated least-squares plane of C or R independently; normal points toward model +Y; record planarity and abs normal Y. Not a new semantic label.',
          'distance':'Exact centroid-to-union-of-finite-triangles nearest distance, and vertex/mid-edge/centroid 7-point finite-surface sampled min/max. Sampled minimum is NOT exact triangle-triangle minimum.',
          'projection':'Orthogonally project Y face and each C/R triangle onto that region plane, clip convex triangles; record any positive projected overlap >1e-12 L^2, max per-triangle and pairwise summed overlap. Pairwise sum is NOT a union area.',
          'near':'Centroid finite-surface distance <= tau L, tau=0.005/0.01/0.02; separately record 7-point sampled minimum <= tau L; no success/failure threshold.',
          'local_duplicate':'Y has exact duplicate group, representative has positive footprint overlap with C/R and centroid distance <= tau L. Descriptive localization, does not prove causal repeated-seat generation.',
          'bottom_candidate':'Nonduplicate Y face, abs(normal dot C plane normal)>=0.9, positive projection overlap, whole face signed coordinate below min C signed coordinate by >1e-7 L, C plane abs(Y normal)>=0.7. This is a plausible lower layer, NOT certified reasonable or illegal. Record all distances, no automatic removal.',
          'legality':'Reuse frozen whole-source facepair classification only. Near, parallel or projected-overlapping alone never means illegal. Missing classification retained as NO_RECORDED_CONTACT or NA.',
          'interface':'Reconstruct indexed C/Y edge incidence from true source arrays, verify inherited SOURCE_INDEX edge endpoints and source face IDs. Inherit legal status; do not rerun all triangle-pair intersections.',
          'budget':{'models':0,'training':0,'new_generation':0,'new_noise':0,'new_triangle_intersection_tests':0,'geometry_conditions':98}}
    if (OUT/'rules.json').exists():
        old=json.loads((OUT/'rules.json').read_text());assert old['conditions']==rule['conditions'] and old['version']==rule['version']
        return
    js(OUT/'rules.json',rule);js(OUT/'protected_before.json',hashes);js(OUT/'resources_before.json',resources())
    print(json.dumps({'prepared':98,'rules_sha256':sha(OUT/'rules.json'),'protected_files':len(hashes)}),flush=True)

def run():
    start=time.time();rule=json.loads((OUT/'rules.json').read_text());L=rule['L']
    before=json.loads((OUT/'protected_before.json').read_text())
    for r in before:
        if r['exists']:assert sha(r['path'])==r['sha256'],r['path']
    m=json.loads((OLD/'data/train_manifest.json').read_text());oldstats={r['condition_id']:r for r in rows(AUDIT/'statistics/per_condition.csv')}
    exps={r['condition_id']:r for r in rows(AUDIT/'statistics/exposure_association.csv') if r['arm']=='USER_EXPANDED_67'}
    inter={r['condition_id']:r for r in rows(AUDIT/'source_interfaces.csv') if r['topology_domain']=='SOURCE_INDEX' and r['direction']=='AS_STORED_SOURCE'}
    edges=collections.defaultdict(list)
    for r in rows(AUDIT/'C_Y_edge_details.csv'):
        if r['topology_domain']=='SOURCE_INDEX':edges[r['condition_id']].append(r)
    allc=[];allg=[];allf=[];alle=[];blocked=[]
    for ix,cid in enumerate(rule['conditions']):
        cell=OUT/'conditions'/cid;cell.mkdir(parents=True,exist_ok=True)
        if (cell/'complete.json').exists():
            z=json.loads((cell/'complete.json').read_text());allc.append(z['condition']);allg+=z['groups'];allf+=z['faces'];alle+=z['interfaces'];continue
        try:
            t=m['tasks'][cid];parent=m['parents'][t['uid']]
            assert sha(parent['npz'])==parent['sha256'];assert sha(t['npz'])==t['sha256'];assert sha(t['quality_path'])==t['quality_sha256']
            with np.load(parent['npz'],allow_pickle=False) as a:v=a['model_vertices'].copy();f=a['source_faces'].copy()
            with np.load(t['npz'],allow_pickle=False) as q:
                ci=q['source_face_ids'].copy();yi=q['free_source_ids'].copy();ri=q['region_source_face_ids'].copy();co=q['C_original'].copy()
            assert sorted(np.r_[ci,yi].tolist())==list(range(len(f))) and not set(ci)&set(yi)
            triangles=v[f];C=triangles[ci].astype(np.float64);R=triangles[ri].astype(np.float64)
            assert np.array_equal(co,triangles[ci]);assert np.array_equal(ci,np.array(t['source_face_ids']));assert np.array_equal(yi,np.array(t['free_source_ids']))
            quality=json.loads(Path(t['quality_path']).read_text());assert quality['input_npz_sha256']==sha(parent['npz'])
            qpair={tuple(sorted(x['pair'])):x for x in quality['pair_contacts']}
            dupmap=collections.defaultdict(list)
            for j in yi:dupmap[triangle_key(triangles[j])].append(int(j))
            groups=sorted(sorted(g) for g in dupmap.values() if len(g)>1)
            assert groups==sorted(sorted(g) for g in json.loads(oldstats[cid]['Y_exact_duplicate_groups'])),cid
            duplicate_ids={j for g in groups for j in g};group_id={j:gix for gix,g in enumerate(groups) for j in g}
            c0,cn,cb,ca,ce=frame(C);r0,rn,rb,ra,re=frame(R)
            ee=collections.defaultdict(list)
            for j,ff in enumerate(f):
                for aa,bb in zip(ff,np.roll(ff,-1)):ee[tuple(sorted([int(aa),int(bb)]))].append(j)
            cic=set(map(int,ci));yic=set(map(int,yi));truecf={key:ids for key,ids in ee.items() if set(ids)&cic and set(ids)&yic}
            inherited=inter[cid];assert len(truecf)==int(inherited['C_F_shared_edge_count'])
            eout=[]
            for e in edges[cid]:
                key=tuple(sorted([int(e['vertex_u']),int(e['vertex_v'])]));sids=json.loads(e['source_face_ids'])
                assert key in truecf and sorted(sids)==sorted(truecf[key])
                eout.append({'condition_id':cid,'source_vertex_u':key[0],'source_vertex_v':key[1],
                             'source_face_ids':compact(sids),'actual_source_incidence_verified':True,'inherited_valid_edge':e['valid_edge'],
                             'inherited_pair_classification':e['pair_classification'],'inherited_rejected_reasons':e['rejected_reasons']})
            assert sum(e['inherited_valid_edge']=='True' for e in eout)==int(inherited['C_F_valid_edge_count'])
            faceout=[]
            for j in yi:
                tt=triangles[j].astype(np.float64);centroid=tt.mean(0);samples=np.vstack([tt,(tt+np.roll(tt,-1,axis=0))/2,centroid])
                dc=point_surface_distance(samples,C);dr=point_surface_distance(samples,R)
                nn=np.cross(tt[1]-tt[0],tt[2]-tt[0]);norm=float(np.linalg.norm(nn));unit=nn/norm if norm else np.zeros(3)
                pc=projection_stats(tt,C,c0,cb,L);pr=projection_stats(tt,R,r0,rb,L)
                signed=(tt-c0)@cn;sc=(C-c0)@cn
                relc=[qpair.get(tuple(sorted([int(j),int(k)]))) for k in ci]
                contacts=[x for x in relc if x]
                lower=bool(int(j) not in duplicate_ids and abs(unit@cn)>=0.9 and pc['projection_any_positive'] and max(signed)<float(sc.min())-1e-7*L and abs(cn[1])>=0.7)
                row={'condition_id':cid,'parent_uid':t['uid'],'condition_type':t['condition_type'],'source_face_id':int(j),'exact_duplicate_group_id':group_id.get(int(j),'NA_NOT_DUPLICATE'),
                     'centroid':compact(centroid.tolist()),'Y_area':norm/2,'C_centroid_distance_over_L':float(dc[-1]/L),'C_sample7_min_distance_over_L':float(dc.min()/L),'C_sample7_max_distance_over_L':float(dc.max()/L),
                     'R_centroid_distance_over_L':float(dr[-1]/L),'R_sample7_min_distance_over_L':float(dr.min()/L),'R_sample7_max_distance_over_L':float(dr.max()/L),
                     'abs_normal_dot_C_plane':float(abs(unit@cn)),'abs_normal_dot_R_plane':float(abs(unit@rn)),
                     'signed_C_plane_min_over_L':float(min(signed)/L),'signed_C_plane_max_over_L':float(max(signed)/L),
                     'Y_y_min':float(tt[:,1].min()),'Y_y_max':float(tt[:,1].max()),'possible_lower_layer_not_certified':lower,
                     'C_contact_categories':compact(sorted(set(x['category'] for x in contacts))) if contacts else 'NO_RECORDED_CONTACT',
                     'C_contact_classifications':compact(sorted(set(x['classification'] for x in contacts))) if contacts else 'NO_RECORDED_CONTACT',
                     'interpretation':'EXACT_DUPLICATE_IN_Y' if int(j) in duplicate_ids else ('POSSIBLE_LOWER_LAYER_NOT_CERTIFIED' if lower else 'OTHER_SOURCE_FACE')}
                row.update({'C_'+k:v for k,v in pc.items()});row.update({'R_'+k:v for k,v in pr.items()})
                for tau in TAUS:
                    tag=str(tau);row['C_centroid_near_'+tag]=bool(dc[-1]<=tau*L);row['R_centroid_near_'+tag]=bool(dr[-1]<=tau*L)
                    row['C_sample7_near_'+tag]=bool(dc.min()<=tau*L);row['R_sample7_near_'+tag]=bool(dr.min()<=tau*L)
                faceout.append(row)
            byface={r['source_face_id']:r for r in faceout};groupout=[]
            for gix,g in enumerate(groups):
                r=dict(byface[g[0]]);r.update({'duplicate_group_id':gix,'duplicate_source_face_ids':compact(g),'multiplicity':len(g),'extra_face_count':len(g)-1,
                        'original_saved_group_verified':True,'exact_numeric_geometry_verified':True,
                        'duplicate_internal_pair_categories':compact(sorted(set(qpair.get(tuple(sorted([a,b])),{}).get('category','NO_RECORDED_CONTACT') for i,a in enumerate(g) for b in g[i+1:])))})
                for tau in TAUS:
                    tag=str(tau)
                    for region in ['C','R']:r[region+'_local_duplicate_'+tag]=bool(r[region+'_projection_any_positive'] and r[region+'_centroid_near_'+tag])
                groupout.append(r)
            c={'condition_id':cid,'parent_uid':t['uid'],'condition_type':t['condition_type'],'N':len(f),'K':len(ci),'R_faces':len(ri),'Y_faces':len(yi),
               'condition_source_face_ids':compact(ci.tolist()),'region_source_face_ids':compact(ri.tolist()),'C_original_match':True,'duplicate_groups_match_saved':True,
               'actual_exposures':int(exps[cid]['actual_exposures']),'Y_duplicate_groups':len(groups),'Y_duplicate_faces':len(duplicate_ids),
               'C_plane_normal':compact(cn.tolist()),'C_plane_abs_normal_Y':float(abs(cn[1])),'C_area_integrated_plane_RMS_over_L':float(np.sqrt(max(0,ce[0]))/L),
               'R_plane_normal':compact(rn.tolist()),'R_area_integrated_plane_RMS_over_L':float(np.sqrt(max(0,re[0]))/L),
               'C_y_min':float(C[:,:,1].min()),'C_y_max':float(C[:,:,1].max()),'source_interface_actual_shared_edges':len(truecf),
               'source_interface_inherited_valid_edges':int(inherited['C_F_valid_edge_count']),'source_interface_ids_verified':True,
               'source_interface_boundary_fraction_inherited':float(inherited['valid_interface_boundary_length_fraction']),
               'possible_lower_layer_faces_not_certified':sum(x['possible_lower_layer_not_certified'] for x in faceout),'status':'PASS'}
            for tau in TAUS:
                tag=str(tau)
                for region in ['C','R']:
                    c[region+'_local_duplicate_groups_'+tag]=sum(x[region+'_local_duplicate_'+tag] for x in groupout)
                    c[region+'_any_duplicate_centroid_near_'+tag]=sum(x[region+'_centroid_near_'+tag] for x in groupout)
                    c[region+'_any_duplicate_sample7_near_'+tag]=sum(x[region+'_sample7_near_'+tag] for x in groupout)
            js(cell/'complete.json',{'condition':c,'groups':groupout,'faces':faceout,'interfaces':eout,'rules_sha256':sha(OUT/'rules.json')})
            allc.append(c);allg+=groupout;allf+=faceout;alle+=eout
        except Exception as e:
            b={'condition_id':cid,'status':'BLOCKED','reason':str(e),'traceback':traceback.format_exc()};blocked.append(b);js(cell/'blocked.json',b)
        if (ix+1)%10==0:print(json.dumps({'completed':ix+1,'passed':len(allc),'blocked':len(blocked)}),flush=True)
    csvout(OUT/'per_condition.csv',allc);csvout(OUT/'per_duplicate_group.csv',allg);csvout(OUT/'per_Y_face.csv',allf);csvout(OUT/'source_interface_verification.csv',alle);csvout(OUT/'blocked.csv',blocked)
    summary=[]
    for kind in ['ALL','SEAT_FULL','SEAT_HALF']:
        cc=[r for r in allc if kind=='ALL' or r['condition_type']==kind];gg=[r for r in allg if kind=='ALL' or r['condition_type']==kind]
        s={'type':kind,'conditions':len(cc),'parents':len(set(r['parent_uid'] for r in cc)),'actual_exposures':sum(r['actual_exposures'] for r in cc),
           'conditions_with_any_Y_duplicate':sum(r['Y_duplicate_groups']>0 for r in cc),'duplicate_groups_counted_per_condition':len(gg),
           'exposures_with_any_Y_duplicate':sum(r['actual_exposures'] for r in cc if r['Y_duplicate_groups']>0),
           'conditions_with_possible_lower_layer_not_certified':sum(r['possible_lower_layer_faces_not_certified']>0 for r in cc),
           'conditions_with_verified_legal_source_interface':sum(r['source_interface_inherited_valid_edges']>0 for r in cc)}
        for tau in TAUS:
            tag=str(tau)
            for region in ['C','R']:
                key=region+'_local_duplicate_groups_'+tag
                s[region+'_conditions_with_local_duplicate_'+tag]=sum(r[key]>0 for r in cc)
                s[region+'_exposures_with_local_duplicate_'+tag]=sum(r['actual_exposures'] for r in cc if r[key]>0)
                s[region+'_local_duplicate_groups_'+tag]=sum(r[key] for r in cc)
        summary.append(s)
    csvout(OUT/'summary.csv',summary)
    after=[dict(r,after_sha256=sha(r['path']) if Path(r['path']).exists() else None) for r in before]
    unchanged=all(r['sha256']==r['after_sha256'] for r in after);js(OUT/'protected_after.json',after)
    js(OUT/'source_hashes.json',{'entry':str(Path(__file__)),'sha256':sha(__file__),'rules_sha256':sha(OUT/'rules.json')})
    result={'status':'COMPLETE' if not blocked and len(allc)==98 and unchanged else 'COMPLETE_WITH_BLOCKED','elapsed_seconds':time.time()-start,'conditions':len(allc),
            'group_rows':len(allg),'Y_face_rows':len(allf),'interface_rows':len(alle),'blocked':len(blocked),'protected_files':len(before),'protected_unchanged':unchanged,
            'model_calls':0,'new_intersection_tests':0,'summary':summary}
    js(OUT/'COMPLETE.json',result);js(OUT/'resources_after.json',resources());print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run'],required=True);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:run()
