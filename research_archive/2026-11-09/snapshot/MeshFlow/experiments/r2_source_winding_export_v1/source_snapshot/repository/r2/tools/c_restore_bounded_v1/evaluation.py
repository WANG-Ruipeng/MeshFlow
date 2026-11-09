"""Fixed12 CPU evaluation. No calls to historical runners or model modules."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import concurrent.futures, multiprocessing, resource, sys, time, html
from pathlib import Path
import numpy as np
from . import common as C
from r2.pilots.anchor_aware_denoiser_v1 import evaluation as E
from r2.tools.c_restore_coupling_v1 import evaluation as H

def meta(r):return dict(E.meta(E.canonical_identity(r)),method=r['method'])
def start(kind,r):
 p=C.OUT/'events'/r['method']/r['request_id']/(kind+'_started.json')
 if p.exists():raise RuntimeError('Unfinished attempted stage; no implicit retry: '+str(p))
 C.write(p,dict(stage=kind,**meta(r),utc=time.time()))
def finish(kind,r,**kw):C.write(C.OUT/'events'/r['method']/r['request_id']/(kind+'_complete.json'),dict(stage=kind,**meta(r),**kw))

def orientation_one(r):
 from .orientation import orient_fixed_C
 d=C.read(r['diagnostic_path']);a=C.payload(r['post_path']);K=int(r['K'])
 dest=C.OUT/'oriented'/r['method']/r['request_id'];receipt=dest/'receipt.json'
 ident=dict(**meta(r),post_sha256=C.sha(r['post_path']),diagnostic_sha256=C.sha(r['diagnostic_path']),registration_sha256=C.sha(C.OUT/'registration.json'))
 if receipt.exists():
  q=C.read(receipt);assert q['identity']==ident and q['status']=='PASS';C.verify(q['files']);return q
 if dest.exists():raise RuntimeError('Incomplete orientation output')
 start('ORIENTATION',r);tick=time.perf_counter()
 result=orient_fixed_C(a['vertices'],a['faces'],K);f=result['faces'];flips=result['flips'];tri=a['vertices'][f]
 perm=np.tile(np.arange(3,dtype=np.int64),(len(f),1));perm[flips]=[0,2,1]
 assert C.same(f[:K],a['faces'][:K]) and C.same(tri[:K],a['output'][:K])
 assert np.array_equal(np.sort(f,axis=1),np.sort(a['faces'],axis=1))
 assert C.same(tri,np.take_along_axis(a['output'],perm[:,:,None],axis=1))
 # Preserve source face identity; corner identities follow the reversal explicitly.
 source_corners=np.take_along_axis(a['source_corner_ids'],perm,axis=1)
 C.save_npz(dest/'oriented.npz',output=tri,vertices=a['vertices'],faces=f,C=tri[:K],free=tri[K:],
  source_face_ids=a['source_face_ids'],source_corner_ids=source_corners,flips=flips,corner_permutation=perm)
 _,_,PC,_=E.helpers();PC.write_indexed_ply(C.checked(dest/'oriented.ply'),a['vertices'],f)
 v2,f2=PC.read_indexed_ply(dest/'oriented.ply');assert C.same(v2,a['vertices']) and C.same(f2,f)
 pairs=C.read(d['quality']['pair_audit_path'])
 after,edge_rows,reached=E.actual_valid_graph(a['vertices'],f,K,pairs);after.update(H.opposite_reach(tri,K,edge_rows))
 before=d['topology']
 for key in ('C_F_shared_edge_count','C_F_valid_edge_count','graph_valid_edges','reached_free_faces','nonmanifold_edges','boundary_edges'):
  assert after[key]==before[key],(key,after[key],before[key])
 assert abs(after['reached_free_area_fraction']-before['reached_free_area_fraction'])<1e-12
 C.csvwrite(dest/'components.csv',result['component_rows']);C.csvwrite(dest/'constraints.csv',result['edge_rows'])
 C.csvwrite(dest/'valid_edges_after.csv',edge_rows)
 C.write(dest/'orientation_details.json',dict(summary=result['summary'],components=result['component_rows'],edges=result['edge_rows']))
 eq=dict(C_bytes_and_winding_preserved=True,vertex_array_bytes_unchanged=True,face_count_unchanged=True,
  every_unordered_index_triangle_unchanged=True,source_face_ids_unchanged=True,source_corner_mapping_updated=True,
  quality_reference_path=r['diagnostic_path'],quality_reference_POST_sha256=d['quality']['identity']['processed_sha256'],
  quality_scalar_status='REUSED_UNORIENTED_GEOMETRY_EQUIVALENCE; no claim of a new scorer execution',
  orientability_not_outward_or_manifold_certification=True)
 q=dict(status='PASS',identity=ident,summary=result['summary'],before=before,after=after,equivalence=eq,
  reached_free_output_face_ids=reached,seconds=time.perf_counter()-tick,
  files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
 C.write(receipt,q);finish('ORIENTATION',r,seconds=q['seconds'],receipt_sha256=C.sha(receipt));return q

def orient_baselines(reg):
 for r in reg['baselines']:orientation_one(r)
 C.write(C.OUT/'baseline_orientation_completion.json',dict(status='PASS',outputs=24,new_model_forwards=0))

def postprocess(reg):
 _,_,PC,config=E.helpers();assert config==reg['postprocess_config']
 output=list(reg['baselines'])
 for orig in reg['records']:
  r=dict(orig,method='BLEND');rid=r['request_id'];den=C.OUT/'pre_geo/BLEND'/rid/'denoised.npz'
  rec=C.read(den.with_name('receipt.json'));assert rec['status']=='PASS' and rec['output_sha256']==C.sha(den)
  dest=C.OUT/'outputs/BLEND'/rid;rp=dest/'postprocess.json'
  if rp.exists():
   pp=C.read(rp);assert pp['status']=='PASS' and pp['input']['file_sha256']==C.sha(den)
   for n,v in pp['files'].items():assert C.sha(dest/n)==v['sha256']
  else:
   staging=C.checked(dest.with_name(dest.name+'.staging'))
   if dest.exists() or staging.exists():raise RuntimeError('Incomplete GEO; no automatic replay')
   start('GEO',r);pp=PC.run(den,int(r['K']),staging,input_key='output')
   C.write(staging/'postprocess.json',E.rewrite_paths(pp,staging,dest));os.replace(staging,dest)
   finish('GEO',r,seconds=pp['seconds'])
  a=C.payload(dest/'post.npz');raw=C.payload(r['raw_path'])['raw'].reshape(-1,3,3)
  assert C.same(a['output'][:int(r['K'])],raw[:int(r['K'])]) and C.same(a['vertices'][a['faces']],a['output'])
  v,f=PC.read_indexed_ply(dest/'post.ply');assert C.same(v,a['vertices']) and C.same(f,a['faces'])
  output.append(dict(r,denoised_path=str(den),post_path=str(dest/'post.npz'),post_sha256=C.sha(dest/'post.npz'),
   post_ply=str(dest/'post.ply'),diagnostic_path=str(C.OUT/'diagnostics/BLEND'/rid/'diagnostic.json')))
 C.write(C.OUT/'post_index.json',output);C.csvwrite(C.OUT/'post_index.csv',output)
 C.write(C.OUT/'post_completion.json',dict(status='PASS',new_GEO_C=12,reused_POST=24,outputs=36))
 return output

def score_one(r):
 tick=time.perf_counter();r=E.canonical_identity(r);diag=Path(r['diagnostic_path']);post=Path(r['post_path']);den=Path(r['denoised_path'])
 identity=dict(**meta(r),post_sha256=C.sha(post),input_sha256=C.sha(den),registration_sha256=C.sha(C.OUT/'registration.json'))
 if diag.exists():
  d=C.read(diag);assert d['identity']==identity and d['status']=='PASS';C.verify(d['quality']['files']);return dict(path=str(diag),seconds=0,reused=True,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 A,Q,_,_=E.helpers();dest=C.OUT/'scores/BLEND'/r['request_id'];qr=dest/'quality.json'
 if qr.exists():
  quality=C.read(qr);assert quality['identity']['processed_sha256']==C.sha(post);C.verify(quality['files'])
 else:
  staging=C.checked(dest.with_name(dest.name+'.staging'))
  if dest.exists() or staging.exists():raise RuntimeError('Incomplete quality attempt')
  start('QUALITY',r);quality=Q.score_quality(r,post,staging/'details');quality=E.rewrite_paths(quality,staging,dest)
  C.write(staging/'quality.json',quality);os.replace(staging,dest);finish('QUALITY',r,seconds=quality['seconds'])
 a=C.payload(post);pre=C.payload(den)['output'];K=int(r['K'])
 top,edges,reached=E.actual_valid_graph(a['vertices'],a['faces'],K,C.read(quality['pair_audit_path']));top.update(H.opposite_reach(a['output'],K,edges))
 pret,preedges,prereached,prepairs=H.exact_pre_graph(pre,K,Q,A.L);pret.update(H.opposite_reach(pre,K,preedges))
 bv,bm,ba=H.boundary(r,a['output'],A);pv,pm,pba=H.boundary(r,pre,A)
 C.csvwrite(diag.parent/'valid_edges_POST.csv',edges);C.csvwrite(diag.parent/'valid_edges_PRE.csv',preedges)
 C.write(diag.parent/'PRE_edge_local_pairs.json',prepairs)
 shape=Q._old().cpu_math()['shape'].shape_quality(pre[K:],A.L)
 d=dict(status='PASS',identity=identity,quality=quality,quality_reused=False,topology=top,
  pre_GEO_topology=pret,reached_free_output_face_ids=reached,pre_GEO_reached_free_face_ids=prereached,
  vertices=bv,metrics=bm,boundary_audit=ba,pre_GEO_vertices=pv,pre_GEO_metrics=pm,
  pre_GEO_quality_status='NA_NOT_BUDGETED',pre_GEO_shape=shape,
  seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 C.write(diag,d)
 if 'torch' in sys.modules:raise RuntimeError('CPU-only contract violated: torch imported')
 return dict(path=str(diag),seconds=d['seconds'],reused=False,peak_RSS_KiB=d['peak_RSS_KiB'])

def score(reg,rows):
 starttime=time.perf_counter();new=[r for r in rows if r['method']=='BLEND']
 C.write(C.OUT/'resources_before_score.json',C.resources())
 first=score_one(new[0]);res=C.resources();C.write(C.OUT/'resources_after_first_score.json',res)
 mem={line.split(':')[0]:int(line.split()[1])*1024 for line in res['memory'].splitlines() if line.split()[1].isdigit()}
 peak=first['peak_RSS_KiB']*1024
 workers=2 if (os.cpu_count() or 1)>=4 and mem['MemAvailable']>max(4<<30,3*peak) and res['loadavg'][0]<(os.cpu_count() or 1)-2 else 1
 C.write(C.OUT/'CPU_dispatch.json',dict(workers=workers,threads_per_worker=2,first_authorized_job=first,
  available_RAM_bytes=mem['MemAvailable'],loadavg=res['loadavg'],GPU_workers=0,
  reason='Measured first authorized quality job; CPU/RAM/load guard; at most two independent remaining jobs',
  timing_scope='CPU scoring concurrent wall time; not an isolated benchmark'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
  receipts=[first,*pool.map(score_one,new[1:],chunksize=1)]
 for r in new:orientation_one(r)
 C.write(C.OUT/'score_completion.json',dict(status='PASS',new_POST_quality=12,reused_quality=24,
  receipts=receipts,CPU_workers=workers,seconds=time.perf_counter()-starttime,new_model_forwards=0))
 C.write(C.OUT/'resources_after_score.json',C.resources())
 return receipts

def aggregate(rows):
 vm=[];verts=[];qualities=[];tops=[];ors=[];dist=[];decisions=[];components=[];constraints=[];mech=[]
 for r in rows:
  m=meta(r);d=C.read(r['diagnostic_path']);K=int(r['K']);a=C.payload(r['post_path'])
  op=C.OUT/'oriented'/r['method']/r['request_id'];o=C.read(op/'receipt.json')
  qualities.append(dict(**m,phase='POST',**d['quality']['metrics'],free_face_denominator=d['quality']['free_face_denominator']))
  for phase,tk,mk,vk in [('PRE_GEO','pre_GEO_topology','pre_GEO_metrics','pre_GEO_vertices'),('POST','topology','metrics','vertices')]:
   tops.append(dict(**m,phase=phase,**d[tk]));vm.extend(dict(**m,phase=phase,**v) for v in d[mk]);verts.extend(dict(**m,phase=phase,**v) for v in d[vk])
   ds=[v['distance_model'] for v in d[vk]]
   dist.append(dict(**m,phase=phase,boundary_vertices=len(ds),distance_mean_model=float(np.mean(ds)),distance_p95_model=float(np.quantile(ds,.95)),distance_max_model=max(ds)))
  tops.append(dict(**m,phase='POST_ORIENTED',**o['after']))
  ors.append(dict(**m,**o['summary'],before_opposed_edges=o['before']['C_F_valid_opposite_winding_edge_count'],after_opposed_edges=o['after']['C_F_valid_opposite_winding_edge_count'],
   before_opposed_reach=o['before']['opposite_winding_reached_free_area_fraction'],after_opposed_reach=o['after']['opposite_winding_reached_free_area_fraction'],
   unoriented_reach=o['before']['reached_free_area_fraction'],seconds=o['seconds']))
  components.extend(dict(**m,**v) for v in C.rows(op/'components.csv'));constraints.extend(dict(**m,**v) for v in C.rows(op/'constraints.csv'))
  q=d['quality']['metrics'];strict=next(v for v in d['metrics'] if v['setting']=='raw_tau_1e-6');loose=next(v for v in d['metrics'] if v['setting']=='L_tau_0.02')
  decisions.append(dict(**m,CF=q['CF_proper_affected_free_fraction'],CF_illegal=q['CF_illegal_affected_free_fraction'],FF=q['FF_illegal_affected_free_fraction'],I=q['ConflictCov_C'],
   small_angle_lt5=q['free_minimum_angle_lt5_fraction'],strict_AVMR=strict['VMR'],strict_PMR=strict['perfect'],AVMR_02L=loose['VMR'],
   distance_mean_model=float(np.mean([v['distance_model'] for v in d['vertices']])),pre_distance_mean_model=float(np.mean([v['distance_model'] for v in d['pre_GEO_vertices']])),
   valid_edges=d['topology']['C_F_valid_edge_count'],opposed_before=d['topology']['C_F_valid_opposite_winding_edge_count'],opposed_after=o['after']['C_F_valid_opposite_winding_edge_count'],
   reachable_area=d['topology']['reached_free_area_fraction'],opposed_reach_before=d['topology']['opposite_winding_reached_free_area_fraction'],opposed_reach_after=o['after']['opposite_winding_reached_free_area_fraction'],
   free_faces=len(a['output'])-K,deleted_free_faces=int(r['N'])-len(a['output'])))
  if r['method']=='BLEND':
   b=C.read(Path(r['denoised_path']).with_name('diagnostics.json'));mech.append(dict(**m,**{k:v for k,v in b.items() if isinstance(v,(str,int,float,bool)) or v is None}))
 tables={'AVMR_PMR_per_output':vm,'boundary_vertices':verts,'quality_per_output':qualities,'topology_per_output':tops,'orientation_per_output':ors,
  'boundary_distance_per_output':dist,'decision_per_output':decisions,'orientation_components':components,'orientation_constraints':constraints,'blend_mechanism':mech}
 for name,data in tables.items():C.csvwrite(C.OUT/(name+'.csv'),data)
 numerics=[k for k,v in decisions[0].items() if isinstance(v,(int,float)) and k not in E.META]
 summaries=[]
 for group,key in [('ALL',None),('REGION','region_type'),('PARENT','parent_id'),('NOISE','candidate')]:
  for method in ('D0_BASE','RBF','BLEND'):
   vals=sorted({r[key] for r in decisions}) if key else ['ALL']
   for value in vals:
    subset=[r for r in decisions if r['method']==method and (not key or r[key]==value)]
    summaries.append(dict(group=group,value=value,method=method,n=len(subset),**{k:float(np.mean([r[k] for r in subset])) for k in numerics},
     outputs_with_valid_edges=sum(r['valid_edges']>0 for r in subset),outputs_with_opposed_edges_before=sum(r['opposed_before']>0 for r in subset),outputs_with_opposed_edges_after=sum(r['opposed_after']>0 for r in subset)))
 C.csvwrite(C.OUT/'decision_summary.csv',summaries)
 paired=[]
 for r in decisions:
  if r['method']=='D0_BASE':continue
  for base in ('D0_BASE','RBF'):
   if base==r['method']:continue
   ref=next(x for x in decisions if x['request_id']==r['request_id'] and x['method']==base)
   paired.append(dict(**meta(r),reference=base,**{k+'_delta':r[k]-ref[k] for k in numerics}))
 C.csvwrite(C.OUT/'paired_per_parent_noise.csv',paired)
 # Preserve complete per-threshold reductions, with parent/noise/region splits.
 vnums=['boundary_vertex_count','matched_count','VMR','perfect']
 vsummary=H.summary(vm,('method','phase','domain','setting','threshold_model','scale','tau'),vnums)
 C.csvwrite(C.OUT/'AVMR_PMR_summary.csv',vsummary)
 C.write(C.OUT/'aggregate_completion.json',dict(status='PASS',outputs=36,orientation_outputs=36,POST_quality_new=12,POST_quality_reused=24,AVMR_rows=len(vm),full_PRE_quality='NA_NOT_BUDGETED'))

def render(rows):
 import matplotlib.pyplot as plt
 from PIL import Image
 G=E.module('_bounded_existing_renderer',C.ROOT/'r2/tools/student_multistep_denoise_v1/gallery.py')
 reg=C.registration();gdir=C.OUT/'gallery';gdir.mkdir(exist_ok=True);tick=time.perf_counter();receipts=[];cards=[]
 methods=('D0_BASE','RBF','BLEND')
 for orig in reg['records']:
  rid=orig['request_id'];K=int(orig['K']);group={r['method']:r for r in rows if r['request_id']==rid}
  meshes={m:C.payload(group[m]['post_path'])['output'] for m in methods}
  xyz=np.concatenate([x.reshape(-1,3) for x in meshes.values()]).astype(float);lo,hi=xyz.min(0),xyz.max(0)
  bounds=dict(center_model_xyz=((lo+hi)/2).tolist(),span=float((hi-lo).max())*1.12,bounds_model_xyz=[lo.tolist(),hi.tolist()])
  for vi,camera in enumerate(((22.,-60.),(22.,120.))):
   png=gdir/(rid+'__v'+str(vi)+'.png');jpg=png.with_suffix('.jpg');receipt=png.with_suffix('.json')
   identity=dict(request_id=rid,methods=methods,camera=camera,bounds=bounds,source_hashes={group[m]['post_path']:C.sha(group[m]['post_path']) for m in methods})
   identity=C.clean(identity)
   if receipt.exists():
    q=C.read(receipt);assert q['identity']==identity;assert C.sha(png)==q['png_sha256'] and C.sha(jpg)==q['jpg_sha256']
   else:
    fig=plt.figure(figsize=(18,6),dpi=140)
    for j,m in enumerate(methods):
     ax=fig.add_subplot(1,3,j+1,projection='3d');G.draw(ax,meshes[m],K,bounds,camera)
     ax.set_title(m+' | F='+str(len(meshes[m])-K),fontsize=14)
    fig.suptitle(orig['parent_id']+' | '+orig['region_type']+' '+orig['tier']+' | n'+str(orig['candidate'])+' | view '+str(vi+1),fontsize=14)
    fig.text(.5,.018,'Blue: original C | Gray: free | Same camera/scale | Pre-flip POST; orientation changes no surface geometry',ha='center',fontsize=11)
    fig.subplots_adjust(left=0,right=1,top=.89,bottom=.04,wspace=0)
    C.atomic(png,lambda f:fig.savefig(f,format='png',dpi=140),True);plt.close(fig)
    with Image.open(png) as im:C.atomic(jpg,lambda f:im.convert('RGB').save(f,format='JPEG',quality=85,optimize=True),True)
    q=dict(status='COMPLETE',identity=identity,png=str(png),jpg=str(jpg),png_sha256=C.sha(png),jpg_sha256=C.sha(jpg));C.write(receipt,q)
   receipts.append(q)
  cards.append('<section><h2>'+html.escape(rid)+'</h2>'+''.join('<a href="'+rid+'__v'+str(v)+'.jpg"><img style="width:99%" src="'+rid+'__v'+str(v)+'.jpg"></a>' for v in (0,1))+'</section>')
 C.textfile(gdir/'index.html','<!doctype html><meta charset="utf-8"><title>Bounded C restoration</title><h1>12 fixed paired requests</h1><p>Three POST methods, two matched views. Geometry before winding repair.</p>'+''.join(cards))
 C.write(gdir/'manifest.json',dict(status='PASS',images=24,requests=12,receipts=receipts,seconds=time.perf_counter()-tick))
