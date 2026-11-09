"""Four saved stages, inherited central cuts, real halfspace cutaways.

Geometry/raster functions are the already-verified section audit renderer.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,html,time,resource
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .evaluation import C,ROOT,OUT,ARMS,requests,mesh_path,dispatch_cpu
from r2.tools.seam_s2000_section_audit_v1 import render as R

STAGES=('RAW','D0','RESTORED','POST')
PREVIOUS=ROOT/'experiments/r2_seam_s2000_section_audit_v1'

def one(job):
    label,q,dest,dh,protocol=job;dest=Path(dest);rid=q['request_id'];p=dest/(rid+'.png');rp=p.with_suffix('.json')
    if rp.exists():
        r=C.read(rp);assert r['definition_sha256']==dh
        for f,h in r['outputs'].items():assert C.sha(f)==h
        return r
    if p.exists():raise ValueError('Unreceipted image retained '+str(p))
    tick=time.perf_counter();K=q['K'];bounds=protocol['bounds'][q['condition_id']];inputs={};meshes={};identities={}
    for stage in STAGES:
        path=mesh_path(label,rid,stage);inputs[str(path)]=C.sha(path);a=C.arrays(path)
        key='raw' if stage=='RAW' else 'official_unconstrained_proposal' if stage=='D0' else 'output'
        meshes[stage]=a[key].astype(np.float64);identities[stage]=a['source_face_ids'].astype(np.int32) if stage=='POST' else np.arange(len(meshes[stage]),dtype=np.int32)
        assert np.isfinite(meshes[stage]).all() and len(meshes[stage])==len(identities[stage])
        assert np.array_equal(identities[stage][:K],np.arange(K))
    size=R.SIZE;head=R.HEADER;foot=R.FOOTER
    page=Image.new('RGB',(4*size,head+4*size+foot),'white');d=ImageDraw.Draw(page)
    d.text((12,8),rid+' | '+q['split']+' | '+label+' | same inherited section planes and scale',font=R.font(24),fill='black')
    d.text((12,40),'C blue / free gray. D0 blue C HAS MOVED. Purple cut rims are artificial. Same slot is NOT a semantic match across models.',font=R.font(19),fill='#973030')
    arrays={};panels=[];checks=[]
    for col,stage in enumerate(STAGES):
        d.text((col*size+12,77),stage+' proposal (C moved)' if stage=='D0' else stage,font=R.font(24),fill='black')
        tri=meshes[stage];slots=identities[stage]
        for vi,spec in enumerate(protocol['sections']):
            axis=spec['axis'];pos=bounds[spec['position_field']][spec['position_index']];name=spec['name']
            for mode in ('section','cutaway'):
                row=vi if mode=='section' else vi+2
                if mode=='section':im,pixels,geom=R.sections_panel(tri,slots,K,axis,pos,bounds,1e-12*C.L)
                else:
                    v=protocol['cutaway'][name];im,pixels,geom=R.cutaway_panel(tri,slots,K,axis,pos,bounds,1e-12*C.L,v['azimuth'],v['elevation'])
                prefix=stage+'_'+name+'_'+mode;offset=(col*size,head+row*size);page.paste(im,offset)
                arrays[prefix+'_pixel_raw_slot']=pixels
                for key,value in geom.items():arrays[prefix+'_'+key]=value
                if mode=='section':
                    err=float(np.max(np.abs(geom['segments'][:,:,axis]-pos))) if len(geom['segments']) else 0.
                    assert err<1e-9*C.L;checks.append(dict(panel=prefix,segments=len(geom['segments']),maximum_plane_error=err))
                else:
                    err=float(np.max(geom['clipped_triangles'][:,:,axis]-pos)) if len(geom['clipped_triangles']) else 0.
                    assert err<1e-9*C.L;checks.append(dict(panel=prefix,clipped_triangles=len(geom['clipped_triangles']),maximum_outside_halfspace=err))
                panels.append(dict(prefix=prefix,stage=stage,mode=mode,plane_axis=axis,plane_position=pos,offset_xy=list(offset),size=size))
    d.text((12,head+4*size+5),'A section can miss off-plane connections. Use the full indexed mesh and both legal/unfiltered edge graphs to establish disconnection.',font=R.font(20),fill='black')
    d.text((12,head+4*size+34),'POST source_face_ids preserve generated RAW slot identity only within this model/request. No complete source mesh appears here.',font=R.font(20),fill='black')
    d.text((12,head+4*size+60),'All clipping is for display only; saved meshes and exact per-segment/per-pixel source identities remain unchanged.',font=R.font(19),fill='#555555')
    with p.open('xb') as f:page.save(f,format='PNG')
    preview=dest/(rid+'_sections.png')
    with preview.open('xb') as f:page.crop((0,0,4*size,head+2*size)).save(f,format='PNG')
    identity=dest/(rid+'_identity.npz')
    with identity.open('xb') as f:np.savez_compressed(f,**arrays)
    r=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',label=label,request_id=rid,split=q['split'],path=str(p),image_path=str(p),image_sha256=C.sha(p),
        panels=panels,geometry_checks=checks,outputs={str(x):C.sha(x) for x in (p,preview,identity)},source_hashes=inputs,definition_sha256=dh,
        seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,GEO_runs=0)
    C.write(rp,r);print('FOUR_STAGE_RENDER',label,rid,flush=True);return r

def run(label):
    assert label in ARMS;dest=OUT/'sections'/label;dp=dest/'definition.json';done=dest/'completion.json'
    if done.exists():
        r=C.read(done)
        for pp in r['images']:
            for p,h in pp['outputs'].items():assert C.sha(p)==h
        return r
    protocol=C.read(PREVIOUS/'protocol.json');qs=requests()
    definition=dict(version='hinge_quality_four_stage_same_section_v1',label=label,
        previous_section_protocol_path=str(PREVIOUS/'protocol.json'),previous_section_protocol_sha256=C.sha(PREVIOUS/'protocol.json'),
        bounds=protocol['bounds'],sections=protocol['sections'],cutaway=protocol['cutaway'],
        geometry_hashes={str(mesh_path(label,q['request_id'],s)):C.sha(mesh_path(label,q['request_id'],s)) for q in qs for s in STAGES},
        sources={str(p):C.sha(p) for p in (Path(__file__),Path(R.__file__))},
        stage_identity='Generated RAW slots only; cross-model slot numbers are not source-face or semantic counterparts',
        review='PENDING_ACTUAL_VIEW; no labels manufactured from scores')
    C.write(dp,definition)
    jobs=[(label,q,str(dest),C.sha(dp),protocol) for q in qs];images,n,seconds=dispatch_cpu(jobs,one,dest,workers=2)
    (dest/'index.html').write_text('<meta charset="utf-8"><h1>'+label+' four saved stages</h1><p>Same historical section planes; exact clipped halfspace, purple artificial rim.</p>'+''.join(
        '<h2>'+r['request_id']+'</h2><a href="'+r['request_id']+'.png"><img style="width:100%" loading="lazy" src="'+r['request_id']+'.png"></a>' for r in images),encoding='utf-8')
    r=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',images=images,workers=n,seconds=seconds,definition_sha256=C.sha(dp));C.write(done,r);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label',choices=ARMS);a=p.parse_args();r=run(a.label);print(r['status'],len(r['images']),flush=True)
