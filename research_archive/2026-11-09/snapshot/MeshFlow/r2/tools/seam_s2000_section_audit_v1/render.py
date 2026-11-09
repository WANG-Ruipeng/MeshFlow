"""Saved S2000 finite sections and exact halfspace cutaways; CPU, no model/GEO."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, concurrent.futures, hashlib, html, json, multiprocessing, resource, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_s2000_section_audit_v1'
DEST=OUT/'gallery'
STAGES=('RAW','D0','RESTORED','POST')
SIZE=700
HEADER=110
FOOTER=85
BLUE=np.array([22,134,192.]);GRAY=np.array([185,191,197.]);PURPLE=np.array([174,52,171.])

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def resolve(p):
    p=str(p).replace('\\','/')
    if os.name!='nt' and len(p)>2 and p[1]==':':p='/mnt/'+p[0].lower()+p[2:]
    return Path(p)
def resources():
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        a=line.split();mem[a[0].rstrip(':')]=int(a[1])*1024
    return dict(unix=time.time(),logical_cores=os.cpu_count(),load_average=list(os.getloadavg()),
        available_RAM_bytes=mem.get('MemAvailable'),process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        CPU_only=True,threads_per_worker=1,GPU_dispatches=0)
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def basis(az,el):
    a,e=np.deg2rad([az,el]);d=np.array([np.cos(e)*np.cos(a),np.sin(e),np.cos(e)*np.sin(a)])
    right=np.cross([0,1,0],d);right/=np.linalg.norm(right);up=np.cross(d,right)
    return np.stack([right,up,d],axis=1)

def finite_sections(tri,axis,pos,eps):
    """Historical finite intersection arithmetic, retaining every source face."""
    segments=[];owners=[];coplanar=[]
    for index,face in enumerate(np.asarray(tri,dtype=np.float64)):
        delta=face[:,axis]-float(pos)
        if np.all(np.abs(delta)<=eps):
            for a,b in ((0,1),(1,2),(2,0)):
                if np.linalg.norm(face[a]-face[b])>eps:
                    segments.append(face[[a,b]]);owners.append(index);coplanar.append(True)
            continue
        points=[]
        for a,b in ((0,1),(1,2),(2,0)):
            if abs(delta[a])<=eps:points.append(face[a])
            if delta[a]*delta[b]<0:points.append(face[a]+(face[b]-face[a])*(-delta[a]/(delta[b]-delta[a])))
        unique=[]
        for point in points:
            if not any(np.linalg.norm(point-prior)<=eps for prior in unique):unique.append(point)
        if len(unique)>=2:
            a,b=max(((a,b) for a in range(len(unique)) for b in range(a+1,len(unique))),key=lambda ab:np.linalg.norm(unique[ab[0]]-unique[ab[1]]))
            if np.linalg.norm(unique[a]-unique[b])>eps:
                segments.append([unique[a],unique[b]]);owners.append(index);coplanar.append(False)
    return np.asarray(segments,dtype=np.float64).reshape(-1,2,3),np.asarray(owners,dtype=np.int32),np.asarray(coplanar,dtype=bool)

def clip_triangles(tri,axis,pos,eps):
    """Sutherland-Hodgman finite polygon clipping; no centroid selection/cap."""
    clipped=[];owners=[];rims=[];rimowners=[]
    for fi,face in enumerate(np.asarray(tri,dtype=np.float64)):
        poly=[]
        for i in range(3):
            a,b=face[i],face[(i+1)%3];da=a[axis]-pos;db=b[axis]-pos
            ia=da<=0;ib=db<=0
            if ia:poly.append(a.copy())
            if ia!=ib:
                point=a+(b-a)*(-da/(db-da));point[axis]=pos;poly.append(point)
        clean=[]
        for point in poly:
            if not clean or np.linalg.norm(point-clean[-1])>eps:clean.append(point)
        if len(clean)>1 and np.linalg.norm(clean[0]-clean[-1])<=eps:clean.pop()
        if len(clean)<3:continue
        for j in range(1,len(clean)-1):
            f=np.asarray([clean[0],clean[j],clean[j+1]])
            if np.linalg.norm(np.cross(f[1]-f[0],f[2]-f[0]))>eps*eps:clipped.append(f);owners.append(fi)
        # Only newly produced rims of genuinely crossing faces; coplanar C is not a cap/rim.
        delta=face[:,axis]-pos
        if np.min(delta)<-eps and np.max(delta)>eps:
            on=[x for x in clean if abs(x[axis]-pos)<=eps]
            if len(on)>=2:
                a,b=max(((a,b) for a in range(len(on)) for b in range(a+1,len(on))),key=lambda ab:np.linalg.norm(on[ab[0]]-on[ab[1]]))
                rims.append([on[a],on[b]]);rimowners.append(fi)
    return (np.asarray(clipped,dtype=np.float64).reshape(-1,3,3),np.asarray(owners,dtype=np.int32),
        np.asarray(rims,dtype=np.float64).reshape(-1,2,3),np.asarray(rimowners,dtype=np.int32))

def sections_panel(tri,slots,K,axis,pos,bounds,eps):
    project=(0,1) if axis==2 else (2,1)
    seg,owner,coplanar=finite_sections(tri,axis,pos,eps);low=np.asarray(bounds['local_low'])[list(project)];high=np.asarray(bounds['local_high'])[list(project)]
    span=high-low;scale=min((SIZE-105)/span[0],(SIZE-145)/span[1]);center=(low+high)/2
    def pixel(x):
        v=(x[...,list(project)]-center)*scale
        return np.stack([SIZE/2+v[...,0],SIZE/2-v[...,1]],axis=-1)
    px=pixel(seg);im=Image.new('RGB',(SIZE,SIZE),'white');d=ImageDraw.Draw(im)
    pixowner=Image.new('I',(SIZE,SIZE),-1);od=ImageDraw.Draw(pixowner)
    # All free first then C, matching historical source-C overlay priority.
    order=sorted(range(len(seg)),key=lambda j:slots[owner[j]]<K)
    for j in order:
        c=tuple(BLUE.astype(int)) if slots[owner[j]]<K else (116,123,131)
        line=[tuple(x) for x in px[j]];width=3 if slots[owner[j]]<K else 2
        d.line(line,fill=c,width=width);od.line(line,fill=int(slots[owner[j]]),width=width)
    # Match historical axes clipping: finite lines outside inherited local
    # limits must not appear in plot margins. Full uncut segments stay in NPZ.
    xlo=SIZE/2-span[0]*scale/2;xhi=SIZE/2+span[0]*scale/2;ylo=SIZE/2-span[1]*scale/2;yhi=SIZE/2+span[1]*scale/2
    for rect in ((0,0,SIZE,int(np.ceil(ylo))-1),(0,int(np.floor(yhi))+1,SIZE,SIZE),
                 (0,0,int(np.ceil(xlo))-1,SIZE),(int(np.floor(xhi))+1,0,SIZE,SIZE)):
        d.rectangle(rect,fill='white');od.rectangle(rect,fill=-1)
    d.rectangle((xlo,ylo,xhi,yhi),outline='#dddddd',width=1)
    for frac in (0,.5,1):
        x=xlo+frac*(xhi-xlo);y=yhi-frac*(yhi-ylo)
        d.text((x-20,yhi+5),f'{low[0]+frac*span[0]:.2f}',font=font(15),fill='#555555')
        d.text((max(0,xlo-52),y-7),f'{low[1]+frac*span[1]:.2f}',font=font(15),fill='#555555')
    name='front: z' if axis==2 else 'side: x'
    d.text((12,12),f'Finite section {name} = {pos:.8f}',font=font(20),fill='black')
    d.text((12,SIZE-30),('X / Y' if axis==2 else 'Z / Y')+' coordinates; saved local bounds; no silhouette projection',font=font(16),fill='#555555')
    return im,np.asarray(pixowner,dtype=np.int32),dict(segments=seg,face_indices=owner,raw_slots=slots[owner],coplanar=coplanar,pixel_endpoints=px,scale=scale,center=center)

def cutaway_panel(tri,slots,K,axis,pos,bounds,eps,az,el):
    clipped,owner,rims,rimowners=clip_triangles(tri,axis,pos,eps)
    B=basis(az,el);center=np.asarray(bounds['whole_center']);span=bounds['whole_span'];scale=SIZE*.86/span
    def project(t):
        v=(t-center)@B;p=v.copy();p[...,0]=SIZE*.5+v[...,0]*scale;p[...,1]=SIZE*.5-v[...,1]*scale
        return p,v
    p,v=project(clipped);img=np.full((SIZE,SIZE,3),255,np.uint8);depth=np.full((SIZE,SIZE),-np.inf);pixowner=np.full((SIZE,SIZE),-1,np.int32)
    for j,t in enumerate(p):
        lo=np.maximum(np.floor(t[:,:2].min(0)).astype(int),0);hi=np.minimum(np.ceil(t[:,:2].max(0)).astype(int),SIZE-1)
        if np.any(hi<lo):continue
        x0,y0=t[0,:2];x1,y1=t[1,:2];x2,y2=t[2,:2];den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den)<1e-12:continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
        w0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den;w1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;w2=1-w0-w1
        z=w0*t[0,2]+w1*t[1,2]+w2*t[2,2];ds=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        valid=(w0>=-1e-9)&(w1>=-1e-9)&(w2>=-1e-9)&(z>ds+1e-12)
        normal=np.cross(v[j,1]-v[j,0],v[j,2]-v[j,0]);nn=np.linalg.norm(normal);shade=.67+.33*abs(normal[2])/nn if nn else .8
        color=BLUE if slots[owner[j]]<K else GRAY
        pixels=img[lo[1]:hi[1]+1,lo[0]:hi[0]+1];pixels[valid]=(color*shade).astype(np.uint8);ds[valid]=z[valid]
        own=pixowner[lo[1]:hi[1]+1,lo[0]:hi[0]+1];own[valid]=slots[owner[j]]
    # Purple is only a rendering-created cut rim, depth tested (no cap).
    rp,_=project(rims)
    for j,line in enumerate(rp):
        count=max(2,int(np.linalg.norm(line[1,:2]-line[0,:2])*3)+1)
        points=line[0]+np.linspace(0,1,count)[:,None]*(line[1]-line[0]);xy=np.floor(points[:,:2]).astype(int)
        for point,(x,y) in zip(points,xy):
            if 0<=x<SIZE and 0<=y<SIZE and point[2]>=depth[y,x]-2/scale:
                img[y,x]=PURPLE.astype(np.uint8);pixowner[y,x]=slots[rimowners[j]]
    im=Image.fromarray(img);d=ImageDraw.Draw(im);name='z' if axis==2 else 'x'
    d.text((12,12),f'3D cutaway: keep {name} <= {pos:.8f}',font=font(20),fill='black')
    d.text((12,40),f'az {az}, elev {el}; purple = artificial cut rim; NO CAP',font=font(16),fill='#63315f')
    return im,pixowner,dict(clipped_triangles=clipped,face_indices=owner,raw_slots=slots[owner],cut_rims=rims,cut_rim_raw_slots=slots[rimowners],camera_basis=B,center=center,scale=scale)

def one(job):
    q,definition_hash,protocol=job;rid=q['request_id'];p=DEST/(rid+'.png');rp=p.with_suffix('.json')
    if rp.exists():
        r=read(rp)
        if r['definition_sha256']!=definition_hash:raise ValueError('Definition mismatch')
        for f,h in r['outputs'].items():
            if sha(f)!=h:raise ValueError('Existing output changed: '+f)
        return r
    if p.exists():raise ValueError('Unreceipted image; preserve, no automatic overwrite')
    tick=time.perf_counter();K=q['K'];L=protocol['L'];bounds=protocol['bounds'][q['condition_id']];inputs={};meshes={};identities={}
    for stage in STAGES:
        path=resolve(q['paths'][stage]);h=sha(path)
        if h!=protocol['source_hashes'][q['paths'][stage]]:raise ValueError('Input hash mismatch')
        inputs[str(path)]=h
        with np.load(path,allow_pickle=False) as a:
            key='raw' if stage=='RAW' else 'official_unconstrained_proposal' if stage=='D0' else 'output'
            meshes[stage]=a[key].astype(np.float64)
            identities[stage]=a['source_face_ids'].astype(np.int32).reshape(-1) if stage=='POST' else np.arange(len(meshes[stage]),dtype=np.int32)
        tri=meshes[stage];ids=identities[stage]
        assert tri.ndim==3 and tri.shape[1:]==(3,3) and np.isfinite(tri).all()
        assert len(tri)==len(ids) and len(set(ids.tolist()))==len(ids) and ids.min()>=0 and ids.max()<q['N']
        assert np.array_equal(ids[:K],np.arange(K))
    page=Image.new('RGB',(4*SIZE,HEADER+4*SIZE+FOOTER),'white');draw=ImageDraw.Draw(page)
    draw.text((12,8),rid+' | '+q['split']+' | PLAIN_S2000 | four saved stages, identical inherited planes and scale',font=font(24),fill='black')
    draw.text((12,40),'C blue / free gray. D0 blue C HAS MOVED. Cut rims purple are artificial, not cracks. No model or GEO run.',font=font(21),fill='#973030')
    arrays={};panels=[]
    for col,stage in enumerate(STAGES):
        draw.text((col*SIZE+12,77),stage+' proposal (C moved)' if stage=='D0' else stage,font=font(24),fill='black')
        tri=meshes[stage];slots=identities[stage]
        for vi,spec in enumerate(protocol['sections']):
            axis=spec['axis'];pos=bounds[spec['position_field']][spec['position_index']];name=spec['name']
            for mode in ('section','cutaway'):
                row=vi if mode=='section' else vi+2
                if mode=='section':im,pixels,geom=sections_panel(tri,slots,K,axis,pos,bounds,1e-12*L)
                else:
                    view=protocol['cutaway'][name];im,pixels,geom=cutaway_panel(tri,slots,K,axis,pos,bounds,1e-12*L,view['azimuth'],view['elevation'])
                prefix=stage+'_'+name+'_'+mode;offset=(col*SIZE,HEADER+row*SIZE);page.paste(im,offset)
                arrays[prefix+'_pixel_raw_slot']=pixels
                for key,value in geom.items():arrays[prefix+'_'+key]=value
                panels.append(dict(prefix=prefix,stage=stage,mode=mode,plane_axis=axis,plane_position=pos,offset_xy=list(offset),size=SIZE,visible_raw_slots=np.unique(pixels[pixels>=0]).tolist()))
    draw.text((12,HEADER+4*SIZE+5),'Finite cuts can miss off-plane faces. Coplanar triangle edges may overprint. An isolated cut segment does not prove a detached face.',font=font(20),fill='black')
    draw.text((12,HEADER+4*SIZE+34),'POST raw-slot mapping comes from saved source_face_ids. Complete original indexed POST, not this clipped view, defines connectivity.',font=font(20),fill='black')
    draw.text((12,HEADER+4*SIZE+60),'All clipped coordinates are rendering-only; untouched saved meshes and exact per-segment/per-pixel source IDs remain auditable.',font=font(17),fill='#555555')
    with p.open('xb') as f:page.save(f,format='PNG')
    preview=DEST/(rid+'_sections.png')
    with preview.open('xb') as f:page.crop((0,0,4*SIZE,HEADER+2*SIZE)).save(f,format='PNG')
    data=DEST/(rid+'_identity.npz')
    with data.open('xb') as f:np.savez_compressed(f,**arrays)
    r=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',request_id=rid,split=q['split'],image_path=str(p),image_sha256=sha(p),
        sections_preview=str(preview),identity_npz=str(data),outputs={str(x):sha(x) for x in (p,preview,data)},source_hashes=inputs,
        definition_sha256=definition_hash,panels=panels,seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_scope='Process lifetime high-water including imports, full page and compressed identity archive',pid=os.getpid(),
        model_calls=0,GPU_calls=0,GEO_runs=0,input_mesh_mutations=0)
    write(rp,r);print('RENDERED',rid,round(r['seconds'],2),flush=True);return r

def run(first_only=False):
    DEST.mkdir(parents=True,exist_ok=True);protocol=read(OUT/'protocol.json')
    assert protocol['status']=='FROZEN' and len(protocol['requests'])==28
    hist=resolve(protocol['historical_definition']['path']);assert sha(hist)==protocol['historical_definition']['sha256']
    definition=dict(version='s2000_sections_cpu_render_v1',protocol_sha256=sha(OUT/'protocol.json'),code_sha256=sha(__file__),
        historical_definition=protocol['historical_definition'],bounds=protocol['bounds'],sections=protocol['sections'],cutaway=protocol['cutaway'],
        stages=list(STAGES),rows=['front finite section','side finite section','front exact 3D halfspace','side exact 3D halfspace'],
        panel_pixels=SIZE,header_pixels=HEADER,epsilon=1e-12*protocol['L'],colors=dict(C=BLUE.tolist(),free=GRAY.tolist(),artificial_cut=PURPLE.tolist()),
        identity='Each NPZ pixel_raw_slot is panel-local; page coordinates use panel offset_xy. All finite segment and clipped triangle identities retain original RAW slot. Background=-1. Section overdraw favors C; use full segment arrays for occluded identities.',
        raster='FP64 finite barycentric z-buffer, no backface culling, no artificial cap, exact ties favor earlier C; purple rim depth visibility tolerance 2 screen pixels.',
        limitations='Isolated finite lines and cutaway boundaries do not establish whole-mesh disconnectedness. D0 moves C. No outward-normal certification.')
    dp=DEST/'definition.json'
    if dp.exists():assert read(dp)==definition
    else:write(dp,definition)
    if (DEST/'completion.json').exists():
        prior=read(DEST/'completion.json')
        for r in prior['pages']:
            for f,h in r['outputs'].items():assert sha(f)==h
        print('COMPLETE_VERIFIED',len(prior['pages']));return
    tick=time.perf_counter();before=resources();jobs=[(q,sha(dp),protocol) for q in protocol['requests']]
    first=one(jobs[0]);after=resources();available=after['available_RAM_bytes'] or 0;peak=first['peak_RSS_KiB']*1024
    workers=2 if (after['logical_cores'] or 1)>=3 and available>2.4*peak+(1<<30) else 1
    write(DEST/('dispatch_'+str(time.time_ns())+'.json'),dict(before=before,after_first_actual_page=after,first_page=first,
        actual_workers=1 if first_only else workers,max_workers=2,first_only=first_only,
        reason='Actual authorized first page measured, no probe; two independent page workers only after RAM/CPU check; one thread each'))
    if first_only:print('FIRST_READY',first['image_path'],flush=True);return
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pages=[first,*list(pool.map(one,jobs[1:]))]
    write(DEST/'pages.json',dict(definition_sha256=sha(dp),pages=pages))
    (DEST/'index.html').write_text('<meta charset="utf-8"><h1>S2000 saved four-stage sections</h1><p>Blue C; gray free. Purple cut rims are artificial. D0 moves C. Isolated section lines are not automatically detached faces.</p>'+''.join(
        '<h2>'+html.escape(r['request_id'])+' '+r['split']+'</h2><p><a href="'+r['request_id']+'.png">Four-stage full sections + cutaways</a> | <a href="'+r['request_id']+'_sections.png">Section-only preview</a></p><img loading="lazy" style="width:100%" src="'+r['request_id']+'.png">' for r in pages),encoding='utf-8')
    write(DEST/'completion.json',dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',pages=pages,workers=workers,
        wall_seconds=time.perf_counter()-tick,resources_after=resources(),model_calls=0,GPU_calls=0,GEO_runs=0,input_mesh_mutations=0))
    print('COMPLETE',len(pages),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--first',action='store_true');args=parser.parse_args()
    if args.run or args.first:run(first_only=args.first)
    else:print('PREPARED_CPU_ONLY: --first first actual frozen page; --run all28 with resume')
