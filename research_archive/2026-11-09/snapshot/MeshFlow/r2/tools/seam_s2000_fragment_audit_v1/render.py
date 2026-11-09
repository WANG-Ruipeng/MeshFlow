"""CPU-only saved four-stage meshes; no torch, neural code, GPU or mesh edits."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, concurrent.futures, hashlib, html, json, math, multiprocessing, resource, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_s2000_fragment_audit_v1'
DEST=OUT/'gallery'
SIZE=600
VIEWS=((-60,23),(120,23))
STAGES=('RAW','D0','RESTORED','POST')
BLUE=np.array([22,134,192.])
GRAY=np.array([185,191,197.])
COLORS={'small_angle':[225,47,53],'tiny_area':[237,142,32],'both':[161,60,180]}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def resolve(p):
    p=str(p).replace('\\','/')
    if os.name!='nt' and len(p)>2 and p[1]==':':p='/mnt/'+p[0].lower()+p[2:]
    return Path(p)
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def basis(azimuth,elevation):
    a,e=np.deg2rad([azimuth,elevation]);d=np.array([np.cos(e)*np.cos(a),np.sin(e),np.cos(e)*np.sin(a)])
    right=np.cross([0,1,0],d);right/=np.linalg.norm(right);up=np.cross(d,right)
    return np.stack([right,up,d],axis=1)
def face_flags(tri,K,L):
    t=tri.astype(np.float64);cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);area=.5*np.linalg.norm(cross,axis=1)
    angles=[]
    for i in range(3):
        u=t[:,(i+1)%3]-t[:,i];v=t[:,(i+2)%3]-t[:,i]
        angles.append(np.degrees(np.arctan2(np.linalg.norm(np.cross(u,v),axis=1),np.einsum('ij,ij->i',u,v))))
    small=np.min(np.stack(angles,axis=1),axis=1)<5.;tiny=area/(L*L)<1e-5
    small[:K]=False;tiny[:K]=False
    return small,tiny

def raster(tri,K,B,center,span,colors):
    # Small adaptation of the historical finite barycentric CPU z-buffer.
    # No face sorting/geometry edit; exact depth ties favor earlier source C.
    v=(tri.astype(np.float64)-center)@B;p=v.copy()
    p[:,:,0]=SIZE*.5+v[:,:,0]*(SIZE*.86/span);p[:,:,1]=SIZE*.5-v[:,:,1]*(SIZE*.86/span)
    img=np.full((SIZE,SIZE,3),255,np.uint8);depth=np.full((SIZE,SIZE),-np.inf);owner=np.full((SIZE,SIZE),-1,np.int32)
    for fi,t in enumerate(p):
        lo=np.maximum(np.floor(t[:,:2].min(0)).astype(int),0);hi=np.minimum(np.ceil(t[:,:2].max(0)).astype(int),SIZE-1)
        if np.any(hi<lo):continue
        x0,y0=t[0,:2];x1,y1=t[1,:2];x2,y2=t[2,:2];den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den)<1e-12:continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
        w0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den;w1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;w2=1-w0-w1
        z=w0*t[0,2]+w1*t[1,2]+w2*t[2,2];ds=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        valid=(w0>=-1e-9)&(w1>=-1e-9)&(w2>=-1e-9)&(z>ds+1e-12)
        normal=np.cross(v[fi,1]-v[fi,0],v[fi,2]-v[fi,0]);nn=np.linalg.norm(normal)
        shade=.67+.33*abs(normal[2])/nn if nn else .8
        base=colors[fi];rgb=np.broadcast_to((base*shade).astype(np.uint8),(*xx.shape,3)).copy()
        # Approximately 0.6 screen-pixel wire, depth-tested with its own face.
        side=np.array([np.linalg.norm(t[1,:2]-t[2,:2]),np.linalg.norm(t[2,:2]-t[0,:2]),np.linalg.norm(t[0,:2]-t[1,:2])])
        alt=abs(den)/np.maximum(side,1e-12)
        edge=np.minimum(np.minimum(w0*alt[0],w1*alt[1]),w2*alt[2])<.6
        rgb[edge]=(base*.55).astype(np.uint8)
        pixels=img[lo[1]:hi[1]+1,lo[0]:hi[0]+1];pixels[valid]=rgb[valid];ds[valid]=z[valid]
        own=owner[lo[1]:hi[1]+1,lo[0]:hi[0]+1];own[valid]=fi
    return Image.fromarray(img),owner

def resources():
    mem={}
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            a=line.split();mem[a[0].rstrip(':')]=int(a[1])*1024
    return dict(unix=time.time(),logical_cores=os.cpu_count(),load_average=list(os.getloadavg()) if hasattr(os,'getloadavg') else None,
                available_RAM_bytes=mem.get('MemAvailable'),process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                CPU_only=True,threads_per_worker=1,GPU_dispatches=0)

def one(job):
    q,definition_hash,db,source_hashes,L=job
    p=DEST/(q['request_id']+'.png');rp=p.with_suffix('.json')
    if rp.exists():
        r=read(rp)
        if r['definition_sha256']!=definition_hash or sha(p)!=r['image_sha256']:raise ValueError('Existing page changed')
        return r
    if p.exists():raise ValueError('Unreceipted PNG; preserve and inspect before explicit repair')
    tick=time.perf_counter();meshes={};inputs={};K=q['K']
    for stage in STAGES:
        path=resolve(q['paths'][stage]);actual=sha(path)
        if actual!=source_hashes[q['paths'][stage]]:raise ValueError('Input SHA mismatch: '+str(path))
        inputs[str(path)]=actual
        with np.load(path,allow_pickle=False) as a:
            key='raw' if stage=='RAW' else 'official_unconstrained_proposal' if stage=='D0' else 'output'
            meshes[stage]=a[key].copy()
        if meshes[stage].ndim!=3 or meshes[stage].shape[1:]!=(3,3) or not np.isfinite(meshes[stage]).all():raise ValueError('Invalid saved triangles')
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,23);sm=ImageFont.truetype(fontpath,18);tinyfont=ImageFont.truetype(fontpath,16)
    page=Image.new('RGB',(5*SIZE,2*SIZE+200),'white');draw=ImageDraw.Draw(page)
    draw.text((15,10),q['request_id']+' | PLAIN_S2000 | '+q['split']+' | identical historical scale, saved geometry only',font=font,fill='black')
    draw.text((15,43),'C blue / free gray. D0 blue triangles HAVE MOVED: this column does not preserve C and is not a valid completion.',font=sm,fill='#9c2424')
    headings=('RAW','D0 proposal: C moved','RESTORED anchors','POST: original GEO','POST diagnostic colors')
    small,tiny=face_flags(meshes['POST'],K,L);visibility=[]
    for col,stage in enumerate((*STAGES,'POST')):
        draw.text((col*SIZE+15,77),headings[col],font=font,fill='black')
        tri=meshes[stage];colors=np.tile(GRAY,(len(tri),1));colors[:K]=BLUE
        if col==4:
            colors[small&~tiny]=COLORS['small_angle'];colors[tiny&~small]=COLORS['tiny_area'];colors[small&tiny]=COLORS['both']
        for row,(az,el) in enumerate(VIEWS):
            im,owner=raster(tri,K,basis(az,el),np.array(db['center']),float(db['span']),colors)
            page.paste(im,(col*SIZE,110+row*SIZE))
            draw.text((col*SIZE+12,113+row*SIZE),'az '+str(az)+' / elev '+str(el),font=tinyfont,fill='#444444')
            visibility.append(dict(column=col,stage=stage,azimuth=az,elevation=el,C_visible_pixels=int(((owner>=0)&(owner<K)).sum()),free_visible_pixels=int((owner>=K).sum())))
    draw.text((15,2*SIZE+119),'Diagnostic: red=min angle<5 deg; orange=area/L^2<1e-5; purple=both. Thin depth-tested triangle wire.',font=sm,fill='black')
    draw.text((15,2*SIZE+146),'Highlights are descriptive, not automatic defects. RAW soup components are not visible-fragment labels; raster may miss subpixel gaps.',font=sm,fill='black')
    draw.text((15,2*SIZE+173),'No geometry edit, weld, deletion, model or GEO call. Images cannot certify connectivity or the absence of intersections.',font=tinyfont,fill='#555555')
    with p.open('xb') as f:page.save(f,format='PNG')
    r=dict(status='PENDING_ACTUAL_IMAGE_REVIEW',request_id=q['request_id'],split=q['split'],page_path=str(p),image_sha256=sha(p),
        definition_sha256=definition_hash,source_hashes=inputs,pixels=list(page.size),columns=list(headings),views=list(VIEWS),
        visible_pixels=visibility,seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_scope='Process lifetime high-water including NumPy/PIL imports and this actual page, not allocator-only',
        pid=os.getpid(),model_calls=0,GPU_calls=0,mesh_edits=0)
    write(rp,r);print('RENDERED',q['request_id'],round(r['seconds'],2),flush=True);return r

def run():
    DEST.mkdir(parents=True,exist_ok=True)
    protocol=read(OUT/'protocol.json');assert protocol['status']=='FROZEN' and len(protocol['requests'])==28
    historical=ROOT/'experiments/r2_seam_pool47_couple_v1/gallery/training/definition.json'
    db=read(historical)['depth_bounds']
    definition=dict(version='saved_s2000_fragment_four_stage_cpu_v1',protocol_sha256=sha(OUT/'protocol.json'),
        code_sha256=sha(__file__),historical_frame=str(historical),historical_frame_sha256=sha(historical),
        depth_bounds=db,views=list(VIEWS),columns=[*STAGES,'POST_DIAGNOSTIC'],raster_size=SIZE,wire_screen_pixels=.6,
        diagnostic_thresholds=dict(min_angle_degrees=5,tiny_area_over_L2=1e-5),colors=COLORS,
        historical_raster_reference='r2/pilots/regular_patch_control_v1/depth_gallery.py: finite barycentric z-buffer adapted locally without imports',
        limitations='D0 is unconstrained C moved. Highlighted faces need not be defects. Visible fragments require actual image review. No topology certification.')
    dp=DEST/'definition.json'
    if dp.exists():assert read(dp)==definition
    else:write(dp,definition)
    if (DEST/'completion.json').exists():
        prior=read(DEST/'completion.json')
        for r in prior['images']:assert sha(r['page_path'])==r['image_sha256']
        print('COMPLETE_VERIFIED',len(prior['images']));return
    tick=time.perf_counter();before=resources();dh=sha(dp)
    jobs=[(q,dh,db[q['condition_id']],protocol['source_hashes'],protocol['L']) for q in protocol['requests']]
    first=one(jobs[0]);after=resources();available=after['available_RAM_bytes'] or 0;peak=first['peak_RSS_KiB']*1024
    workers=2 if (after['logical_cores'] or 1)>=3 and available>2.4*peak+(1<<30) else 1
    write(DEST/('dispatch_'+str(time.time_ns())+'.json'),dict(before=before,after_first_actual_page=after,first_page=first,
        actual_workers=workers,max_workers=2,reason='First authorized real page measured, then independent pages at one thread each; no synthetic probe'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        images=[first,*list(pool.map(one,jobs[1:]))]
    write(DEST/'pages.json',dict(definition_sha256=dh,pages=images))
    (DEST/'index.html').write_text('<meta charset="utf-8"><h1>S2000 saved four-stage fragment audit</h1><p>D0 column moves C; diagnostic colors do not certify defects.</p>'+''.join(
        '<h2>'+html.escape(r['request_id'])+' '+r['split']+'</h2><a href="'+r['request_id']+'.png"><img loading="lazy" style="width:100%" src="'+r['request_id']+'.png"></a>' for r in images),encoding='utf-8')
    write(DEST/'completion.json',dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',images=images,workers=workers,
        wall_seconds=time.perf_counter()-tick,resources_after=resources(),model_calls=0,GPU_calls=0,mesh_edits=0))
    print('COMPLETE',len(images),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:print('PREPARED_CPU_ONLY: --run renders exactly the frozen28 pages')
