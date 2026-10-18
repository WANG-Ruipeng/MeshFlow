"""CPU barycentric z-buffer supplement; same actual meshes, no metric/model calls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import io,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import common as C
DEST=C.OUT/'evaluation/depth_gallery'
SIZE=480
COLS=[('SMALL_C200','RAW'),('CREF_C200','RAW'),('SMALL_C200','POST'),('CREF_C200','POST')]

def basis(azimuth,elevation=23):
    a,e=np.deg2rad([azimuth,elevation]);d=np.array([np.cos(e)*np.cos(a),np.sin(e),np.cos(e)*np.sin(a)])
    right=np.cross([0,1,0],d);right/=np.linalg.norm(right);up=np.cross(d,right)
    return np.stack([right,up,d],axis=1)

def raster(tri,k,B,center,span,size=SIZE):
    v=(tri.astype(np.float64)-center)@B
    p=v.copy();p[:,:,0]=size*.5+v[:,:,0]*(size*.86/span);p[:,:,1]=size*.5-v[:,:,1]*(size*.86/span)
    img=np.full((size,size,3),255,np.uint8);depth=np.full((size,size),-np.inf);owner=np.full((size,size),-1,np.int32)
    for fi,t in enumerate(p):
        lo=np.maximum(np.floor(t[:,:2].min(0)).astype(int),0);hi=np.minimum(np.ceil(t[:,:2].max(0)).astype(int),size-1)
        if np.any(hi<lo):continue
        x0,y0=t[0,:2];x1,y1=t[1,:2];x2,y2=t[2,:2];den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den)<1e-12:continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
        w0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den;w1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;w2=1-w0-w1
        z=w0*t[0,2]+w1*t[1,2]+w2*t[2,2];ds=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1];valid=(w0>=-1e-9)&(w1>=-1e-9)&(w2>=-1e-9)&(z>ds+1e-12)
        normal=np.cross(v[fi,1]-v[fi,0],v[fi,2]-v[fi,0]);nn=np.linalg.norm(normal);shade=.67+.33*abs(normal[2])/nn if nn else .8
        base=np.array([22,134,192]) if fi<k else np.array([185,191,197]);rgb=np.broadcast_to((base*shade).astype(np.uint8),(*xx.shape,3)).copy()
        edge=(np.minimum(np.minimum(w0,w1),w2)<.008);rgb[edge]=(base*.52).astype(np.uint8)
        pixels=img[lo[1]:hi[1]+1,lo[0]:hi[0]+1];pixels[valid]=rgb[valid];ds[valid]=z[valid];own=owner[lo[1]:hi[1]+1,lo[0]:hi[0]+1];own[valid]=fi
    return Image.fromarray(img),owner

def main():
    DEST.mkdir(parents=True,exist_ok=True);r=C.read(C.OUT/'registration.json');idx={}
    for stage in ['raw','post']:
        for rec in C.read(C.OUT/'evaluation'/f'{stage}_index.json')['rows']:idx[rec['request_id'],rec['label'],stage.upper()]=rec
    meshes={};refs={}
    for key,rec in idx.items():
        field=key[2].lower();p=rec[field+'_path'];h=rec[field+'_sha256']
        if C.sha(p)!=h:raise ValueError('Geometry changed')
        a=C.arrays(p);meshes[key]=(a['raw'] if field=='raw' else a['output']).reshape(-1,3,3);refs[p]=h
    views=[basis(-60),basis(120)];bounds={}
    for cid in set(q['condition_id'] for q in r['requests']):
        pts=np.concatenate([meshes[q['request_id'],arm,stage].reshape(-1,3) for q in r['requests'] if q['condition_id']==cid for arm,stage in COLS])
        center=(pts.min(0)+pts.max(0))/2;span=max(np.ptp((pts-center)@B,axis=0)[:2].max() for B in views)
        bounds[cid]=dict(center=center.tolist(),span=float(span))
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontpath,17);small=ImageFont.truetype(fontpath,14);images=[]
    for q in r['requests']:
        page=Image.new('RGB',(SIZE*4,SIZE*2+120),'white');draw=ImageDraw.Draw(page)
        draw.text((10,8),q['parent_id'][:8]+' | FULL K='+str(q['K'])+' | n'+str(q['candidate'])+' | per-pixel CPU depth; blue C / gray free',font=font,fill='black')
        b=bounds[q['condition_id']];records=[]
        for col,(arm,stage) in enumerate(COLS):
            draw.text((col*SIZE+14,40),arm+' '+stage,font=font,fill='black')
            for row,B in enumerate(views):
                mesh=meshes[q['request_id'],arm,stage];im,owner=raster(mesh,q['K'],B,np.array(b['center']),b['span']);page.paste(im,(col*SIZE,65+row*SIZE))
                records.append(dict(label=arm,stage=stage,view=row,visible_C_pixels=int(((owner>=0)&(owner<q['K'])).sum()),visible_F_pixels=int((owner>=q['K']).sum())))
        draw.text((10,SIZE*2+72),'Two opposite orthographic views; same physical scale within condition. No geometry edit or topology certification; refer to finite sections too.',font=small,fill='black')
        p=DEST/(q['request_id']+'.png');buf=io.BytesIO();page.save(buf,format='PNG');C._write_bytes(p,buf.getvalue())
        images.append(dict(request_id=q['request_id'],path=str(p),sha256=C.sha(p),pixels=list(page.size),visibility=records))
        print(json.dumps(dict(depth_render_complete=len(images),total=24)),flush=True)
    v=dict(status='COMPLETE',version='CPU_BARYCENTRIC_ZBUFFER_V1',registration_sha256=C.sha(C.OUT/'registration.json'),source_code_sha256=C.sha(__file__),images=images,source_hashes=refs,bounds=bounds,views=[B.tolist() for B in views],new_geometry=0,model_calls=0,scoring_calls=0,
        limitation='Floating-point exact-depth ties favor earlier C face; unrelated pixels are not forced blue. Orthographic pixel raster can miss subpixel gaps. No MSAA. This supplements unchanged finite-cut V1 figures and original reviews, not semantic adjudication.')
    C.write(DEST/'manifest.json',v);C._write_bytes(DEST/'index.html',('<meta charset="utf-8"><h1>Depth-correct supplement</h1>'+''.join('<h2>'+x['request_id']+'</h2><img style="width:100%" src="'+Path(x['path']).name+'">' for x in images)).encode())
    C.event('DEPTH_DISPLAY_SUPPLEMENT_COMPLETE',images=24,model_calls=0,geometry_changes=0)
if __name__=='__main__':main()