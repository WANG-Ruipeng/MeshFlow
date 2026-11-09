"""Paired scientific CPU figures, fixed scales, no invented review labels."""
from __future__ import annotations
import os
for _n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_n]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import html
import io
import json
import sys
from pathlib import Path
import numpy as np
from r2.pilots.seam_slot_c200_v1 import common as C
from r2.pilots.seam_slot_c200_v1.render import REVIEW_FIELDS, write_bytes
from r2.pilots.regular_patch_control_v1 import depth_gallery as D
from r2.tools.cref_layer_audit_v1 import render as H
from . import evaluate as E

VERSION='occupancy_three_layout_two_backend_depth_sections_v1'
GROUPS={'RAW':[(a,'RAW') for a in E.ARMS],
        'POST':[(a,s) for a in E.ARMS for s in ('POST','CAPACITY_POST')]}

def run(root):
    root=C.resolve(root);m=E.manifest(root);dest=root/'gallery';done=dest/'completion.json'
    if done.exists():
        r=C.read(done);E.E.verify(r['files']);E.E.verify(r['source_hashes']);E.E.verify(r['input_hashes']);return r
    source_hashes={str(p):C.sha(p) for p in (Path(__file__),Path(D.__file__),Path(H.__file__),Path(E.__file__))}
    definition=dict(version=VERSION,groups=GROUPS,rows=['depth_view_minus60','depth_view_plus120','finite_front_section','finite_side_section'],
        colors=dict(C='blue',free='gray'),scale='Same physical whole-view scale across all 4 noises, all 3 layouts, RAW and both POST backends within a C.',
        sections='C-only bound central finite sections, unchanged previous renderer.',
        source_oracle='DIAGNOSTIC source-label assisted layout; not deployment or generalization.',
        limitations='Exact depth ties favor C. Display cannot certify exact attachment/illegal contacts. Purple source winding is not externally certified outward. No automatic review labels.',
        model_calls=0,geometry_changes=0,scoring_calls=0,actual_reviews=0,nonblind=True)
    C.write(dest/'definition.json',definition)
    cells={};protected={};views=[D.basis(-60,23),D.basis(120,23)]
    columns=list(dict.fromkeys(x for cc in GROUPS.values() for x in cc))
    for q in m['requests']:
        inp=C.arrays(q['inference_npz']);original=inp['C_vertices'][inp['C_faces_original']]
        protected[q['inference_npz']]=C.sha(q['inference_npz'])
        for arm,stage in columns:
            p=E.artifact(root,arm,q['request_id'],stage);a=C.arrays(p);tri=a['output'].reshape(-1,3,3)
            if not np.array_equal(tri[:q['K']],original):raise ValueError('C changed in gallery')
            protected[str(p)]=C.sha(p);cells[q['request_id'],arm,stage]=dict(C=tri[:q['K']],F=tri[q['K']:],C_original=original,tri=tri,path=str(p),sha256=C.sha(p))
    bounds={};depth={}
    for cond in m['conditions']:
        cid=cond['condition_id'];qq=[q for q in m['requests'] if q['condition_id']==cid]
        gg=[cells[q['request_id'],a,s] for q in qq for a,s in columns]
        bounds[cid]=H.bounds_for_condition(gg,None,C.L)
        points=np.concatenate([g['tri'].reshape(-1,3) for g in gg]);center=(points.min(0)+points.max(0))/2
        span=max(np.ptp((points-center)@b,axis=0)[:2].max() for b in views)
        depth[cid]=dict(center=center.tolist(),span=float(span))
    C.write(dest/'bounds.json',dict(depth=depth,sections=bounds));plt=H._mpl();pages=[];templates=[]
    for q in m['requests']:
        rid=q['request_id'];cid=q['condition_id'];db=depth[cid]
        for group,cols in GROUPS.items():
            p=dest/(rid+'_'+group+'.png');rp=p.with_suffix('.json')
            ident=dict(version=VERSION,request_id=rid,group=group,source_hashes=source_hashes,depth=db,sections=bounds[cid],
                geometry=[dict(label=a,stage=s,path=cells[rid,a,s]['path'],sha256=cells[rid,a,s]['sha256']) for a,s in cols])
            if rp.exists():
                record=C.read(rp)
                if record['identity']!=ident or C.sha(p)!=record['image_sha256']:raise ValueError('Changed cached render')
            else:
                if p.exists():raise RuntimeError('Unreceipted image: explicit recovery needed')
                fig=plt.figure(figsize=(4.1*len(cols),15.2),dpi=120,facecolor='white')
                grid=fig.add_gridspec(4,len(cols),left=.025,right=.99,bottom=.055,top=.94,wspace=.12,hspace=.25)
                visible=[]
                for col,(arm,stage) in enumerate(cols):
                    g=cells[rid,arm,stage]
                    for row,b in enumerate(views):
                        im,own=D.raster(g['tri'],q['K'],b,np.array(db['center']),db['span'],size=420)
                        ax=fig.add_subplot(grid[row,col]);ax.imshow(im);ax.set_axis_off()
                        ax.set_title(arm+' | '+stage+' | view'+str(row+1),fontsize=9)
                        visible.append(dict(label=arm,stage=stage,view=row,visible_C_pixels=int(((own>=0)&(own<q['K'])).sum()),visible_F_pixels=int((own>=q['K']).sum())))
                    for row,view in ((2,'front'),(3,'side')):
                        ax=fig.add_subplot(grid[row,col]);H.draw_section(ax,g,[],bounds[cid],C.L,view,1)
                        ax.set_title(arm+' | '+ax.get_title(),fontsize=8)
                fig.suptitle(q['parent_id'][:8]+' | fixed n'+str(q['noise_index'])+' | K='+str(q['K'])+' N='+str(q['N'])+'\nFrozen SEAM generator; original / corrected all-on / source-oracle layouts',fontsize=13,y=.985)
                fig.text(.5,.022,'Blue input C / gray free. Same scale. SOURCE_ORACLE uses source labels and is a diagnostic. Actual image review recorded separately.',ha='center',fontsize=9)
                buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120,facecolor='white');plt.close(fig);write_bytes(p,buf.getvalue())
                record=dict(status='RENDERED_NOT_REVIEWED',identity=ident,image_sha256=C.sha(p),visibility=visible);C.write(rp,record)
            pages.append(dict(request_id=rid,group=group,path=str(p),sha256=C.sha(p)))
            for arm,stage in cols:templates.append(dict(request_id=rid,parent_id=q['parent_id'],condition_id=cid,noise_index=q['noise_index'],label=arm,stage=stage,reviewer='',review_status='PENDING_ACTUAL_VIEW',**{x:'' for x in REVIEW_FIELDS},notes='',reviewed_image_path=str(p),reviewed_image_sha256=C.sha(p)))
        print(json.dumps(dict(rendered_request=rid,pages=len(pages))),flush=True)
    # Smaller contact sheets support checking every noise and layout together.
    contacts=[]
    for cond in m['conditions']:
        cid=cond['condition_id'];qq=sorted([q for q in m['requests'] if q['condition_id']==cid],key=lambda q:q['noise_index']);db=depth[cid]
        for group,cols in GROUPS.items():
            for vi,b in enumerate(views):
                p=dest/(cond['parent_id'][:8]+'_'+group+'_contact_view'+str(vi+1)+'.png')
                fig,axes=plt.subplots(4,len(cols),figsize=(3.0*len(cols),11.5),dpi=120)
                for row,q in enumerate(qq):
                    for col,(a,s) in enumerate(cols):
                        im,_=D.raster(cells[q['request_id'],a,s]['tri'],q['K'],b,np.array(db['center']),db['span'],size=320)
                        axes[row,col].imshow(im);axes[row,col].set_axis_off();axes[row,col].set_title('n'+str(q['noise_index'])+' '+a+'\n'+s,fontsize=8)
                fig.suptitle(cond['parent_id'][:8]+' | '+group+' | paired fixed noises | view'+str(vi+1),fontsize=12)
                fig.tight_layout(rect=[0,0,1,.96]);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120);plt.close(fig);write_bytes(p,buf.getvalue())
                contacts.append(dict(parent_id=cond['parent_id'],group=group,view=vi+1,path=str(p),sha256=C.sha(p)))
    C.csvwrite(dest/'review_template_NOT_REVIEWED.csv',templates)
    doc='<meta charset="utf-8"><title>Frozen occupancy layout comparison</title><h1>Frozen generator, three occupancy layouts</h1><p>Four TRAIN C, sixteen fixed requests. SOURCE_ORACLE uses source labels. Rendering is not an actual review.</p>'
    for c in contacts:doc+='<h2>'+html.escape(c['parent_id'][:8]+' '+c['group']+' view'+str(c['view']))+'</h2><a href="'+Path(c['path']).name+'"><img loading="lazy" style="width:100%" src="'+Path(c['path']).name+'"></a>'
    doc+='<h1>Full views and finite sections</h1>'
    for p in pages:doc+='<p><a href="'+Path(p['path']).name+'">'+html.escape(p['request_id']+' '+p['group'])+'</a></p>'
    write_bytes(dest/'index.html',doc.encode('utf-8'));E.E.verify(protected);E.E.verify(source_hashes)
    if 'torch' in sys.modules:raise RuntimeError('Renderer imported model framework')
    r=dict(status='RENDERED_PENDING_ACTUAL_REVIEW',pages=pages,contacts=contacts,geometry_cells=144,actual_reviews=0,model_calls=0,
        source_hashes=source_hashes,input_hashes=protected,files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
    C.write(done,r);return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(E.OUT));a=p.parse_args();r=run(a.root)
    print(json.dumps(dict(status=r['status'],pages=len(r['pages']),contacts=len(r['contacts']))),flush=True)

if __name__=='__main__':main()
