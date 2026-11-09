"""CPU-only visualization of the frozen two-triangle inputs and four RAWs.

No model calls, mesh changes, repair, welding, smoothing or score computation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LP_NUM_THREADS'):
    os.environ.setdefault(name,'2')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import raw_renderer as renderer

DEFAULT_OUT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def font(size,bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',size)


def centered(draw,text,cx,y,size=30,bold=False,color='#222831'):
    f=font(size,bold);b=draw.textbbox((0,0),text,font=f)
    draw.text((cx-(b[2]-b[0])/2,y),text,font=f,fill=color)


def layout(path,title,subtitle,columns,row_labels,rows,panel_size,footer):
    path=Path(path)
    ncols=len(columns);nrows=len(rows);side=175 if row_labels else 25;gap=24;right=25;top=150;bottom=75
    width=side+ncols*panel_size+(ncols-1)*gap+right
    height=top+nrows*panel_size+(nrows-1)*gap+bottom
    canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    draw.text((25,18),title,font=font(38,True),fill='#20262E')
    draw.text((25,70),subtitle,font=font(24),fill='#4D5660')
    boxes=[]
    for j,label in enumerate(columns):centered(draw,label,side+j*(panel_size+gap)+panel_size/2,111,26,True)
    for i,row in enumerate(rows):
        if len(row)!=ncols:raise ValueError('Layout columns differ')
        y=top+i*(panel_size+gap)
        if row_labels:
            for n,line in enumerate(row_labels[i]):draw.text((24,y+panel_size//2-55+n*33),line,font=font(24,n==0),fill='#25313D')
        for j,p in enumerate(row):
            x=side+j*(panel_size+gap)
            with Image.open(p) as panel:
                if panel.size!=(panel_size,panel_size):raise ValueError('No image rescaling/cropping in layout')
                canvas.paste(panel.convert('RGB'),(x,y))
            boxes.append(dict(row=i,column=j,image=str(p),xyxy=[x,y,x+panel_size,y+panel_size],no_crop=True,no_rescale=True))
    draw.text((25,height-52),footer,font=font(22),fill='#4D5660')
    if path.exists():raise FileExistsError('Existing layout is preserved: '+str(path))
    canvas.save(path,dpi=(180,180))
    return dict(path=str(path),sha256=sha(path),size=list(canvas.size),panel_boxes=boxes,all_panels_uncropped=True)


def run(out=DEFAULT_OUT):
    out=Path(out);started=time.perf_counter();sources={};receipts=[];new_calls=0
    visual=out/'visualization';panels=visual/'panels';panels.mkdir(parents=True,exist_ok=True)
    def remember(p):
        p=Path(p);sources[str(p)]=sha(p);return p
    plan=json.loads(remember(out/'plan.json').read_text())
    conditions=sorted(plan['conditions'],key=lambda x:x['width']);seeds=sorted({r['seed'] for r in plan['requests']})
    if len(conditions)!=2 or len(seeds)!=2 or len(plan['requests'])!=4:raise ValueError('Expected fixed 2 widths by 2 seeds')
    C={};input_obj=[]
    for item in conditions:
        name=item['condition'];p=remember(out/'inputs'/(name+'.npy'))
        if sources[str(p)]!=plan['input_file_hashes'][str(p.relative_to(out))]:raise ValueError('Input C file identity changed')
        C[name]=np.load(p,allow_pickle=False).reshape(-1,3,3)
        if C[name].dtype!=np.float32 or C[name].shape!=(2,3,3):raise ValueError('Exactly two original FP32 triangles required')
        obj=remember(out/'inputs'/(name+'.obj'));lines=obj.read_text().splitlines()
        vertices=[x for x in lines if x.startswith('v ')];faces=[x for x in lines if x.startswith('f ')]
        if len(vertices)!=4 or len(faces)!=2:raise ValueError('Input OBJ must contain actual four vertices and two faces')
        edge_counts={}
        for triangle in C[name]:
            for a,b in ((0,1),(1,2),(2,0)):
                edge=tuple(sorted((tuple(triangle[a]),tuple(triangle[b]))));edge_counts[edge]=edge_counts.get(edge,0)+1
        if sum(v==2 for v in edge_counts.values())!=1 or sum(v==1 for v in edge_counts.values())!=4:raise ValueError('Input topology changed')
        input_obj.append(dict(condition=name,obj=str(obj),sha256=sources[str(obj)],vertices=4,faces=2,shared_diagonals=1,boundary_edges=4))
    renderer_source=remember(Path(renderer.__file__));script_source=remember(Path(__file__))
    def panel(name,c,f,camera,size,mode):
        nonlocal new_calls
        p=panels/(name+'.png');j=p.with_suffix('.json')
        if j.exists():
            receipt=json.loads(j.read_text());r=receipt['render']
            if sha(p)!=r['png_sha256'] or r['camera']!=camera or receipt['renderer_sha256']!=sources[str(renderer_source)]:raise ValueError('Cached panel identity mismatch')
            if r['input_geometry_sha256']['C']!=renderer._hash(c) or r['input_geometry_sha256']['F']!=renderer._hash(f):raise ValueError('Cached geometry mismatch')
        else:
            r=renderer.render_panel(c,f,p,camera,size=(size,size),mode=mode,context_edges=True)
            new_calls+=1
            receipt=dict(render=r,renderer_sha256=sources[str(renderer_source)],input_triangle_edges='Display only: thin dark gray C polygon edges, including the one exact shared diagonal')
            write_json(j,receipt)
        receipts.append(dict(name=name,path=str(j),sha256=sha(j),render=r));return p
    empty=np.empty((0,3,3),np.float32)
    clean_camera=renderer.frame_for([C[x['condition']] for x in conditions],azimuth=45,elevation=35,aspect=1)
    clean=[panel('input_clean_'+x['condition'],C[x['condition']],empty,clean_camera,700,'context') for x in conditions]
    inputs_path=out/'inputs_only.png'
    if not inputs_path.exists():
        input_layout=layout(inputs_path,'Clean two-triangle seat inputs','One connected planar patch; no GT mesh or extra context triangles.',
            ['Width 2.4  |  depth 2.4','Width 3.0  |  depth 2.4'],None,[clean],700,
            'Blue: fixed C. Thin gray edges expose the single shared diagonal. Each OBJ has 4 vertices and 2 faces.')
        write_json(visual/'inputs_only_layout.json',input_layout)
    else:input_layout=json.loads((visual/'inputs_only_layout.json').read_text())
    run_records={};pending=[]
    for request in plan['requests']:
        p=out/request['output_dir']/'run.json'
        if not p.exists():pending.append(request['id']);continue
        r=json.loads(remember(p).read_text())
        if r['status']!='PASS':pending.append(request['id']);continue
        run_records[request['id']]=r
    if pending:
        renderer.close_renderer()
        result=dict(status='INPUTS_COMPLETE_OUTPUTS_PENDING',pending_requests=pending,inputs_only=input_layout,new_render_calls=new_calls,model_calls=0)
        write_json(visual/'pending.json',result);return result
    outputs={}
    for request in plan['requests']:
        record=run_records[request['id']];p=remember(out/request['output_dir']/'raw.npz')
        if sources[str(p)]!=record['raw_sha256']:raise ValueError('RAW hash differs from PASS run record')
        with np.load(p,allow_pickle=False) as z:
            raw=z['output'].copy().reshape(-1,3,3);rawC=z['C'].copy().reshape(-1,3,3)
        context=C[request['condition']]
        if raw.dtype!=np.float32 or raw.shape!=(160,3,3) or raw[:2].tobytes()!=context.tobytes() or rawC.tobytes()!=context.tobytes():raise ValueError('RAW/C identity or shape changed')
        outputs[(request['condition'],request['seed'])]=raw
    all_geometry=[C[x['condition']] for x in conditions]+list(outputs.values())
    global_camera=renderer.frame_for(all_geometry,azimuth=45,elevation=22,aspect=1)
    board=[]
    for item in conditions:
        name=item['condition'];row=[panel('main_'+name+'_input',C[name],empty,global_camera,900,'context')]
        for seed in seeds:row.append(panel('main_'+name+'_'+str(seed),C[name],outputs[(name,seed)][2:],global_camera,900,'output'))
        board.append(row)
    comparison_path=out/'control_comparison.png'
    if not comparison_path.exists():
        comparison=layout(comparison_path,'Two-triangle control | HYBRID START RAW','The same global camera and scale are used for all six panels; original coordinates and all vertices retained.',
            ['Input C','Seed '+str(seeds[0])+' | RAW','Seed '+str(seeds[1])+' | RAW'],
            [['width 2.4','depth 2.4','K=2, N=160'],['width 3.0','depth 2.4','K=2, N=160']],board,900,
            'Blue: exact input C. Orange: generated free triangles. Shared Gaussian by seed. No smoothing, cropping or repair.')
        write_json(visual/'control_comparison_layout.json',comparison)
    else:comparison=json.loads((visual/'control_comparison_layout.json').read_text())
    side_camera=renderer.frame_for(all_geometry,azimuth=90,elevation=5,aspect=1)
    side=[]
    for item in conditions:
        name=item['condition'];side.append([panel('side_'+name+'_'+str(seed),C[name],outputs[(name,seed)][2:],side_camera,800,'output') for seed in seeds])
    side_path=out/'side_view.png'
    if not side_path.exists():
        side_layout=layout(side_path,'Side view | existing RAW geometry','Orthographic azimuth 90 degrees, elevation 5 degrees; one shared camera and scale for these four panels.',
            ['Seed '+str(seeds[0])+' | RAW','Seed '+str(seeds[1])+' | RAW'],
            [['width 2.4','K=2, N=160'],['width 3.0','K=2, N=160']],side,800,
            'The slight elevation keeps the blue seat plane visible. This view illustrates geometry; it does not certify attachment.')
        write_json(visual/'side_view_layout.json',side_layout)
    else:side_layout=json.loads((visual/'side_view_layout.json').read_text())
    renderer.close_renderer()
    if not all(sha(p)==expected for p,expected in sources.items()):raise AssertionError('Visualization changed an original input/result/source')
    main=[x for x in receipts if x['name'].startswith('main_')]
    if len(main)!=6 or any(x['render']['camera']!=global_camera for x in main):raise AssertionError('Main board does not share one camera')
    if not all(x['render']['all_vertices_inside'] and x['render']['input_arrays_unchanged'] for x in receipts):raise AssertionError('Rendering geometry/crop invariant failed')
    manifest=dict(status='PASS',inputs_only=input_layout,control_comparison=comparison,side_view=side_layout,
        source_files=sources,source_files_unchanged=True,input_OBJ_topology=input_obj,
        cameras=dict(clean_input_preview=clean_camera,global_main_comparison=global_camera,global_side_comparison=side_camera),
        camera_policy='All four outputs and two contexts determine the single mainboard frame. Inputs-only deliberately uses a closer common frame for triangle inspection. Side-view has a separate single shared frame.',
        panel_receipts=receipts,panel_count=len(receipts),new_render_calls=new_calls,
        render_seconds=sum(x['render']['seconds']['total'] for x in receipts),total_visualization_seconds=time.perf_counter()-started,
        model_calls=0,training_updates=0,geometry_processing_calls=0,mesh_repairs=0,geometry_indices_changed=False,
        source_script_sha256=sha(script_source),renderer_source_sha256=sha(renderer_source),
        renderer_backend=receipts[0]['render']['gl_capabilities'],threads=dict(OMP=2,MKL=2,OPENBLAS=2,LP=2),
        limitation='Exact C preservation and viewing contact are not topology welding, intersection certification or proof of control success.')
    write_json(visual/'visualization_manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['status','panel_count','new_render_calls','render_seconds','total_visualization_seconds','model_calls']},indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    run(parser.parse_args().out)
