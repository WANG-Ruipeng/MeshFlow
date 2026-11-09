"""Render finalized predeclared conversion/regression/cost selection, zero scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import json,time,sys,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from PIL import Image,ImageDraw,ImageFont
from r2.tools.partial_edge_binding_v1 import assets as A
from r2.tools.partial_edge_binding_v1.render import render_case,write_gallery

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,indent=2,sort_keys=True)+'\n'
    if path.exists() and path.read_text(encoding='utf8') != data:
        raise ValueError('Refusing changed selection/receipt overwrite: '+str(path))
    if not path.exists():path.write_text(data,encoding='utf8')

def render_one(item):
    row,record=item
    dest=A.OUT/'gallery'/record['label']/(record['short_id']+'.jpg')
    return render_case(row,record['post_path'],dest,label=record['label'])

def main():
    start=time.perf_counter(); out=A.OUT
    completed=A.read(out/'execution_complete.json')
    if completed['status']!='COMPLETE' or completed['cells']!=144:
        raise ValueError('Wait for the complete 144-cell barrier')
    rows=[A.read(p)['row'] for p in sorted((out/'cells').glob('*/*/complete.json'))]
    if len(rows)!=144:raise ValueError('Incomplete cells')
    inputs={r['request_id']:r for r in A.read(out/'input_manifest.json')['rows']}
    selected={}
    def add(r,reason):
        key=(r['label'],r['short_id'])
        selected.setdefault(key,dict(row=r,reasons=[]))['reasons'].append(reason)
    for r in rows:
        if r['screening_joint'] and not r['baseline_joint']:add(r,'NEW_SCREENING_JOINT')
        if r['baseline_joint'] and not r['screening_joint']:add(r,'LOST_BASELINE_JOINT')
        # Deliberately include every geometrically modified original positive;
        # no subjective threshold or figure-based exclusion.
        if r['baseline_joint'] and r['accepted']>0:add(r,'ORIGINAL_JOINT_WITH_ANY_ACCEPTED_BINDING')
    for label in sorted({r['label'] for r in rows}):
        arm=[r for r in rows if r['label']==label]
        for key in ('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C'):
            worst=sorted(arm,key=lambda r:(-r['delta_pp_'+key],r['request_id']))[0]
            add(worst,'MAX_DELTA_'+key)
    ordered=[selected[k] for k in sorted(selected)]
    selection=dict(status='FROZEN',candidate_cells=144,selected_cells=len(ordered),
        policy='All new/lost screening-joint; every original joint with >=1 accepted binding; one maximum delta CF/FF/I per arm, ties request_id; deduplicate cell keys',
        selection=[dict(label=x['row']['label'],request_id=x['row']['request_id'],short_id=x['row']['short_id'],post_path=x['row']['post_path'],reasons=x['reasons']) for x in ordered],
        automatic_visual_labels=0)
    save(out/'gallery/selection.json',selection)
    receipts=[]
    jobs=[(inputs[x['row']['request_id']],x['row']) for x in ordered]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for i,(x,receipt) in enumerate(zip(ordered,pool.map(render_one,jobs))):
            r=x['row'];receipt['selection_reasons']=x['reasons'];receipts.append(receipt)
            print(json.dumps(dict(rendered=i+1,total=len(ordered),case=r['label']+'/'+r['short_id'])),flush=True)
    manifest=write_gallery(receipts,out/'gallery')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    sheets=[]
    for offset in range(0,len(receipts),6):
        group=receipts[offset:offset+6]; tile_w,tile_h=460,795
        page=Image.new('RGB',(tile_w*3,tile_h*2),'white');draw=ImageDraw.Draw(page)
        for j,rec in enumerate(group):
            im=Image.open(rec['path']).convert('RGB');im.thumbnail((tile_w,tile_h-30),Image.Resampling.LANCZOS)
            x=(j%3)*tile_w;y=(j//3)*tile_h
            page.paste(im,(x,y+25));draw.text((x+4,y+3),rec['label']+' / '+rec['short_id'],font=font,fill='black')
        path=out/'gallery/contact_sheets'/('sheet_'+str(offset//6).zfill(2)+'.jpg');path.parent.mkdir(parents=True,exist_ok=True)
        page.save(path,quality=94,subsampling=0)
        sheets.append(dict(path=str(path),sha256=A.sha(path),cases=[r['label']+'/'+r['short_id'] for r in group]))
    receipt=dict(status='COMPLETE',selected_cases=len(receipts),all_cell_count=144,seconds=time.perf_counter()-start,
        images=receipts,contact_sheets=sheets,selection_path=str(out/'gallery/selection.json'),
        renderer_sha256=A.sha(Path(__file__)),model_calls=0,GPU_calls=0,scoring_calls=0,automatic_visual_labels=0,
        review_status='NOT_REVIEWED_BY_THIS_RENDER_STAGE')
    save(out/'render_completion.json',receipt)
    print(json.dumps(dict(status='COMPLETE',images=len(receipts),contact_sheets=len(sheets),seconds=receipt['seconds'])),flush=True)

if __name__=='__main__':main()
