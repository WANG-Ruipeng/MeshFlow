"""CPU-only compact evidence bundle. No scoring, models or mesh modification."""
import csv,difflib,html,json,shutil,zipfile
from pathlib import Path
from PIL import Image
import numpy as np
from .common import *

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=rows(OUT/'analysis/per_parent.csv')
    uids=sorted({r['uid'] for r in data})
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    metrics=[('SeamCov','Seam coverage delta (pp): higher better',100),
             ('GT_to_free_MSE2048','GT to free MSE change (%): lower better',None),
             ('ConflictCov_C','Conflict coverage delta (pp): lower better',100)]
    # Resolve the precise frozen original spelling; never substitute another distance.
    available={r['metric'] for r in data}
    gt=[m for m in available if '2048' in m and ('GT_to' in m or 'GT2' in m) and 'MSE' in m]
    if metrics[1][0] not in available:
        assert len(gt)==1,gt
        metrics[1]=(gt[0],metrics[1][1],None)
    for i,panel in enumerate(('OLD','FRESH')):
        for j,(metric,title,scale) in enumerate(metrics):
            ax=axes[i,j]
            for label,offset,color,short in [('HIST_CAPPED500',-.16,'#cb8754','Historical CAPPED500'),('HALF_CAPPED500',.16,'#2d819d','Half CAPPED500')]:
                selected={r['uid']:r for r in data if r['panel']==panel and r['candidate_label']==label and r['reference_label']=='START' and r['metric']==metric}
                values=[]
                for uid in uids:
                    r=selected[uid];v=float(r['delta'])
                    values.append(v*scale if scale else 100*v/float(r['reference_value']))
                ax.bar(np.arange(8)+offset,values,width=.31,color=color,label=short)
            ax.axhline(0,color='#333',lw=.7);ax.set_xticks(range(8),[u[:4] for u in uids]);ax.set_title(panel+' | '+title,fontsize=10)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Paired parent means relative to START | same 8 observed DEV parents, 4 fixed noises\nNo training; alpha=0.5 only; parent percent changes are not the aggregate percent change',fontsize=12)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'parent_tradeoffs.png',dpi=150);fig.savefig(folder/'parent_tradeoffs.pdf');plt.close(fig)
    return dict(status='PASS',source_sha256=digest(OUT/'analysis/per_parent.csv'),model_calls=0)

def package():
    assert read_json(OUT/'audit/runtime_final.json')['status']=='PASS'
    assert read_json(OUT/'analysis/results.json')['completed_cells']==96
    report=OUT/'CAPPED500_HALF_REPORT.md';assert report.exists()
    reviews=[OUT/'visuals/reviews'/name for name in ('root.json','coupling.json','diagnostics.json')]
    review_rows=[]
    for p in reviews:
        v=read_json(p);assert v['status']=='PASS' and v['reviewed_outputs']==len(v['rows'])
        review_rows.extend(v['rows'])
    expected={(label,q['request_id']) for label in LABELS for panel in ('OLD','FRESH') for q in read_json(OUT/'manifests'/f'{panel}.json')['rows']}
    assert len(review_rows)==96 and {(r['label'],r['request_id']) for r in review_rows}==expected
    review_receipt=dict(status='PASS',AI_reviewed_outputs=96,human_usability='NOT_RUN',selection='All registered cells, none omitted',
        files={str(p):digest(p) for p in reviews},quality_pass_inferred=False,model_calls=0)
    atomic_json(OUT/'audit/visual_review_complete.json',review_receipt)
    dest=OUT/'delivery/review';dest.mkdir(parents=True,exist_ok=True)
    inventory={}
    from . import evaluation as ev
    all_rows=[]
    for label in LABELS:
        for panel in ('OLD','FRESH'):
            all_rows.extend(read_json(OUT/'evaluation'/label/panel/'per_output.json'))
    assert len(all_rows)==96 and {(r['label'],r['request_id']) for r in all_rows}==expected
    ev.write_csv(OUT/'analysis/all96_per_output.csv',all_rows)
    ev.write_csv(OUT/'analysis/all96_AI_visual_review.csv',review_rows)
    def copy(p,relative=None):
        p=Path(p);rel=Path(relative) if relative else p.relative_to(OUT)
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target);inventory[str(rel)]=dict(sha256=digest(target),bytes=target.stat().st_size,source=str(p))
    for p in (report,OUT/'protocol.json',OUT/'accounting.json'):copy(p)
    for folder in ('analysis','registration','manifests','audit','resources','ledger','status','cpu_tests','engineering','figures'):
        for p in sorted((OUT/folder).rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.md','.diff','.py','.png','.pdf'):copy(p)
    for p in sorted((OUT/'visuals/reviews').glob('*.json')):copy(p)
    for label in LABELS:
        for panel in ('OLD','FRESH','ALL'):
            base=OUT/'evaluation'/label/panel
            for p in base.glob('*'):
                if p.is_file() and p.suffix in ('.csv','.json','.jsonl'):copy(p)
    for p in (WEIGHT.with_suffix('.json'),):copy(p)
    protocol=read_json(OUT/'protocol.json');source_paths={Path(p) for p in protocol['protected_files'] if Path(p).suffix=='.py'}
    for group in ('control','legacy'):
        for rel,sha in protocol['original_scorer_identity'][group].items():
            p=ROOT/rel;assert digest(p)==sha;source_paths.add(p)
    for rel in ('experiments/mf_h1_local0/chair_capped500_gradbudget_v1/evaluation.py',
                'experiments/mf_h1_local0/chair_capped500_gradbudget_v1/analysis.py',
                'experiments/mf_h1_local0/chair_control_feedback_auto_v1/analysis.py'):
        p=ROOT/rel
        if p.exists():source_paths.add(p)
    from . import evaluation as ev
    source_paths.update(Path(m.__file__) for m in (ev.gates,ev.prior,ev.old,ev.control,ev.quality))
    source_paths.update(Path(__file__).parent.glob('*.py'))
    hashes=[];diff=[]
    for p in sorted(source_paths):
        rel=p.relative_to(ROOT);copy(p,Path('actual_code')/rel)
        hashes.append(dict(path=str(rel),sha256=digest(p),bytes=p.stat().st_size,new=p.parent==Path(__file__).parent))
        if p.parent==Path(__file__).parent:
            diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+str(rel)))
    atomic_json(OUT/'source_files.json',hashes);copy(OUT/'source_files.json')
    (OUT/'code.diff').write_text(''.join(diff),encoding='utf-8');copy(OUT/'code.diff')
    conversions=[];gallery=['<!doctype html><meta charset="utf-8"><title>Half CAPPED500 full gallery</title>',
        '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%}a{color:#167}</style>',
        '<h1>All 96 meshes: C / START / historical CAPPED500 / half CAPPED500 / GT</h1>',
        '<p>Two noises per parent page; four views plus wireframe and saved S/I/E overlays. Blue C, orange free. AI visual review only; human usability not measured. Lossless WebP decoded RGB verified equal to original PNG.</p>']
    for panel in ('OLD','FRESH'):
        latest=read_json(OUT/'visuals'/('latest_'+panel+'.json'));m=read_json(latest['manifest'])
        assert m['status']=='COMPLETE' and m['all_geometry_unchanged'] and m['all_vertices_inside']
        copy(Path(latest['manifest']));copy(Path(m['camera_manifest']))
        for page in m['parent_pages']:
            p=Path(page['path']);assert digest(p)==page['sha256']
            im=Image.open(p).convert('RGB');target=dest/'gallery'/panel/(p.stem+'.webp');target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():im.save(target,format='WEBP',lossless=True,method=6)
            assert np.array_equal(np.asarray(im),np.asarray(Image.open(target).convert('RGB')))
            rel=target.relative_to(dest);inventory[str(rel)]=dict(sha256=digest(target),bytes=target.stat().st_size,source=str(p))
            conversions.append(dict(source=str(p),source_sha256=digest(p),path=str(rel),sha256=digest(target),decoded_RGB_bitwise_equal=True))
            link=html.escape(str(rel).replace('\\','/'));gallery.append(f'<h2>{panel} {html.escape(p.stem)}</h2><a href="{link}"><img loading="lazy" src="{link}"></a>')
    atomic_json(dest/'gallery_pixel_identity.json',conversions)
    (dest/'OPEN_GALLERY.html').write_text('\n'.join(gallery),encoding='utf-8')
    (dest/'README.md').write_text('Read CAPPED500_HALF_REPORT.md first; OPEN_GALLERY.html contains every parent/noise/model.\n\nAll 96 per-output metrics and 3 pairwise contrasts are included, with original scoring source, code.diff and hashes. Weight/RAW are retained locally and intentionally omitted from this compact review bundle.\n\nRecover locally: python -B -m experiments.mf_h1_local0.chair_capped500_half_v1.run --stage all . Completed sampling is verified and skipped; no new noise, alpha, training or retry is authorized. Rendering/evaluation use identity-checked caches. AI visual records are in visuals/reviews; no user study.\n',encoding='utf-8')
    for p in dest.rglob('*'):
        if p.is_file() and p.name!='package_manifest.json':inventory[str(p.relative_to(dest))]=dict(sha256=digest(p),bytes=p.stat().st_size)
    atomic_json(dest/'package_manifest.json',inventory)
    archive=OUT/'delivery/CAPPED500_HALF_REVIEW.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(dest))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,info in inventory.items():
            import hashlib
            assert hashlib.sha256(z.read(name)).hexdigest()==info['sha256']
    assert archive.stat().st_size<10000000,archive.stat().st_size
    receipt=dict(status='PASS',archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),files=len(inventory)+1,
        lossless_gallery_pages=len(conversions),all_zip_members_hash_verified=True,raw_and_weight_omitted=True,model_calls=0)
    atomic_json(OUT/'delivery/receipt.json',receipt);return receipt

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--figures',action='store_true');a=p.parse_args()
    print(json.dumps(figures() if a.figures else package()))
