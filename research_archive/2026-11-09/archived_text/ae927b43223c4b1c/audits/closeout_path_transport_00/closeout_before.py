"""CPU-only closeout of saved results and actually recorded image judgments."""
from __future__ import annotations
import argparse
import csv
import difflib
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO/'experiments/r2_bench_control_transfer_v1'
FIELDS = ['recognizable_complete_bench', 'C_plausibly_used',
          'visible_detached_fragments', 'visible_holes_or_tears', 'visible_fold_or_overlap']

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def write(p, v):
    p=Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def csvwrite(p, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys);w.writeheader();w.writerows(rows)

def main(root=ROOT):
    root=Path(root);review=root/'visual_review';delivery=root/'delivery'
    delivery.mkdir(exist_ok=True)
    pages=read(root/'gallery_manifests/paired.json')['pages']
    expected={(p['request_id'],a,s) for p in pages for a in ('PLAIN','SEAM') for s in ('RAW','POST')}
    rows=[];parts=[]
    for name in ('part_a.json','part_b.json'):
        path=review/name;part=read(path);parts.append(dict(path=name,sha256=sha(path)))
        assert part['generation_or_metric_calls']==0
        rows.extend(part['rows'])
    actual={(r['request_id'],r['arm'],r['stage']) for r in rows}
    assert len(rows)==len(actual)==96 and actual==expected
    for r in rows:
        assert all(r[k] in ('YES','NO','UNCERTAIN') for k in FIELDS)
        assert r['reviewer'].startswith('AI_PROXY') and r['notes'].strip()
        assert sha(root/r['image_path'])==r['image_sha256']
    order={p['request_id']:i for i,p in enumerate(pages)}
    rows.sort(key=lambda r:(order[r['request_id']],r['arm'],r['stage']))
    write(review/'per_output.json',dict(protocol_sha256=sha(root/'visual_review_protocol.json'),
        rows=rows,parts=parts,actual_images_viewed=24,observations=96,
        nonblind=True,two_reviewers_partition_pages_not_double_rating=True,model_calls=0))
    csvwrite(review/'per_output.csv',[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v)
                                     for k,v in r.items()} for r in rows])
    summary=[]
    for group in ('TRAIN','DEV','ALL'):
        for arm in ('PLAIN','SEAM'):
            for stage in ('RAW','POST'):
                sub=[r for r in rows if (group=='ALL' or r['split']==group) and r['arm']==arm and r['stage']==stage]
                result=dict(group=group,arm=arm,stage=stage,n=len(sub))
                for field in FIELDS:
                    counts=Counter(r[field] for r in sub)
                    for verdict in ('YES','UNCERTAIN','NO'):result[field+'__'+verdict]=counts[verdict]
                summary.append(result)
    csvwrite(review/'summary.csv',summary)
    def table(group):
        lines=['| 模型/阶段 | n | 完整长凳 | C合理参与 | 可见悬片 | 可见裂口 | 可见折层/重叠 |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for r in summary:
            if r['group']!=group:continue
            verdicts=['/'.join(str(r[f+'__'+v]) for v in ('YES','UNCERTAIN','NO')) for f in FIELDS]
            lines.append('| '+r['arm']+'/'+r['stage']+' | '+str(r['n'])+' | '+' | '.join(verdicts)+' |')
        return '\n'.join(lines)
    text='''# 实际非盲 AI_PROXY 图审

24张配对PNG全部实际查看，每张包括两侧表面及C位置front/side有限截面，共96个模型/阶段单元。两名AI评阅者各承担12页，不是每页双盲复评，也不是人类评价。评阅时未查阅数值评分，五项判断来自实际图像，原始中文notes完整保存。

下表每格均为 **YES / UNCERTAIN / NO**。完整长凳与C合理参与的YES是视觉肯定；悬片、裂口和折层的YES是可见问题。可见悬片字段可能包含悬空的已给C，不能当作纯free碎片率；每条notes明确区分。截面孤立线段未自动判为浮片。合理底板/侧板不要求改当座面。

## DEV，8父×2噪声

'''+table('DEV')+'\n\n## 全组，12父×2噪声\n\n'+table('ALL')+'''

可见收益与代价并存：`f495bd67/n1` SEAM主体更完整，`9d56087c/n1` SEAM保留较完整支撑，`5dda4ef4/n0`两模型C均可自然参与；`e7bb6ef5`两噪声的POST支撑明显消失，`9fd5b67a/n0`POST腿部严重退化。上述例子只解释全组记录，不替代全部样本，也不认证索引接续。

ROOT另实际复看c3d8c664/n0、9fd5b67a/n0、e7bb6ef5/n0三页；这些为重复spot check，不追加到96行分母。逐行证据：per_output.csv/json及保留原part_a/part_b.json。
'''
    (review/'REVIEW_SUMMARY.md').write_text(text,encoding='utf-8')
    integrity=read(root/'final_integrity.json')
    assert integrity['status']=='PASS'
    audit=read(root/'training_audit/FINAL_TRAINING_AUDIT.json')
    assert audit['status']=='PASS' and audit['paired_stream_exact']
    assert all(audit['arms'][a]['physical_updates']==3000 for a in ('PLAIN','SEAM'))
    assert read(root/'evaluation/score_completion.json')['cells']==96
    completions={}
    for arm in ('PLAIN','SEAM'):
        completions[arm]={s:read(root/'evaluation'/arm/(s+'_complete.json')) for s in ('raw','post')}
        assert all(v['outputs']==24 for v in completions[arm].values())
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    source_paths=set(Path(__file__).parent.glob('*.py'))|set(Path(__file__).parent.glob('*.ps1'))|set(Path(__file__).parent.glob('*.md'))
    for p in read(root/'registration.json')['code_hashes']:
        source_paths.add(Path(p.replace('E:\\MeshFlow','/mnt/e/MeshFlow').replace('\\','/')))
    source_rows=[];diff=[]
    for p in sorted(source_paths):
        relative=p.relative_to(REPO);dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==sha(p)
        else:shutil.copyfile(p,dest)
        source_rows.append(dict(path=str(relative),sha256=sha(p),snapshot=str(dest.relative_to(root))))
        if p.parent==Path(__file__).parent:
            diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(True),fromfile='/dev/null',tofile=str(relative)))
    write(delivery/'source_hashes.json',dict(scope='New pilot implementation and registered reusable dependencies; not a full environment image',files=source_rows))
    (delivery/'code.diff').write_text(''.join(diff),encoding='utf-8')
    write(delivery/'code_diff_scope.json',dict(scope='New independent pilot files represented against /dev/null; not a git-index diff. Existing scorer/maintained code unchanged per final_integrity.',files=len(source_rows)))
    report=root/'REPORT.md';report_text=report.read_text(encoding='utf-8')
    marker='\n## 10. 实际DEV图审计数'
    report_text=report_text.split(marker)[0]
    report.write_text(report_text+marker+'\n\n每格为YES/UNCERTAIN/NO；前两列YES是视觉肯定，后三列YES是可见问题。非盲AI_PROXY，不是人评或拓扑认证；悬片可能含悬空C，详见逐行notes。\n\n'+table('DEV')+'\n',encoding='utf-8')
    ended=datetime.now(timezone.utc);started=datetime(2026,10,18,9,12,32,tzinfo=timezone.utc)
    resources={}
    for name,args in {'gpu':['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv'],
                      'gpu_processes':['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']}.items():
        run=subprocess.run(args,capture_output=True,text=True);resources[name]=dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr)
    try:
        import psutil
        resources['cpu']=dict(logical=psutil.cpu_count(),available_RAM_bytes=psutil.virtual_memory().available,total_RAM_bytes=psutil.virtual_memory().total)
    except ImportError:resources['cpu']='NA; see original resource records'
    write(delivery/'resources_final.json',resources)
    receipt=dict(status='COMPLETE_STOPPED',first_execution_utc=started.isoformat(),closed_utc=ended.isoformat(),
        elapsed_seconds=(ended-started).total_seconds(),within_four_hour_window=(ended-started).total_seconds()<14400,
        category='Bench',train_parents=64,dev_parents=8,conditional_updates_per_arm=3000,
        generator_training_forwards=48000,generator_training_backwards=48000,generator_updates=6000,
        occupancy_training_forwards=24000,occupancy_training_backwards=24000,occupancy_updates=3000,
        generated_raw=48,post_outputs=48,Euler50_forwards=2400,occupancy_inference_forwards=12,
        D0_forwards=48,GEO_calls=48,score_cells=96,numeric_fields=124,gallery_pages=24,actual_AI_PROXY_observations=96,
        GPU_failures=0,scientific_replays=0,automatic_extensions=0,changed_default=False,commit=False,push=False,
        old_negative_cleanup_files=13,old_negative_cleanup_logical_bytes=17602302735,
        new_rolling_cleanup_files=2,new_rolling_cleanup_logical_bytes=3130105078,
        disk_note='Logical bytes deleted, not net disk gain; newly retained official D0 and training checkpoints consume space.',
        conclusion='LOCAL_C_F_TRANSFER_WITHOUT_BULK_ATTACHMENT_AND_WITH_QUALITY_COSTS',
        final_integrity_sha256=sha(root/'final_integrity.json'),training_audit_sha256=sha(root/'training_audit/FINAL_TRAINING_AUDIT.json'),
        model_calls_during_closeout=0)
    write(root/'FINAL_RECEIPT.json',receipt)
    # Small review package: all paired pages/CSV/code, no mesh arrays, trajectories or weights.
    included=set(root.glob('*.md'))|{root/'FINAL_RECEIPT.json',root/'final_integrity.json',root/'registration.json',root/'visual_review_protocol.json'}
    for directory in ('visual_review','numerical_review','source_snapshot','scoring_source_snapshot','gallery_manifests','delivery'):
        included.update(p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.md','.json','.csv','.py','.ps1','.diff'))
    included.update((root/'gallery/paired').glob('*.png'));included.add(root/'gallery/paired/index.html')
    included.update(p for p in (root/'evaluation').glob('*.csv'))
    included.update((root/'evaluation/paired_complete').glob('*.csv'))
    for filename in ('DATASET_CARD.md','manifest.json','input_visual_review.csv','summary.json'):
        p=root/'data'/filename
        if p.exists():included.add(p)
    for p in (root/'learning_curves').glob('*'):
        if p.suffix in ('.csv','.png','.json'):included.add(p)
    for filename in ('FINAL_TRAINING_AUDIT.json','CURVE_REVIEW.md','exposure_per_parent.csv'):
        p=root/'training_audit'/filename
        if p.exists():included.add(p)
    for p in (root/'cleanup').rglob('*receipt.json'):included.add(p)
    # Exclude a prior packaging receipt if closeout is explicitly repeated on same evidence.
    included.discard(delivery/'review_package.json')
    entries=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(included)]
    package=root/'BENCH_CONTROL_TRANSFER_REVIEW.zip'
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(included):z.write(p,str(p.relative_to(root)))
        z.writestr('PACKAGE_CONTENTS.json',json.dumps(dict(files=entries,excludes=['weights','RAW/POST NPZ/PLY','Euler trajectories','full training exposure stream']),ensure_ascii=False,indent=2))
    with zipfile.ZipFile(package) as z:assert z.testzip() is None
    write(delivery/'review_package.json',dict(path=package.name,bytes=package.stat().st_size,sha256=sha(package),file_count=len(entries)+1,files=entries,
        exclusions='Large weights, RAW/POST meshes/trajectories and full training logs remain local; this is a review package, not full replay bundle.'))
    print(json.dumps(dict(status=receipt['status'],elapsed_minutes=receipt['elapsed_seconds']/60,review_rows=96,zip_bytes=package.stat().st_size,zip_sha256=sha(package)),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(ROOT));a=parser.parse_args();main(a.root)
