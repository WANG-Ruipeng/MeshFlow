"""CPU bookkeeping only: verify completed receipts, archive sources and review evidence."""
from __future__ import annotations
import ast
import collections
import csv
import difflib
import html
import json
from pathlib import Path
import zipfile
from . import common as C


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refuse overwrite: '+str(path))
    else:
        path.write_bytes(data)


def main():
    out=C.OUT
    reg=C.read(out/'registration.json')
    C.verify(reg['protected']); C.verify(reg['inputs'])
    jobs=C.read(out/'inputs/jobs.json'); cells=C.read(out/'inputs/cells.json')
    assert len(jobs)==11 and len(cells)==16
    calls=collections.Counter(); intervals=[]; sampling_seconds=0
    for j in jobs:
        d=out/'outputs'/j['cell_id']; r=C.read(d/'complete.json')
        assert r['status']=='COMPLETE' and r['generator_forwards']==50
        assert r['training_updates']==r['D0_forwards']==r['new_noise']==0
        assert r['audit']['anchored_all51_exact']
        C.verify(r['files'])
        log=[json.loads(s) for s in (d/'calls.jsonl').read_text().splitlines()]
        counts=collections.Counter(x['event'] for x in log)
        assert counts=={'FORWARD_STARTED':50,'FORWARD_COMPLETE':50}, counts
        calls.update(counts); intervals.append((min(x['unix'] for x in log),max(x['unix'] for x in log)))
        sampling_seconds+=r['audit']['seconds']
    assert calls['FORWARD_COMPLETE']==reg['generator_forward_budget']==550
    active=peak=0
    for _,delta in sorted([(a,1) for a,b in intervals]+[(b,-1) for a,b in intervals]):
        active+=delta;peak=max(peak,active)
    code={}
    for p in sorted((out/'workers').glob('*/code_hashes.json')):
        book=C.read(p);C.verify(book);code.update(book)
    scored=[]
    for cell in cells:
        if cell['label']=='ORIGINAL':continue
        p=out/'scores'/cell['cell_id']/'audit_receipt.json';r=C.read(p)
        C.verify(r['files']); assert r['model_calls']==r['D0_calls']==r['new_queries']==0
        scored.append(r)
    assert len(scored)==10
    assert sum(x['full_quality_scans'] for x in scored)==10
    assert sum(x['extra_common_CF_scans'] for x in scored)==8
    assert sum(x['extra_common_FF_scans'] for x in scored)==0
    met=C.read(out/'analysis/metrics_v2/metrics_complete.json')
    assert met['collected']==16 and met['fields']==177 and not met['missing']
    C.verify(met['protected']);C.verify(met['files'])
    rev1=C.read(out/'visual_review/root_f495_bcb.json')['rows']
    rev2=C.read(out/'visual_review/independent_459.json')['records']
    assert len(rev1)==len(rev2)==8
    seen=set()
    for r in rev1+rev2:
        assert r['actually_viewed'] and not r['blind'] and r['stage']=='RAW'
        p=r.get('source_image',r.get('source_path'));h=r.get('source_image_sha256',r.get('source_sha256'))
        assert C.sha(p)==h
        seen.add(r['label']+'__'+r['request_id'])
    assert seen=={c['cell_id'] for c in cells}
    pages=sorted((out/'gallery').glob('*/comparison.png'));assert len(pages)==6
    body=['<!doctype html><meta charset="utf-8"><title>冻结 EXPOSURE：C 三角划分</title>',
          '<style>body{font:16px sans-serif;max-width:1500px;margin:28px auto;padding:0 20px}img{width:100%}section{margin:36px 0}a{color:#1667a8}</style>',
          '<h1>同一物理 C，不同三角划分：RAW 固定对照</h1>',
          '<p>6 请求、16 格，零训练。蓝色为 C。使用原固定视角和位置截面；全部已有实际非盲 AI_PROXY 记录。</p>',
          '<p>459 的两个非自然 count 组合仅用于诊断。其他父保持原 count 档位。款式变化本身不判失败。</p>',
          '<p><a href="../REPORT.md">完整报告</a> · <a href="../analysis/trajectory_pair_curves.png">轨迹曲线</a></p>']
    for p in pages:
        rel=p.relative_to(out/'gallery').as_posix()
        body.append('<section><h2>'+html.escape(p.parent.name)+'</h2><a href="'+rel+'"><img src="'+rel+'" alt="固定双视图与位置截面对照"></a></section>')
    put(out/'gallery/index.html','\n'.join(body).encode('utf-8'))
    # Snapshot actual scoring mathematics and wrappers, without importing them.
    sources={C.resolve(p) for p in code}
    sources.update(Path(__file__).parent.glob('*.py'))
    relatives=['r2/pilots/bench_clean_multipart_seam_v1/post_cpu.py',
        'r2/pilots/seam_abc_v1/evaluate.py','r2/pilots/seam_abc_v1/eval_common.py',
        'r2/pilots/seam_followup_v1/binding.py','r2/pilots/seam_hinge_quality_v1/topology.py',
        'r2/pilots/user_condition_finetune_v1/evaluation.py',
        'r2/pilots/anchor_aware_denoiser_v1/evaluation.py',
        'r2/pilots/user_mix_from_official_v1/evaluation.py',
        'r2/tools/avmr_pmr_v1/run.py','r2/tools/c_restore_coupling_v1/evaluation.py',
        'r2/tools/postprocess_avmr_pmr_v1/quality.py',
        'r2/tools/hinge_v2_cpu_v1/scoring.py','r2/tools/hinge_v2_cpu_v1/holes.py',
        'r2/tools/partial_edge_binding_v1/metrics.py','r2/data_tools/user_conditions_v1/render.py']
    sources.update(C.ROOT/p for p in relatives)
    legacy=Path('/mnt/e/MeshFlow-Control/src/meshflow_control/evaluation')
    sources.update(legacy/n for n in ('__init__.py','control.py','geometry.py','shape.py','surface.py','quality.py','seams.py','pair_audit.py','intersections.py','hashing.py'))
    sources.add(Path('/mnt/e/MeshFlow-Control/tools/score_recipe_factorial_conflict.py'))
    snap=[]
    for p in sorted(sources):
        data=p.read_bytes();ast.parse(data.decode('utf-8-sig'), filename=str(p))
        rel=p.relative_to(C.ROOT) if p.is_relative_to(C.ROOT) else Path('external/MeshFlow-Control')/p.relative_to('/mnt/e/MeshFlow-Control')
        target=out/'source_archive'/rel;put(target,data)
        snap.append(dict(source=str(p),archive=str(target.relative_to(out)),sha256=C.sha(p),bytes=len(data)))
    C.write(out/'source_hashes.json',dict(scope='Actual model/runtime and metric mathematics with wrappers; source evidence, not a standalone dependency installation',files=snap))
    diff=[]
    for p in sorted(Path(__file__).parent.glob('*.py')):
        diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(True),fromfile='/dev/null',tofile=p.relative_to(C.ROOT).as_posix()))
    put(out/'code.diff',''.join(diff).encode('utf-8'))
    artifact_hashes={str(p.relative_to(out)):C.sha(p) for p in sorted(out.rglob('*'))
                     if p.is_file() and p.name not in ('COMPLETE.json','archive_receipt.json') and p.suffix not in ('.zip','.pyc')}
    C.write(out/'COMPLETE.json',dict(status='COMPLETE',protected_files_verified=len(reg['protected']),
        derived_input_files_verified=len(reg['inputs']),new_scientific_RAW=10,reused_RAW=6,validation_replay=1,
        generator_forward_started=550,generator_forward_complete=550,training_updates=0,backwards=0,D0_forwards=0,
        new_noise_arrays=0,new_POST=0,full_new_scores=10,extra_common_C_CF_scans=8,extra_FF_scans=0,
        unique_comparison_cells=16,actual_AI_PROXY_review_cells=16,view_pages=6,curve_pairs=12,curve_rows=612,
        GPU_processes=2,observed_concurrent_rollouts_peak=peak,sum_rollout_seconds=sampling_seconds,
        scientific_failures=0,scientific_retries=0,pre_execution_sandbox_denial=1,
        validation='Cached baseline replay bitwise identical; each worker state-hash assertion passed; all original registered assets verified',
        score_source_scope='Sampler hashes checked against execution; remaining score mathematics archived at closeout, unchanged original files',
        source_files=len(snap),artifact_hashes=artifact_hashes))
    # Small review package: all tables/images/code and meshes; full trajectories stay locally.
    archive=out/'R2_BENCH_C_TESSELLATION_REVIEW.zip'
    entries={}
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.suffix in ('.zip','.pyc','.ply') or p.name=='archive_receipt.json':continue
        if p.name in ('trajectory.npz','vectors.npz','common_C_vectors.npz'):continue
        entries[p.relative_to(out).as_posix()]=p
    for original,h in reg['protected'].items():
        p=C.resolve(original)
        if p.suffix=='.pt':continue
        rel=p.relative_to(C.ROOT)
        entries['reference/'+rel.as_posix()]=p
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            assert set(z.namelist())==set(entries)
            for name,p in entries.items():assert z.read(name)==p.read_bytes()
    else:
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name,p in sorted(entries.items()):z.write(p,name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    C.write(out/'archive_receipt.json',dict(path=str(archive),sha256=C.sha(archive),bytes=archive.stat().st_size,
        entries=len(entries),contains_weights=False,full_trajectory_location='outputs/*/trajectory.npz (local only)',
        scope='All per-output/per-curve tables, 16 raw meshes incl cached references, images, actual reviews, sources/diff/hashes; omits trajectory and per-query vector arrays'))
    print(json.dumps(dict(status='COMPLETE',forward_calls=550,review_cells=16,source_files=len(snap),archive_bytes=archive.stat().st_size,concurrent_rollouts_peak=peak),ensure_ascii=False))


if __name__=='__main__':main()
