"""Read-only source verification and one-time receipt of this CPU audit."""
import csv
import hashlib
import json
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def resolve(value):
    s = str(value).replace('\\', '/')
    for prefix in ('/mnt/e/MeshFlow/', 'E:/MeshFlow/'):
        if s.startswith(prefix):
            return ROOT / s[len(prefix):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def save_new(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def main():
    start = time.time()
    completed = OUT / 'completion.json'
    if completed.exists():
        receipt = read(completed)
        for rel, digest in receipt['output_hashes'].items():
            assert sha(OUT / rel) == digest, rel
        for rel, digest in receipt['source_hashes'].items():
            assert sha(resolve(rel)) == digest, rel
        print(json.dumps(dict(status='VERIFIED_CACHED_COMPLETE', model_calls=0,
                             outputs=len(receipt['output_hashes'])), indent=2))
        return
    source_hashes = {}
    for relative in ('context_source_hashes.json', 'geometry/input_hashes.json',
                     'coverage/source_hashes.json', 'geometry/archive_supply/input_hashes.json'):
        for name, value in read(OUT / relative).items():
            digest = value['sha256'] if isinstance(value, dict) else value
            p = resolve(name)
            assert sha(p) == digest, str(p)
            source_hashes[str(p.relative_to(ROOT)).replace('\\', '/')] = digest
    observations = [
        ('experiments/r2_bench_parts_v1/gallery/candidates/9fd5b67a2b20b5d1b19874039c8f3e05.png',
         '来源图：座面候选是一条窄座板上表面；腿候选为完整U形端架，含两腿与上横梁。不是相同完整程度的部件。'),
        ('experiments/r2_bench_parts_v1/gallery/candidates/459f90aa2162b1f1d46c340938e2ff1c.png',
         '来源图：历史LEG条件实际为座板加一侧弯曲端支撑，连贯且占比大。'),
        ('experiments/r2_bench_parts_v1/gallery/candidates/f5d370e9fe5b7452dbbc9440457e303e.png',
         '来源图：宽矩形座面为两三角形，另有宽靠背面；视觉可辨认，不认证隐藏几何质量。'),
        ('experiments/r2_bench_state_schedule_v1/gallery/CLEAN_FINAL_POST__9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0.png',
         '生成图：蓝U形端架有合理端部支撑用途，灰座面延伸到另一端；截面仍见局部高度差。合理用途不认证合法接续。'),
        ('experiments/r2_bench_state_schedule_v1/gallery/CLEAN_FINAL_POST__9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n0.png',
         '生成图：三模型灰座面低于蓝窄板，橙seam与灰bulk之间仍有错层；截面可见。'),
        ('experiments/r2_bench_state_schedule_v1/gallery/CLEAN_FINAL_POST__459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0.png',
         '生成图：连贯蓝座板及支撑上方仍有灰色分离/碎裂平行板，三个模型都未消除宏观层差。'),
    ]
    rows = [dict(path=p, sha256=sha(ROOT / p), reviewer='AI_PROXY_ROOT',
                 actual_image_viewed=True, blinded=False, observation=note,
                 formal_rescore=False) for p, note in observations]
    for r in rows:
        source_hashes[r['path']] = r['sha256']
    save_new(OUT / 'visual_observations.json', dict(rows=rows, pages=6,
             scope='Actual image tool views in this audit; not a new full-panel review.'))
    with (OUT / 'geometry/per_parent.csv').open(encoding='utf-8-sig', newline='') as f:
        parents = list(csv.DictReader(f))
    passing = [p for p in parents if p['group']=='TRAIN' and p['current_active']=='True'
               and int(p['source_ID_excess_faces'])==0 and int(p['exact_zero_area_faces_model64'])==0]
    assert len(passing)==15 and sum(int(p['current_active_conditions']) for p in passing)==47
    output_hashes = {str(p.relative_to(OUT)).replace('\\', '/'): sha(p)
                     for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    save_new(completed, dict(status='COMPLETE_CPU_AUDIT', source_hashes=source_hashes,
        output_hashes=output_hashes, all_source_hashes_unchanged=True,
        original_parents=72, original_conditions=316, archive_Bench_parents=560,
        current_TRAIN_parents=58, current_TRAIN_conditions=195,
        existing_outcomes_joined=432, additional_actual_image_pages=6,
        model_calls=0, GPU_calls=0, training_updates=0, generated_meshes=0,
        source_mutations=0, full_triangle_selfintersection='NOT_RUN',
        final_verification_seconds=time.time()-start,
        next_task='User separately authorized coverage expansion + one1500-step arm; not included here.'))
    print(json.dumps(dict(status='COMPLETE_CPU_AUDIT', protected_files=len(source_hashes),
                          output_files=len(output_hashes)), indent=2))

if __name__=='__main__':
    main()
