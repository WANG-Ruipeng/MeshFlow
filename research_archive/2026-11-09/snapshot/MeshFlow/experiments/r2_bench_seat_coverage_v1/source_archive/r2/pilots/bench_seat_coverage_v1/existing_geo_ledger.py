"""Collect existing GEO_C deletion receipts only; no geometric re-evaluation."""
import collections, csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_seat_coverage_v1'
OLD = ROOT / 'experiments/r2_bench_full_seat_curriculum_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    dest = OUT / 'results/existing_geo_deletion_ledger'
    done = dest / 'completion.json'
    if done.exists():
        d = json.loads(done.read_text('utf-8'))
        for p, h in d['files'].items():
            assert sha(Path(p)) == h
        return d
    with (OUT/'results/per_output.csv').open(encoding='utf-8-sig', newline='') as f:
        source = list(csv.DictReader(f))
    rows=[]; sources={}; reasons=collections.Counter()
    for r in source:
        if r['stage'] != 'BASE_POST': continue
        root = OLD if r['result_origin']=='REUSED_READ_ONLY' else OUT
        p = root/'evaluation'/r['label']/r['request_id']/'post/geometry.json'
        g = json.loads(p.read_text('utf-8'))
        s = g['stats']; count = collections.Counter(s['deleted_by_reason'])
        assert sum(count.values()) == s['deleted_free_faces'] == int(float(r['deleted_free_faces']))
        records=collections.Counter(x['reason'] for x in g['face_records'] if not x['known'] and not x['kept'])
        assert count == records
        assert s['input_faces']-s['known_faces']-s['deleted_free_faces'] == int(float(r['actual_free_faces']))
        sources[str(p)] = sha(p); reasons.update(count)
        rows.append(dict(label=r['label'],request_id=r['request_id'],parent_id=r['parent_id'],
            noise_index=r['noise_index'],eval_group=r['eval_group'],result_origin=r['result_origin'],
            input_free_faces=s['input_faces']-s['known_faces'],output_free_faces=s['output_faces']-s['known_faces'],
            deleted_free_faces=s['deleted_free_faces'],**{'deleted_'+k:v for k,v in count.items()},
            radius_model=s['radius_model'],receipt_path=str(p),receipt_sha256=sources[str(p)]))
    assert len(rows)==84
    keys=list(dict.fromkeys(k for r in rows for k in r))
    reason_keys=sorted(k for k in keys if k.startswith('deleted_') and k!='deleted_free_faces')
    for r in rows:
        for k in reason_keys: r.setdefault(k,0)
    dest.mkdir(parents=True, exist_ok=True)
    p=dest/'per_output.csv'
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    result=dict(status='COMPLETE',outputs=84,existing_receipt_sources=sources,files={str(p):sha(p)},
        code_sha256=sha(Path(__file__)),model_calls=0,scoring_calls=0,
        interpretation='Reads original GEO_C face records. Repeated vertex denotes collapse in recorded consolidated indices; this does not identify which upstream stage caused the narrow geometry or certify every deleted triangle was useful.')
    done.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
if __name__=='__main__':
    print({k:v for k,v in run().items() if k not in ('files','existing_receipt_sources')})
