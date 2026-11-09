"""Join actual CPU visual reviews and already-completed evidence; no rescoring."""
from pathlib import Path
from r2.pilots.seam_abc_v1 import eval_common as C
OUT=C.ROOT/'experiments/r2_seam_pool47_couple_v1'

def main():
    rows=[];sources={}
    for name in ('visual_review_cpu_root.json','visual_review_cpu_subset.json'):
        p=OUT/name;r=C.read(p);sources[str(p)]=C.sha(p);rows.extend(r['rows'])
    assert len(rows)==56 and len({(r['request_id'],r['candidate']) for r in rows})==56
    assert len({r['request_id'] for r in rows})==28
    for r in rows:
        assert r['actually_viewed'] is True and C.sha(r['image'])==r['image_sha256']
        assert r['topology_certified_by_images'] is False
    rows.sort(key=lambda r:(r['request_id'],r['candidate']))
    C.csvwrite(OUT/'cpu_binding/visual_review_complete.csv',rows)
    for name in ('cpu_binding/completion.json','cpu_binding_independent_review.json',
                 'corner_rule/completion.json','corner_rule/export_recovery/completion.json'):
        p=OUT/name;value=C.read(p);sources[str(p)]=C.sha(p)
        assert value.get('status') in ('COMPLETE','PASS','COMPLETE_WITH_EXPORT_ONLY_RECOVERY')
    C.write(OUT/'cpu_stage_completion.json',dict(status='COMPLETE',binding_cells=56,original_metric_fields=124,
        actual_review_images=28,paired_visual_rows=56,corner_conditions=50,model_calls=0,
        original_evidence_preserved=True,source_receipts=sources,
        visual='Nonblind AI_PROXY; no gross new body loss in these views, no established broad crack/fragment improvement; no topological certification',
        corner_decision='Not deployed; all9 source alias faces enumerated but rejected by frozen5degree gate; selected feasible patches do not identify source aliases',
        final_experiment_status='TRAINING_PENDING'))
if __name__=='__main__':main()
