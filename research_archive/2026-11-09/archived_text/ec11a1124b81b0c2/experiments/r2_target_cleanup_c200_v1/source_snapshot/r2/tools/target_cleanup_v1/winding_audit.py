"""Read frozen duplicate identities; distinguish coincident two-sided copies."""
import csv,json
from collections import Counter
from .prepare import OUT,read,sha,write,csvwrite

def main():
    source=OUT/'exact_duplicate_groups.csv'
    with source.open(encoding='utf-8-sig',newline='') as f:groups=list(csv.DictReader(f))
    rows=[]
    for r in groups:
        faces=json.loads(r['source_winding_rows']);canonical=json.loads(r['canonical_vertex_id_rows'])
        assert len(faces)==len(canonical)==2
        a,b=faces
        same=any(a==b[j:]+b[:j] for j in range(3))
        opposite=any(a==(b[j:]+b[:j])[::-1] for j in range(3))
        assert same!=opposite
        rows.append(dict(parent_id=r['parent_id'],task_id=r['task_id'],group_index=int(r['group_index']),source_face_rows=json.loads(r['source_face_rows']),
            retained_source_rows=json.loads(r['retained_source_rows']),removed_source_rows=json.loads(r['removed_source_rows']),source_vertex_id_rows=faces,
            source_winding_relation='OPPOSITE_CYCLIC' if opposite else 'SAME_CYCLIC',
            identical_canonical_vertex_id_rows=canonical[0]==canonical[1],canonical_vertex_id_rows=canonical,
            interpretation='Exact same finite triangle and original vertex-ID set; removing opposite source copy changes multiplicity/sidedness, not unoriented surface union'))
    csvwrite(OUT/'source_duplicate_winding_audit.csv',rows)
    value=dict(status='PASS',source_path=str(source),source_sha256=sha(source),groups=len(rows),parents=len({r['parent_id'] for r in rows}),
        source_winding_counts=dict(Counter(r['source_winding_relation'] for r in rows)),
        identical_canonical_vertex_id_rows=sum(r['identical_canonical_vertex_id_rows'] for r in rows),
        canonical_training_target_identity='Same original model vertices and identical canonical vertex rows imply bitwise identical canonical training triangles',
        original_source_intent='UNKNOWN; may have represented two-sided thin sheets; do not label all as accidental corrupt faces',
        retained_contract='C and all verified C/Y owner faces protected; exact coordinate union preserved; face multiplicity and opposite-winding semantics intentionally changed',
        new_geometry_pair_scans=0,new_model_calls=0,new_OT=0,entry_sha256=sha(__file__),rows_path=str(OUT/'source_duplicate_winding_audit.csv'),rows_sha256=sha(OUT/'source_duplicate_winding_audit.csv'))
    write(OUT/'source_duplicate_winding_audit.json',value);print(json.dumps(value,indent=2))

if __name__=='__main__':main()
