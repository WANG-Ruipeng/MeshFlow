"""Read the actual active data once; no model import or geometry mutation."""
from . import common as c


def main():
    source = c.read(c.SOURCE)
    rows=[]
    for row in source['train']:
        a=c.arrays(row['case_path'])
        assert c.sha(row['case_path'])==row['case_sha256']
        rows.append(dict(parent_id=row['parent_id'],condition_id=row['condition_id'],
            case_path=row['case_path'],case_sha256=row['case_sha256'],N=int(a['N']),K=int(a['K']),
            keys={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in a.items()},
            transform_json=str(a.get('transform_json','MISSING')),
            source_faces_equals_model_ids=bool(c.np.array_equal(a.get('source_faces'),a.get('source_face_vertex_ids')))))
    c.write(c.OUT/'inventory.json',dict(source_manifest=str(c.SOURCE),source_sha256=c.sha(c.SOURCE),rows=rows))
    c.write(c.OUT/'resources_initial.json',c.resources())
    c.event('READ_ONLY_INVENTORY',parents=len(rows),model_calls=0)
    print(c.json.dumps(dict(parents=len(rows),N=[r['N'] for r in rows],first=rows[0]),indent=2))

if __name__=='__main__': main()
