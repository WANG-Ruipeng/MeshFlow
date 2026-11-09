"""One recorded part-wise box reconstruction, keeping12 original part extents."""
from .common import *
from .targets import labels_for, save_target

def freeze_recipe():
    path=OUT/'part_recipes/585673.json'
    if path.exists():return read(path)
    parent=next(p for p in freeze()['parents'] if p['parent_id'].startswith('585673'))
    a=load(parent['source']['clean_case_path']);labels=labels_for(parent,a)
    desc=read(OUT/'part_inspection'/(parent['parent_id']+'.json'))
    rows=[]
    for c in desc['components']:
        labels_c=np.unique(labels[c['face_ids']]);assert len(labels_c)==1
        assert c['topology']['open_edges']==0 and c['topology']['nonmanifold_edges']==0
        rows.append(dict(component_id=c['component_id'],source_face_ids=c['face_ids'],
          bbox_min=c['bbox_min'],bbox_max=c['bbox_max'],semantic_bits=int(labels_c[0]),
          role=('SEAT_BOARD' if c['component_id']>=8 else 'LEG' if c['component_id']<4 else 'UNDERSEAT_RAIL')))
    recipe=dict(parent_id=parent['parent_id'],source_sha256=sha(parent['source']['clean_case_path']),
      method='One axis-aligned closed box per original independent component, original component bbox,8 vertices/12 triangles;12 components,144 faces. No union/welding or removal. Original rounded/bevel surface is approximated explicitly.',
      components=rows,proposal_count=1,area_ratio=[.98,1.02],max_bidirectional_probe_L=.0025,
      rejection='Retain proposal as HELD if rounding, gaps, contacts, topology or area fails; do not fit/tune another box after results.',
      semantics='Four whole seat boards and original component0 leg; no split or guessed face label.')
    save(path,recipe);event('PART_RECIPE_FROZEN',parent=parent['parent_id'],path=str(path),sha256=sha(path));return recipe

def make(parent,recipe):
    a=load(parent['source']['clean_case_path']);verts=[];faces=[];lineage=[]
    corner=np.asarray([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],np.float64)
    triangles=np.asarray([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                         [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],np.int64)
    for r in recipe['components']:
        lo=np.asarray(r['bbox_min']);hi=np.asarray(r['bbox_max']);vs=lo+(hi-lo)*corner;off=len(verts)
        verts.extend(vs);faces.extend((triangles+off).tolist());lineage.extend([list(r['source_face_ids']) for _ in triangles])
    mv=np.asarray(verts,np.float32);tr=json.loads(str(a['transform_json']))
    norm=mv*np.float32(.3762/2.)
    source=(norm.astype(float)*tr['divisor3']+np.asarray(tr['center3'])+np.asarray(tr['center2']))*tr['divisor1']+np.asarray(tr['center1'])
    arr=dict(model_vertices=mv,normalized_vertices=norm,source_vertices=source.astype(np.float32),source_faces=np.asarray(faces,np.int64),transform_json=a['transform_json'])
    return arr,lineage

def main():
    recipe=freeze_recipe();parent=next(p for p in freeze()['parents'] if p['parent_id']==recipe['parent_id'])
    arr,lineage=make(parent,recipe)
    save_target(parent,arr,lineage,recipe,'part_rebuild',False)

if __name__=='__main__':main()
