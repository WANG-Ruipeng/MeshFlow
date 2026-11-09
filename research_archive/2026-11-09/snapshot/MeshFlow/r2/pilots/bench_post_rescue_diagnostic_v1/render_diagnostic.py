"""Three saved-stage examples: RAW, restored D0 proposal and original POST."""
from .register import OUT,SOURCE,read,write
from r2.pilots.seam_followup_v1 import gallery as G

def main():
    selected=('f5d370e9fe5b7452dbbc9440457e303e__BACK_SURFACE_1__n1',
              '9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n0',
              '1dd7409c95d59d0da192483aa282f8e5__SEAT_SURFACE_1__n0')
    qq={q['request_id']:q for q in read(SOURCE/'evaluation_inputs/manifest.json')['requests']}
    pages=[]
    for rid in selected:
        q=qq[rid];base=SOURCE/'evaluation/SEAM_S21000'/rid
        pages.append(dict(**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','K','N','split')},
            page_id=rid,stage='Saved geometry stages',kind='SEAM_S21000 deletion diagnosis; no new model calls',
            cells=[dict(label='RAW',path=str(base/'raw/raw.npz')),
                   dict(label='D0 + restore anchors (PRE_GEO)',path=str(base/'post/pre_geo.npz')),
                   dict(label='GEO_C + orientation (POST)',path=str(base/'post/post.npz'))]))
    name='RAW_PRE_GEO_POST'
    path=OUT/'gallery_manifests'/(name+'.json')
    if not path.exists():write(path,dict(pages=pages,model_calls=0,selection='Purposeful two severe deletion cases and one high-deletion retained-body counterexample from existing review; not population success estimation'))
    G.OUT=OUT;G.run(name)
if __name__=='__main__':main()
