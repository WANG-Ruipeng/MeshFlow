"""Register all paired pages, with no cherry-picking by the resulting metric."""
from r2.pilots.seam_abc_v1 import eval_common as C
OUT=C.ROOT/'experiments/r2_seam_followup_v1'
OLD=C.ROOT/'experiments/r2_seam_abc_v1'

def main():
    m=C.read(OLD/'eval_manifest.json');layout=[];binding=[]
    for q in m['requests']:
        common={k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','K','N','split')}
        for stage in ('RAW','POST'):
            cells=[dict(label=a,path=str(OLD/'evaluation'/a/q['request_id']/stage.lower()/(stage.lower()+'.npz'))) for a in ('LONG','B_POOL47')]
            cells.append(dict(label='B_LONG_LAYOUT',path=str(OUT/'layout/evaluation/B_LONG_LAYOUT'/q['request_id']/stage.lower()/(stage.lower()+'.npz'))))
            layout.append(dict(**common,page_id=q['request_id']+'_'+stage,stage=stage,kind='Frozen B / layout intervention',cells=cells))
        for a in ('LONG','A_COUPLE'):
            cells=[dict(label=a,path=str(OLD/'evaluation'/a/q['request_id']/'post/post.npz'))]
            for cap in ('015','020'):
                label=a+'_BIND_'+cap;cells.append(dict(label=label,path=str(OUT/'binding/cells'/label/q['request_id']/'post.npz')))
            binding.append(dict(**common,page_id=a+'_'+q['request_id']+'_POST',stage='POST',kind='CPU midpoint cluster binding',cells=cells))
    C.write(OUT/'gallery_manifests/layout.json',dict(pages=layout,model_calls=0))
    C.write(OUT/'gallery_manifests/binding.json',dict(pages=binding,model_calls=0))
    print('Registered layout56 / binding56 comparison pages')

if __name__=='__main__':main()
