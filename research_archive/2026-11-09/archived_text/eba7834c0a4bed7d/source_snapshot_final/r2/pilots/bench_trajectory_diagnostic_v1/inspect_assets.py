from .common import *
import numpy as np
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'resources_initial.json',resources())
    m=read(V2/'data/manifest.json');e=read(V2/'eval_manifest.json')
    print('TRAIN',[(r['parent_id'][:6],r['part_type'],r['K'],r['N'],r['condition_id']) for r in m['train']])
    print('DEV',[(q['parent_id'][:6],q['part_type'],q['K'],q['N'],q['condition_id']) for q in e['requests'] if q['split']=='DEV' and q['noise_index']==0])
    for name,r in [('train',m['train'][0]),('dev',m['dev'][0])]:
        a=arrays(r['case_path']);print(name, {k:(v.shape,str(v.dtype)) for k,v in a.items()})
    for name in ('metadata_npz','old_layout_npz'):
        print(name,e['requests'][0][name], {k:v.shape for k,v in arrays(e['requests'][0][name]).items()})
    print('OLDHEAD',list((ROOT/'experiments/r2_bench_parts_convergence_v1').glob('training/**/*15000*.pt')))
    print('HEADFILES',list((ROOT/'experiments/r2_bench_parts_convergence_v1').glob('**/*head*.pt')))
    print('RES',read(OUT/'resources_initial.json'))
if __name__=='__main__':main()
