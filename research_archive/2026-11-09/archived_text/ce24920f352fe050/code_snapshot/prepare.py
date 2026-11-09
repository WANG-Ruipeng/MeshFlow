"""Build immutable inputs for the bounded cached layer diagnosis."""
import csv
from . import common as c
SELECTION=[('73b369',3,'RAW_NO_TO_YES'),('73b369',4,'RAW_NO_TO_YES'),('3f7417',2,'RAW_NO_TO_YES'),('85b169',6,'RAW_NO_TO_YES'),('2d08a6',2,'RAW_NO_TO_YES'),('2d08a6',3,'RAW_NO_TO_YES'),('610b16',0,'RAW_NO_TO_YES'),('73b369',1,'RAW_YES_TO_UNCERTAIN'),('608b46',1,'RAW_YES_TO_UNCERTAIN'),('73b369',6,'HIGH_I_FF_TAIL'),('85b169',7,'INTERFACE_LOSS_STABLE_ADOPTION')]

def build():
    ev=c.BASE/'evaluation'; reg=c.read(ev/'registration.json')
    raw=c.read(ev/'raw_index.json')['rows']; post=c.read(ev/'post_index.json')['rows']
    oldrows=list(csv.DictReader((ev/'per_output.csv').open(encoding='utf-8-sig')))
    rows={(r['label'],r['request_id'],r['stage']):r for r in oldrows}
    queries={r['condition_id']:r for r in reg['queries']}
    parents=c.read(c.ROOT/'experiments/r2_user_data_scaling_12h_v1/data/qualified_condition_manifest.json')['parents']
    cells=[];requests=[];refs=[];usedrefs=set();protected={}
    def pin(path,expected=None):
        actual=c.sha(path)
        if expected:assert actual==expected, str(path)
        protected[str(c.resolve(path))]=actual
        return actual
    for prefix,n,why in SELECTION:
        found=[r for r in reg['requests'] if r['parent_id'].startswith(prefix) and int(r['candidate'])==n]
        assert len(found)==1,(prefix,n,len(found))
        q=found[0]; qry=queries[q['condition_id']]
        request=dict(q,selection_reason=why,short_id=prefix+'_n'+str(n));requests.append(request)
        base={k:q[k] for k in ('request_id','condition_id','parent_id','candidate','K','N','panel','condition_type','input_path')}
        base.update(selection_reason=why,short_id=request['short_id'],condition_source_path=q['condition_source'],query_path=qry['path'])
        for field,expected in [('input',q['input_sha256']),('condition_source',q['condition_source_sha256']),('query',qry['sha256'])]:
            base[field+'_sha256']=pin(base[field+'_path'],expected)
        for label in ('SMALL_C200','CREF_C200'):
            for stage,index,key in [('RAW',raw,'raw'),('POST',post,'post')]:
                r=next(r for r in index if r['request_id']==q['request_id'] and r['label']==label)
                old=rows[(label,q['request_id'],stage)]
                cell=dict(base,label=label,stage=stage,cell_id=request['short_id']+'__'+label+'__'+stage,
                    geometry_path=r[key+'_path'],score_path=old['score_path'],vectors_path=str(c.resolve(old['score_path']).parent/'vectors.npz'),
                    pairs_path=str(c.resolve(old['score_path']).parent/'pairs.json'))
                cell['geometry_sha256']=pin(cell['geometry_path'],r[key+'_sha256'])
                cell['score_sha256']=pin(cell['score_path'],old['score_sha256'])
                for f in ('vectors','pairs'):cell[f+'_sha256']=pin(cell[f+'_path'])
                c.load_geometry(cell)
                cells.append(cell)
        if q['condition_id'] not in usedrefs:
            usedrefs.add(q['condition_id'])
            parent=next(p for p in parents if p['parent_uid']==q['parent_id'])
            ref=dict(base,cell_id=prefix+'__SOURCE',label='SOURCE_REFERENCE',stage='SOURCE',
                geometry_path=parent['npz'],vertices_key='model_vertices',faces_key='source_faces',parent_manifest_original_vertices_key=parent['vertices_key'],parent_manifest_original_faces_key=parent['faces_key'])
            ref['geometry_sha256']=pin(ref['geometry_path'],parent['sha256'])
            c.load_geometry(ref);refs.append(ref)
    for path in [ev/'registration.json',ev/'raw_index.json',ev/'post_index.json',ev/'per_output.csv',c.BASE/'results/paired_decisions.csv']:
        pin(path)
    c.csvwrite(c.OUT/'inherited_per_output.csv',[rows[(x['label'],x['request_id'],x['stage'])] for x in cells])
    c.write(c.OUT/'protected_hashes.json',protected)
    value=dict(version=c.VERSION,status='INPUTS_REGISTERED_BEFORE_GEOMETRY_EXECUTION',requests=requests,cells=cells,references=refs,
        budgets=dict(requests=11,cached_model_stage_cells=44,source_reference_cells=6,new_model_calls=0,training_updates=0,new_noise=0,old_quality_rescores=0),
        interpretation='RESULT_SELECTED_MECHANISM_DIAGNOSIS_NOT_POPULATION_SUCCESS_RATE',
        source_reference_policy='LOCAL_THICKNESS_CONTEXT_ONLY_NOT_UNIQUE_GT_RANKING',
        input_preference='Historical K1 retained; future main inputs favor meaningful regular connected patches of >=2 triangles.',
        dispatch=dict(cpu_workers=2,threads_per_worker=1,gpu_workers=0),
        config=dict(L=c.L,parallel_angle_deg=15.,dup_distance_ratio=.005,interior_margin_ratio=.01,
                    local_corridor_ratio=.1,layer_merge_ratio=.0001,layer_cluster_rule='SORTED_ADJACENT_GAP',
                    graph='EXACT_SHARED_EDGES_NO_PROXIMITY_WELD',signed_side='SAVED_SOURCE_C_NORMAL_NOT_GLOBAL_INSIDE_OUTSIDE'))
    c.write(c.OUT/'registration.json',value)
    c.write(c.OUT/'resources_prepare.json',c.resources())
    c.event('INPUTS_REGISTERED',cells=len(cells),references=len(refs),protected=len(protected),model_calls=0)
    print({'cells':len(cells),'references':len(refs),'protected':len(protected)})
if __name__=='__main__':build()