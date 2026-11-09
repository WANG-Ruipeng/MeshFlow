"""Independent CPU read-only provenance/statistics review; no model forward or export."""
from pathlib import Path
import sys,os,json
sys.path.insert(0,str(Path.cwd().resolve()))
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from copy import deepcopy
from experiments.a_radial.fm_tail import artifacts,train
from experiments.a_radial.system_probe import ours_metrics as common
from experiments.a_radial.system_probe.ours_package import compare,require,read
from experiments.a_radial.io_utils import file_hash,array_hash,write_json

run=Path('output/a_fm_tail_20260925_03').resolve();out=Path(__file__).resolve().parent
require(not torch.cuda.is_initialized(),'No GPU');torch.set_num_threads(1)
p,ap,artifact_sha=artifacts.artifact_contract(run);protocol_sha=file_hash(run/'RUN_PROTOCOL.json')
source=Path(p['source_run']);old_protocol=read(source/'RUN_PROTOCOL.json');bayes=read(source/'representation_diagnostics/BAYES_ORACLE_REVIEW.json')
with np.load(run/'data/radial_train.npz',allow_pickle=False) as z:y=z['logr'].astype(np.float64);uids=z['uids'].tolist()
pair=read(run/'artifacts/paired_8k_64k/RESULTS.json');pair_rows={(r['seed'],r['sample_id']):r for r in pair['rows']}
require(len(pair_rows)==len(pair['rows'])==pair['planned']==pair['successful']==128,'Pairing denominator')
training=[];identities={};independent_pair=[];independent_aux=[];bound_files={}

def bind(path):bound_files[str(path)]=file_hash(path)

def node_stats(a,b):
    aa=a['nodes'].astype(np.float64);bb=b['nodes'].astype(np.float64)
    da=np.sqrt(np.square(y-aa).mean(-1));db=np.sqrt(np.square(y-bb).mean(-1))
    require(a['noise'].dtype==b['noise'].dtype and np.array_equal(a['noise'],b['noise']),'Different fixed epsilon')
    return {'paired_logradius_node_RMS':float(np.sqrt(np.square(aa-bb).mean())),
        'first_nearest_training_logradius_RMS':float(da.min()),'second_nearest_training_logradius_RMS':float(db.min()),
        'first_nearest_parent_index':int(da.argmin()),'second_nearest_parent_index':int(db.argmin())}

for seed in (10,11):
    key=str(seed);job=p['jobs'][key];identity=artifacts.final_inputs(run,seed);identities[key]=identity
    inv=Path(identity['fm_path']).parent;parent=torch.load(run/job['parent_checkpoint'],map_location='cpu',weights_only=True)
    initial=torch.load(inv/'step00008000.pt',map_location='cpu',weights_only=True)
    compare(initial['model'],parent['model'],'Inherited8000 model')
    compare(initial['rng'],parent['rng'],'Inherited8000 full RNG')
    compare(initial['batch_rng'],parent['batch_rng'],'Inherited8000 batch generator')
    compare(initial['draw_queue'],parent['draw_queue'],'Inherited8000 sampler queue')
    expected_opt=deepcopy(parent['optimizer'])
    for group in expected_opt['param_groups']:group['lr']=group['initial_lr']=1e-4
    compare(initial['optimizer'],expected_opt,'Inherited Adam except declared new LR')
    compare(initial['source_scheduler'],parent['scheduler'],'Archived original cosine scheduler')
    compare(initial['source_config'],parent['config'],'Archived original source config')
    require(initial['step']==8000 and initial['scheduler']['last_epoch']==8000
        and initial['scheduler']['base_lrs']==[1e-4] and not initial['is_predeclared_final'],'New scheduler initial state')
    del parent,initial,expected_opt
    probe=train.load_probe(run,job)
    require(job['fixed_probe_array_sha256']==bayes['seeds'][key]['probe_sha256'],'Probe differs from original8k')
    require(tuple(probe['xt'].shape)==(4096,64) and tuple(probe['targets'].shape)==(16,64),'Fixed probe shapes')
    observations=[]
    for step in (8000,16000,32000,64000):
        path=inv/f'probe{step:08d}.json';row=read(path);bind(path)
        require(row['step']==step and row['selection_role']=='observation_only' and row['probe_sha256']==job['fixed_probe_array_sha256'],'Probe selection/hash')
        require(row['probe_count']==4096 and row['per_parent_count']==256 and sum(b['count'] for b in row['t_bins'])==4096,'Probe denominator')
        require(row['oracle_pair_MSE']==bayes['seeds'][key]['oracle_pair_MSE'],'Oracle target changed')
        if step==8000:
            for name in ('network_pair_MSE','oracle_pair_MSE','network_vs_conditional_mean_MSE'):compare(row[name],bayes['seeds'][key][name],'Source probe'+name)
        observations.append(row)
    done=read(run/f'training/seed_{seed}/TRAINING_COMPLETE.json')
    training.append({'seed':seed,'new_updates':done['new_updates'],'final_step':64000,'final_checkpoint_sha256':identity['fm_sha256'],
        'Adam_rng_sampler_inheritance_checked':True,'new_declared_constant_minLR_stage':1e-4,
        'observations':observations,'observed_excess_MSE_reduction_fraction_8k_to64k':1-observations[-1]['network_vs_conditional_mean_MSE']/observations[0]['network_vs_conditional_mean_MSE'],
        'late_t_pair_MSE_8k_to64k':[observations[0]['t_bins'][-1]['network_pair_MSE'],observations[-1]['t_bins'][-1]['network_pair_MSE']]})
    primary=artifacts.view_samples(run,seed);aux=artifacts.view_samples(run,seed,True);old=common.seed_inputs(source,seed,old_protocol)
    aux_path=run/f'artifacts/seed{seed}/auxiliary_e128/diagnostics/RESULTS.json';ar=read(aux_path);bind(aux_path)
    aux_rows={r['sample_id']:r for r in ar['rows']};require(len(aux_rows)==64 and ar['success']==64 and ar['failed']==0,'Auxiliary denominator')
    for sid in common.expected_ids():
        before,bi=common.load_sample(old,sid);a,ai=common.load_sample(primary,sid);b,ab=common.load_sample(aux,sid)
        compare(primary['identity'],aux['identity'],'Primary/aux same final checkpoint')
        require(primary['sample_manifest']['fm_steps']==32 and aux['sample_manifest']['fm_steps']==128,'Primary/aux integration contract')
        for lhs,rhs,row in [(before,a,pair_rows[(seed,sid)]),(a,b,aux_rows[sid])]:
            require(row['status']=='ok','Node diagnostic failed')
            calc=node_stats(lhs,rhs)
            for name,value in calc.items():compare(value,row[name],'Independent node statistic '+name)
            require(array_hash(lhs['nodes'])==row['first_nodes_sha256'] and array_hash(rhs['nodes'])==row['second_nodes_sha256'],'Node result provenance')
        require(pair_rows[(seed,sid)]['first_sample_sha256']==bi['sample_npz_sha256']
            and pair_rows[(seed,sid)]['second_sample_sha256']==ai['sample_npz_sha256'],'Cross-run file pairing')
        require(aux_rows[sid]['primary_sample_sha256']==ai['sample_npz_sha256'] and aux_rows[sid]['aux_sample_sha256']==ab['sample_npz_sha256'],'Aux file pairing')
        independent_pair.append({'seed':seed,'sample_id':sid,**node_stats(before,a)})
        independent_aux.append({'seed':seed,'sample_id':sid,**node_stats(a,b)})
    for x in (run/job['fixed_probe'],run/f'artifacts/seed{seed}/samples/SAMPLES.json',run/f'artifacts/seed{seed}/auxiliary_e128/samples/SAMPLES.json'):bind(x)
    require(not (run/f'artifacts/seed{seed}/auxiliary_e128/exports').exists(),'Auxiliary exported into geometry table')
    compare(float(np.mean([r['paired_logradius_node_RMS'] for r in independent_aux if r['seed']==seed])),ar['mean_paired_node_RMS'],'Auxiliary mean')
    for side in ('first','second'):
        counts={uid:sum(r[side+'_nearest_parent_index']==i for r in independent_aux if r['seed']==seed) for i,uid in enumerate(uids)}
        compare(counts,ar['nearest_parent_counts'][side],'Aux parent counts')
    print(json.dumps({'seed':seed,'fixed_probes':4,'primary_E32':64,'auxiliary_E128':64,'same_epsilon_and_arrays_verified':True}),flush=True)

summaries=[]
for seed in ('pooled',10,11):
    rows=[r for r in independent_pair if seed=='pooled' or r['seed']==seed]
    auxrows=[r for r in independent_aux if seed=='pooled' or r['seed']==seed]
    g=next(g for g in pair['groups'] if g['seed']==seed)
    nearest={side:float(np.mean([r[side+'_nearest_training_logradius_RMS'] for r in rows])) for side in ('first','second')}
    compare(nearest,g['nearest_training_logradius_RMS_mean'],'Pair nearest mean')
    compare(float(np.mean([r['paired_logradius_node_RMS'] for r in rows])),g['paired_node_RMS_mean'],'Pair node mean')
    counts={side:{uid:sum(r[side+'_nearest_parent_index']==i for r in rows) for i,uid in enumerate(uids)} for side in ('first','second')}
    compare(counts,g['nearest_parent_counts'],'Pair parent counts')
    summaries.append({'seed':seed,'n':len(rows),'nearest_training_logradius_RMS_mean_8k_64k':nearest,
        'nearest_train_distance_reduction_fraction':1-nearest['second']/nearest['first'],
        'paired_same_noise_node_RMS_8k_to64k_mean':g['paired_node_RMS_mean'],
        '64k_E32_toE128_node_RMS_mean':float(np.mean([r['paired_logradius_node_RMS'] for r in auxrows])),
        'nearest_parent_coverage_count_8k_64k':{side:sum(c>0 for c in counts[side].values()) for side in ('first','second')},
        'nearest_parent_counts_8k_64k':counts,
        'E32_E128_nearest_parent_switches':sum(r['first_nearest_parent_index']!=r['second_nearest_parent_index'] for r in auxrows)})
for path in (run/'RUN_PROTOCOL.json',run/'ARTIFACT_PROTOCOL.json',run/'GPU_GATE.json',run/'artifacts/paired_8k_64k/RESULTS.json'):bind(path)
require(not torch.cuda.is_initialized(),'Unexpected CUDA')
result={'status':'PROBE_AND_SAVED_FIELDS_REVIEW_PASS_GEOMETRY_SUMMARY_PENDING','scope':'independent CPU read-only identity and saved-node statistics review',
    'source_sha256':file_hash(__file__),'bound_reviewed_files':bound_files,'training_protocol_sha256':protocol_sha,'artifact_protocol_sha256':artifact_sha,
    'training':training,'identities':identities,'primary_E32_registered_fields':128,'aux_E128_registered_fields':128,
    'shared_original_8k_epsilon':True,'sample_or_checkpoint_selection':False,'network_forward_executed':False,
    'GPU_or_export_executed':False,'source_run_modified':False,'node_summaries':summaries,
    'limits':['closed16 training support only','nearest-training node error is neither a surface nor distribution/generalization metric',
        'same epsilon across8k64k does not imply same mainfield','aux128 stays separate and never replaces primary32',
        '56k added updates and declared new minimum-LR stage are both changed; not an unchanged cosine continuation']}
write_json(out/'PROBE_FIELDS_REVIEW.json',result)
print(json.dumps({'status':result['status'],'node_summaries':summaries}),flush=True)
