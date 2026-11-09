"""CPU-only fixed-panel candidate summaries, loss curves and stage-correct views.

All scores, point queries, geometry and original sources are read-only.  The
177 native-C fields are copied verbatim; original-C diagnostics and reductions
of saved face/edge records have separate columns and versions.  Neither graph
components nor low conflict fractions certify visible completeness or success.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import defaultdict
import concurrent.futures
import csv
import html
import io
import json
import math
import multiprocessing
from pathlib import Path
import resource
import sys
import time
import numpy as np
from r2.pilots.seam_abc_v1 import eval_common as C
from .analyze import COMMON_METRICS, _aggregate, _numeric, _shape_row

OUT=C.ROOT/'experiments/r2_bench_c_edgeflip_v1'
EVAL=OUT/'candidate_evaluation'
DEST=EVAL/'analysis'
GALLERY=EVAL/'gallery'
LABELS=('EXPOSURE','PLAIN200','FLIP200')
STAGES=('RAW','CLEAN_FINAL_POST')
VERSION='bench_edgeflip_candidates_fixed30_RAW_FINAL_analysis_v1'
GRAPH_VERSION='existing_index_edge_cache_component_reduction_v1'


def _verify(book):
    for path,digest in book.items():
        if C.sha(path)!=digest:
            raise ValueError('Protected file changed: '+str(path))


def _cached(path):
    if not path.exists():
        return None
    saved=C.read(path)
    if saved['source_sha256']!=C.sha(__file__):
        raise ValueError('Completed analysis source changed')
    _verify(saved.get('protected',{}));_verify(saved['files'])
    return saved


def panel():
    reg=C.read(EVAL/'registration.json');_verify(reg['protected'])
    cells=C.read(EVAL/'baseline.json')+C.read(EVAL/'tasks.json')
    if len(cells)!=30 or len({c['cell_id'] for c in cells})!=30:
        raise ValueError('Expected thirty unique output cells')
    by=defaultdict(list)
    for cell in cells:
        by[cell['label']].append(cell)
    if set(by)!=set(LABELS) or any(len(v)!=10 for v in by.values()):
        raise ValueError('Expected the same ten input/noise requests per model')
    identities=lambda rows:{(c['representation'],c['q']['request_id']) for c in rows}
    if any(identities(by[label])!=identities(by['EXPOSURE']) for label in LABELS):
        raise ValueError('Candidate input panels differ')
    if sum(c['representation']=='ORIGINAL' for c in cells)!=18:
        raise ValueError('Expected six original plus four legal flips per model')
    return sorted(cells,key=lambda c:(c['representation'],c['q']['request_id'],LABELS.index(c['label'])))


def _metadata(cell,stage):
    q=cell['q']
    return dict(cell_id=cell['cell_id'],label=cell['label'],representation=cell['representation'],stage=stage,
        parent_id=q['parent_id'],parent_prefix=q['parent_id'][:4],request_id=q['request_id'],
        condition_id=q['condition_id'],noise_index=int(q['noise_index']),split=q['split'],
        report_group=q.get('report_group'),eval_group=q.get('eval_group'),
        N_requested=int(q['N']),K=int(q['K']),requested_free_faces=int(q['N'])-int(q['K']),
        origin=cell.get('baseline_origin',cell.get('origin')),
        evaluation_scope='ORIGINAL_THREE_PARENTS' if cell['representation']=='ORIGINAL' else 'LEGAL_FLIP_TWO_PARENTS')


def _raw_receipt(path):
    for name in ('complete.json','receipt.json'):
        if (path.parent/name).is_file():
            return path.parent/name
    return path.parent/'complete.json'


def spec(cell,stage):
    if stage=='RAW':
        mesh=C.resolve(cell['raw_path'])
        score=C.resolve(cell['score_path']) if 'score_path' in cell else C.resolve(cell['raw_score_path']) if 'raw_score_path' in cell else EVAL/'raw_scores'/cell['cell_id']
        producer=_raw_receipt(mesh)
        common=score/'common_C.json' if cell['representation']=='EDGE_FLIP' else None
    elif stage=='CLEAN_FINAL_POST':
        mesh=C.resolve(cell['final_path']) if 'final_path' in cell else EVAL/'cpu'/cell['cell_id']/'CLEAN_FINAL_POST/post.npz'
        score=C.resolve(cell['final_score_path']) if 'final_score_path' in cell else mesh.parent/'scores'
        producer=mesh.parent.parent/'complete.json'
        common=EVAL/'final_common_C'/cell['cell_id']/'common_C.json' if cell['representation']=='EDGE_FLIP' else None
    else:
        raise ValueError(stage)
    paths=[mesh,producer,score/'complete.json',score/'row.json',score/'score.json',score/'shape_per_face.json',score/'valid_edges.csv']
    if common is not None:
        paths.append(common)
        if stage=='CLEAN_FINAL_POST':paths.append(common.parent/'complete.json')
    return dict(cell=cell,stage=stage,mesh=mesh,score=score,producer=producer,common=common,required=paths)


def _all_specs(require_complete=True):
    cells=panel();specs=[spec(c,s) for c in cells for s in STAGES]
    missing=[str(p) for s in specs for p in s['required'] if not p.is_file()]
    if missing and require_complete:
        raise FileNotFoundError(json.dumps(dict(status='PENDING_ALL_BACKENDS',missing_count=len(missing),first_missing=missing[:12])))
    return cells,specs,missing


def _authenticate(s,protected):
    for receipt in (s['producer'],s['score']/'complete.json'):
        saved=C.read(receipt)
        if saved.get('status') not in ('COMPLETE','PASS'):
            raise ValueError('Incomplete producer/score: '+str(receipt))
        _verify(saved['files'])
    producer=C.read(s['producer'])
    if not any(C.resolve(p)==s['mesh'] and digest==C.sha(s['mesh']) for p,digest in producer['files'].items()):
        raise ValueError('Mesh not sealed by producer: '+str(s['mesh']))
    if s['common'] is not None and s['stage']=='CLEAN_FINAL_POST':
        _verify(C.read(s['common'].parent/'complete.json')['files'])
    for p in s['required']:
        protected[str(p)]=C.sha(p)


def _common(cell,stage,score,path):
    result=dict(_metadata(cell,stage),metric_scope='COMMON_ORIGINAL_PHYSICAL_C',
        original_K=int(cell['original_q']['K']),actual_free_denominator=int(score['free_face_denominator']),
        fixed_C_query_count=2048)
    result['CF_tested_pair_denominator']=result['original_K']*result['actual_free_denominator']
    if cell['representation']=='ORIGINAL':
        metrics,boundary=score['metrics'],score['boundary'];result['common_origin']='REUSED_NATIVE_ORIGINAL_C_SCORE'
    else:
        saved=C.read(path);old=cell['original_q']
        if saved['status']!='PASS' or saved['fixed_free_face_denominator']!=result['actual_free_denominator'] or saved['original_K']!=result['original_K']:
            raise ValueError('Common-C denominator/status differs')
        if saved['original_C_query_sha256']!=old['query_descriptor']['sha256'] or saved['original_C_input_sha256']!=C.sha(old['inference_npz']):
            raise ValueError('Common original-C identities differ')
        metrics,boundary=saved['metrics'],saved['boundary'];result['common_origin']='EXTRA_ORIGINAL_C_CF_I_SCAN'
    for key in COMMON_METRICS:
        result[key]=metrics[key]
    for row in boundary:
        for key in ('status','boundary_vertex_count','matched_count','VMR','perfect','threshold_model','scale','tau'):
            result[f'boundary__{row["setting"]}__{key}']=row[key]
    result['topology_scope']='Common-C CF/I/boundary diagnostics only. Actual-C indexed connectivity remains in the unchanged native177 table.'
    return result


def _graphs(s,shape,native):
    """Original component definition, cached legal edges and cached face areas."""
    from r2.pilots.seam_hinge_quality_v1.topology import components,edges_file
    mesh=C.arrays(s['mesh']);faces=np.asarray(mesh['faces'],np.int64);k=int(s['cell']['q']['K']);n=len(faces)
    per=shape['whole']['per_face']
    if len(per)!=n or [int(p['face']) for p in per]!=list(range(n)):
        raise ValueError('Cached whole-face area order differs')
    area=np.asarray([p['area'] for p in per],np.float64);free_area=float(area[k:].sum())
    edges=edges_file(s['score']/'valid_edges.csv')
    by_v=defaultdict(set);by_e=defaultdict(set)
    for i,face in enumerate(faces):
        for v in face:by_v[int(v)].add(i)
        for j in range(3):by_e[tuple(sorted((int(face[j]),int(face[(j+1)%3]))))].add(i)
    def links(groups):
        result=[]
        for group in groups:
            ids=sorted(group)
            result.extend((ids[i],ids[j]) for i in range(len(ids)) for j in range(i+1,len(ids)))
        return result
    graphs=(('vertex',links(by_v.values())),('edge_unfiltered',links(by_e.values())),
        ('edge_legal',[tuple(map(int,e['incident_face_ids'])) for e in edges if e['valid_edge']]),
        ('edge_legal_opposed',[tuple(map(int,e['incident_face_ids'])) for e in edges if e['valid_edge'] and e['winding_opposite']]))
    out=dict(_metadata(s['cell'],s['stage']),graph_version=GRAPH_VERSION,
        graph_origin='SAVED_ACTUAL_INDICES_AND_CACHED_VALID_EDGE_FLAGS_AND_CACHED_AREAS',actual_free_faces=n-k)
    records={}
    for name,links_value in graphs:
        _,groups,_=components(n,links_value,k,area)
        for group in groups:
            ff=[i for i in group['face_indices'] if i>=k]
            group['free_face_count']=len(ff);group['free_area_model2']=float(area[ff].sum())
        largest=max(groups,key=lambda group:(group['free_area_model2'],-group['component_id']))
        singles=sum(group['face_count']==1 and group['face_indices'][0]>=k for group in groups)
        out.update({f'derived_{name}_component_count':len(groups),
            f'derived_{name}_largest_free_area_model2':largest['free_area_model2'],
            f'derived_{name}_largest_free_area_fraction':largest['free_area_model2']/free_area if free_area else None,
            f'derived_{name}_largest_component_free_faces':largest['free_face_count'],
            f'derived_{name}_isolated_single_free_faces':singles})
        records[name]=groups
        check={'edge_legal':'hinge_diag_isolated_1face_edge_components','edge_unfiltered':'hinge_diag_isolated_1face_unfiltered_edge_components'}.get(name)
        if check and singles!=int(native[check]):
            raise ValueError('Cached graph singletons disagree with original scorer')
    out['interpretation']='Index-based graph statistics. RAW soup has separate free vertices by construction; component/singleton counts are not visible-fragment counts. FINAL legality-filtered separation also differs from physical tearing.'
    return out,records


def _model_pairs(rows,fields):
    by={(r['label'],r['representation'],r['request_id'],r['stage']):r for r in rows};result=[]
    identities=sorted({key[1:] for key in by})
    for representation,rid,stage in identities:
        for reference,candidate in (('EXPOSURE','PLAIN200'),('EXPOSURE','FLIP200'),('PLAIN200','FLIP200')):
            a,b=by[reference,representation,rid,stage],by[candidate,representation,rid,stage]
            out={k:b[k] for k in ('representation','request_id','condition_id','stage','parent_id','parent_prefix','noise_index','split','evaluation_scope')}
            out.update(reference=reference,candidate=candidate,contrast='SAME_INPUT_SAME_NOISE_MODEL_COMPARISON')
            for field in fields:
                av,bv=a.get(field),b.get(field);out[field+'__reference']=av;out[field+'__candidate']=bv
                out[field+'__delta']=bv-av if _numeric(av) and _numeric(bv) else None
            result.append(out)
    if len(result)!=60:raise ValueError('Expected sixty same-input model contrasts')
    return result


def _representation_pairs(rows,fields):
    by={(r['label'],r['representation'],r['request_id'],r['stage']):r for r in rows};result=[]
    for (label,rep,rid,stage),b in sorted(by.items()):
        if rep!='EDGE_FLIP':continue
        a=by[label,'ORIGINAL',rid,stage]
        out={k:b[k] for k in ('label','request_id','condition_id','stage','parent_id','parent_prefix','noise_index','split')}
        out.update(reference='ORIGINAL',candidate='EDGE_FLIP',contrast='SAME_MODEL_SAME_NOISE_C_INTERNAL_FLIP',evaluation_scope='PAIRED_TWO_PARENTS_ONLY')
        for field in fields:
            av,bv=a.get(field),b.get(field);out[field+'__reference']=av;out[field+'__candidate']=bv
            out[field+'__delta']=bv-av if _numeric(av) and _numeric(bv) else None
        result.append(out)
    if len(result)!=24:raise ValueError('Expected twenty-four feasible representation contrasts')
    return result


def collect():
    if 'torch' in sys.modules:raise RuntimeError('Analysis requires a fresh CPU process')
    completion=DEST/'metrics_complete.json';cached=_cached(completion)
    if cached:return cached
    cells,specs,missing=_all_specs(False)
    if missing:return dict(status='PENDING_ALL_BACKENDS',missing_count=len(missing),first_missing=missing[:12])
    fields=C.read(C.ROOT/'experiments/r2_bench_four_routes_v1/exposure/data/eval_manifest.json')['metric_fields']
    if len(fields)!=177 or len(set(fields))!=177:raise ValueError('Original metric dictionary differs')
    protected={str(p):C.sha(p) for p in (EVAL/'tasks.json',EVAL/'baseline.json',EVAL/'registration.json',Path(__file__).with_name('analyze.py'),
        C.ROOT/'r2/pilots/seam_hinge_quality_v1/topology.py')}
    rows=[];common=[];shapes=[];graphs=[];statuses=[];graph_records={}
    train_reg=C.read(OUT/'training_registration.json');training_parents={r['parent_id'] for r in C.read(train_reg['manifest'])['train']}
    protected[str(OUT/'training_registration.json')]=C.sha(OUT/'training_registration.json')
    protected[str(C.resolve(train_reg['manifest']))]=C.sha(train_reg['manifest'])
    for s in specs:
        _authenticate(s,protected)
        cell,stage=s['cell'],s['stage'];q=cell['q'];row=C.read(s['score']/'row.json');score=C.read(s['score']/'score.json');shape=C.read(s['score']/'shape_per_face.json')
        for key in ('request_id','condition_id','parent_id','noise_index','N','K','split'):
            if row[key]!=q[key]:raise ValueError('Saved score identity differs: '+key)
        if score['status']!='PASS' or score['C_query_count']!=2048 or not score['C_preserved_exact'] or not score['C_original_order_preserved']:
            raise ValueError('Saved scorer did not pass input-preservation/query checks')
        mesh=C.arrays(s['mesh']);actual_free=len(mesh['output'])-int(q['K'])
        if int(row['free_face_denominator'])!=actual_free:raise ValueError('Actual free denominator differs')
        if any(key not in row for key in fields):raise ValueError('Missing original metric field')
        meta=_metadata(cell,stage);meta['parent_in_current_training_pool']=q['parent_id'] in training_parents
        full=dict(meta,source_stage_field=row.get('stage'),source_label_field=row.get('label'),score_path=str(s['score']),mesh_path=str(s['mesh']),metric_scope='ACTUAL_NATIVE_C')
        full.update({key:row[key] for key in fields});rows.append(full)
        common.append(dict(_common(cell,stage,score,s['common']),parent_in_current_training_pool=meta['parent_in_current_training_pool']))
        adapted=dict(cell,origin=cell.get('baseline_origin',cell.get('origin')))
        sh=_shape_row(adapted,shape,stage=stage);sh.update(meta);shapes.append(sh)
        graph,records=_graphs(s,shape,row);graphs.append(graph);graph_records[cell['cell_id']+'__'+stage]=records
        statuses.append(dict(meta,source_version=score['version'],source_status=score['status'],status_by_metric=score['status_by_metric'],
            C_preserved_exact=score['C_preserved_exact'],C_original_order_preserved=score['C_original_order_preserved'],
            native_C_query_sha256=q['query_descriptor']['sha256'],required_interface='NA_NOT_REGISTERED',source_reconstruction='NA_NOT_UNIQUE_GENERATIVE_TARGET'))
    paths=[]
    def save(name,values):
        path=DEST/name;C.csvwrite(path,values);paths.append(path)
    def tables(prefix,values,names):
        save(prefix+'_per_output.csv',values)
        groups=('label','stage','representation','evaluation_scope','parent_id','parent_prefix','split')
        parents=_aggregate(values,groups,names);save(prefix+'_per_parent.csv',parents)
        summary=_aggregate(parents,('label','stage','representation','evaluation_scope','split'),names)
        for row in summary:row['aggregation']='MACRO_MEAN_OF_PARENT_MEANS';row['parent_count']=row.pop('outputs')
        save(prefix+'_parent_macro_summary.csv',summary)
        model_pairs=_model_pairs(values,names);save(prefix+'_model_paired.csv',model_pairs)
        deltas=[name+'__delta' for name in names]
        save(prefix+'_model_paired_per_parent.csv',_aggregate(model_pairs,('reference','candidate','stage','representation','evaluation_scope','parent_id','parent_prefix','split'),deltas))
        representation_pairs=_representation_pairs(values,names);save(prefix+'_representation_paired.csv',representation_pairs)
        save(prefix+'_representation_paired_per_parent.csv',_aggregate(representation_pairs,('label','stage','evaluation_scope','parent_id','parent_prefix','split'),deltas))
    tables('native_full177',rows,fields)
    common_fields=list(COMMON_METRICS)+[key for key in common[0] if key.startswith('boundary__')]
    tables('common_C',common,common_fields)
    shape_fields=[key for key in shapes[0] if key.startswith('free_') and key not in ('free_count',)]
    tables('cached_shape',shapes,shape_fields)
    graph_fields=[key for key in graphs[0] if key.startswith('derived_')]
    tables('cached_index_graph',graphs,graph_fields)
    by={(r['cell_id'],r['stage']):r for r in rows};stage_rows=[]
    for cell in cells:
        raw,final=by[cell['cell_id'],'RAW'],by[cell['cell_id'],'CLEAN_FINAL_POST'];out=_metadata(cell,'RAW_TO_CLEAN_FINAL_POST')
        for name in ('free_face_denominator','free_total_surface_area','whole_total_surface_area','deleted_free_faces','new_bridge_faces',
                     'free_minimum_angle_lt5_fraction','free_angle_p05_degrees','free_near_degenerate_fraction','free_exact_zero_area_fraction',
                     'hinge_diag_isolated_1face_edge_components','hinge_diag_isolated_1face_unfiltered_edge_components'):
            a,b=raw.get(name),final.get(name);out[name+'__RAW']=a;out[name+'__FINAL']=b
            out[name+'__delta']=b-a if _numeric(a) and _numeric(b) else None
        for name in ('free_face_denominator','free_total_surface_area','whole_total_surface_area'):
            a,b=raw.get(name),final.get(name);out[name+'__FINAL_over_RAW']=b/a if _numeric(a) and _numeric(b) and a>0 else None
        out['deletion_semantics']='Original scorer missing RAW parent slots; output count and new bridge count remain separate.'
        stage_rows.append(out)
    save('RAW_FINAL_retention_and_costs.csv',stage_rows)
    C.write(DEST/'metric_status_per_output.json',statuses);paths.append(DEST/'metric_status_per_output.json')
    C.write(DEST/'cached_component_records.json',dict(version=GRAPH_VERSION,records=graph_records));paths.append(DEST/'cached_component_records.json')
    _verify(protected)
    result=dict(status='COMPLETE',version=VERSION,outputs=30,stage_rows=60,native_fields=177,common_C_rows=60,
        model_pair_rows=60,representation_pair_rows=24,original_parents=3,feasible_flip_parents=2,
        blocked_f495_flip='BLOCKED_INPUT_GEOMETRY_NO_LEGAL_FLIP; not generated, not a failed output',
        stages=list(STAGES),source_statuses_preserved=True,scoring_calls=0,model_calls=0,new_queries=0,new_geometry_detector_calls=0,
        stage_label_adapter='Logical RAW/CLEAN_FINAL_POST stage is explicit; historical scorer method labels remain in source_stage_field/source_label_field.',
        graph_scope='Separate cached-graph reduction version. RAW index components are not visual fragments; legal and unfiltered graphs stay distinct.',
        common_scope='Original physical C CF/I only; actual-C topology and preservation remain native177.',
        evaluation_scope='Development panel with original historical splits and explicit current training-parent membership; no broad generalization claim.',
        source_sha256=C.sha(__file__),protected=protected,files={str(p):C.sha(p) for p in paths})
    C.write(completion,result);return result


def losses():
    """The exact 200 logged updates per arm; no intermediate checkpoint load."""
    completion=DEST/'training_curves_complete.json';cached=_cached(completion)
    if cached:return cached
    rows=[];protected={str(OUT/'training_registration.json'):C.sha(OUT/'training_registration.json')}
    for arm in ('PLAIN200','FLIP200'):
        folder=OUT/'training'/arm;status_path=folder/'status.json';log_path=folder/'training_log.jsonl'
        if not status_path.is_file() or C.read(status_path).get('status')!='COMPLETE':
            return dict(status='PENDING_TRAINING_COMPLETE',arm=arm)
        status=C.read(status_path)
        if status['stage_step']!=200 or status['global_step']!=25700:
            raise ValueError('Only the registered local200/global25700 endpoints may be summarized')
        saved=[json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(saved)!=200 or [row['stage_step'] for row in saved]!=list(range(1,201)) or any(row['arm']!=arm for row in saved):
            raise ValueError('Training log is not exactly the 200 completed authorized updates')
        for row in saved:
            if row['global_step']!=25500+row['stage_step'] or not all(_numeric(row[key]) for key in ('FM','objective','unweighted_FM')):
                raise ValueError('Malformed training loss row')
        rows.extend(saved)
        protected[str(status_path)]=C.sha(status_path);protected[str(log_path)]=C.sha(log_path)
    a,b=rows[:200],rows[200:]
    for aa,bb in zip(a,b):
        if aa['free_coordinate_denominator']!=bb['free_coordinate_denominator']:
            raise ValueError('Paired training coordinate denominator differs')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    DEST.mkdir(parents=True,exist_ok=True)
    png=DEST/'training_loss_curves.png'
    if png.exists():raise RuntimeError('Unsealed loss-curve image; explicit recovery required')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),dpi=150,facecolor='white')
    try:
        for values,color,label in ((a,'#315b91','PLAIN200'),(b,'#bf6335','FLIP200')):
            steps=np.asarray([r['stage_step'] for r in values])
            for ax,key,title in zip(axes,('objective','unweighted_FM'),('Registered weighted training objective','Unweighted FM over fixed free-coordinate denominator')):
                y=np.asarray([r[key] for r in values],float)
                mean=np.asarray([y[max(0,i-9):i+1].mean() for i in range(len(y))])
                ax.plot(steps,y,color=color,alpha=.18,lw=.6)
                ax.plot(steps,mean,color=color,lw=1.7,label=label+' trailing 10 updates')
                ax.set_title(title,fontsize=10);ax.set_xlabel('Local optimizer update');ax.set_ylabel(key);ax.grid(alpha=.2)
                ax.set_xlim(1,200)
        axes[0].legend(fontsize=8)
        fig.suptitle('Fixed 200-update same-stream comparison | endpoint is global25700\nTraining loss is not a completion-quality or generalization score',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.90));buffer=io.BytesIO();fig.savefig(buffer,format='png')
        with png.open('xb') as handle:handle.write(buffer.getvalue())
    finally:plt.close(fig)
    csvpath=DEST/'training_loss_per_update.csv';C.csvwrite(csvpath,rows)
    snapshots=DEST/'training_loss_original_records.json';C.write(snapshots,rows)
    pairs=[]
    for aa,bb in zip(a,b):
        result=dict(stage_step=aa['stage_step'],global_step=aa['global_step'],free_coordinate_denominator=aa['free_coordinate_denominator'])
        for key in ('objective','unweighted_FM','FM','generator_gradient_norm','all_free_velocity_sse','all_free_endpoint_sse'):
            result[key+'__PLAIN200']=aa[key];result[key+'__FLIP200']=bb[key];result[key+'__delta']=bb[key]-aa[key]
        pairs.append(result)
    paired=DEST/'training_loss_paired_steps.csv';C.csvwrite(paired,pairs)
    _verify(protected)
    result=dict(status='COMPLETE',version=VERSION,arms=2,updates_per_arm=200,rows=400,
        smoothing='Trailing arithmetic mean of at most10 actual updates; raw values shown faintly and fully saved',
        model_calls=0,checkpoint_loads=0,source_sha256=C.sha(__file__),protected=protected,
        files={str(p):C.sha(p) for p in (png,csvpath,snapshots,paired)})
    C.write(completion,result);return result


def _pages():
    cells,specs,_=_all_specs();by={(s['cell']['label'],s['cell']['representation'],s['cell']['q']['request_id'],s['stage']):s for s in specs}
    pages=[]
    for representation,rid in sorted({(c['representation'],c['q']['request_id']) for c in cells}):
        for stage in STAGES:
            ss=[by[label,representation,rid,stage] for label in LABELS]
            q=ss[0]['cell']['original_q']
            for s in ss:
                if s['cell']['original_q']['gallery_bounds']!=q['gallery_bounds'] or s['cell']['original_q']['gallery_depth_bounds']!=q['gallery_depth_bounds']:
                    raise ValueError('Original fixed frame differs between models')
            pages.append(dict(page_id=stage+'__'+representation+'__'+rid,stage=stage,representation=representation,
                request_id=rid,reference_q=q,cells=[dict(label=s['cell']['label'],q=s['cell']['q'],
                    path=str(s['mesh']),producer=str(s['producer']),cell_id=s['cell']['cell_id']) for s in ss]))
    if len(pages)!=20:raise ValueError('Expected twenty fixed three-model pages')
    return pages,specs


def _render_page(page):
    """Existing raster/section definitions, with the true stage in every title."""
    if 'torch' in sys.modules:raise RuntimeError('Render in a fresh CPU process')
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    destination=GALLERY/page['page_id'];png=destination/'comparison.png';receipt=destination/'comparison.json'
    q=page['reference_q'];stage=page['stage'];bounds=q['gallery_bounds'];depth=q['gallery_depth_bounds']
    identity=dict(version=VERSION,stage=stage,representation=page['representation'],page_id=page['page_id'],
        original_reference_request=q['request_id'],bounds=bounds,depth_bounds=depth,
        cells=[dict(label=c['label'],cell_id=c['cell_id'],path=c['path'],mesh_sha256=C.sha(c['path']),
            input_sha256=C.sha(c['q']['inference_npz']),K=int(c['q']['K']),N_requested=int(c['q']['N'])) for c in page['cells']],
        source_sha256=C.sha(__file__),renderer_sha256=C.sha(G.__file__))
    if receipt.exists():
        saved=C.read(receipt)
        if saved['identity']!=identity or C.sha(png)!=saved['image_sha256']:
            raise ValueError('Completed page identity changed')
        return saved
    if png.exists():raise RuntimeError('Unsealed page image; no silent overwrite')
    destination.mkdir(parents=True,exist_ok=True);tick=time.perf_counter();plt=G.H._mpl()
    fig=plt.figure(figsize=(15.9,13.5),dpi=110,facecolor='white')
    grid=fig.add_gridspec(4,3,left=.025,right=.99,bottom=.055,top=.89,wspace=.15,hspace=.23)
    display=[]
    try:
        for col,cell in enumerate(page['cells']):
            cq=cell['q'];k=int(cq['K']);arr=C.arrays(cell['path']);tri,roles=G.roles(arr,k)
            inp=C.arrays(cq['inference_npz']);original=inp['C_vertices'][inp['C_faces_original']]
            if not np.array_equal(tri[:k],original):raise ValueError('Rendered C differs from actual input')
            geometry=dict(C=tri[:k],F=tri[k:],C_original=original,tri=tri)
            facts=dict(label=cell['label'],stage=stage,K=k,N_requested=int(cq['N']),N_output=len(tri),
                role_counts={r:int((roles==r).sum()) for r in ('C','SEAM','BULK','BRIDGE')},views=[])
            for row,azimuth in enumerate((-60,120)):
                raster,scope=G.role_raster(tri,roles,k,G.D.basis(azimuth,23),np.asarray(depth['center']),depth['span'])
                ax=fig.add_subplot(grid[row,col]);ax.imshow(raster);ax.set_axis_off()
                ax.set_title(cell['label']+f' | {stage} | az={azimuth}\nK={k}, requestedN={cq["N"]}, outputN={len(tri)}',fontsize=8)
                facts['views'].append(dict(azimuth=azimuth,**scope))
                if scope['projected_vertices_outside_inherited_frame']:
                    ax.text(.01,.01,'Inherited-frame clipping; see receipt',transform=ax.transAxes,fontsize=7,color='#b34b27')
            for row,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[row,col]);seams=np.flatnonzero(roles[k:]=='SEAM')
                G.H.draw_section(ax,geometry,seams,bounds,C.L,view,1)
                position=bounds['front_positions' if view=='front' else 'side_positions'][1]
                from matplotlib.collections import LineCollection
                axis,project=(2,(0,1)) if view=='front' else (0,(2,1))
                sections,ids=G.H.finite_sections(geometry['F'],axis,position,1e-12*C.L)
                bridge=roles[k:][ids]=='BRIDGE'
                if bridge.any():ax.add_collection(LineCollection(sections[bridge][:,:,project],colors='#41a46c',linewidths=1.6,zorder=3))
                ax.set_title(cell['label']+' | '+stage+' | '+view+f'={position:.5f}',fontsize=7)
            display.append(facts)
        fig.suptitle(page['request_id']+' | '+page['representation']+' | '+stage+'\n'
            'EXPOSURE / PLAIN200 / FLIP200 | Same saved noise, original frame and finite sections\n'
            'Blue: condition C | Gray: generated bulk | Green: added bridge | '+q['split'],fontsize=11)
        fig.text(.5,.018,'Actual AI_PROXY review is separate. Index components or section endpoints alone do not certify visible fragments or cracks. No unique-source-style grading.',ha='center',fontsize=8)
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110)
        with png.open('xb') as handle:handle.write(buffer.getvalue())
    finally:plt.close(fig)
    result=dict(status='PENDING_ACTUAL_VIEW',identity=identity,page_id=page['page_id'],stage=stage,
        image_path=str(png),image_sha256=C.sha(png),display=display,seconds=time.perf_counter()-tick,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,scoring_calls=0,ratings_generated=0)
    C.write(receipt,result);return result


def render(workers=1):
    completion=GALLERY/'render_complete.json';cached=_cached(completion)
    if cached:return cached
    if workers not in (1,2):raise ValueError('Controller must select one or two resource-measured CPU workers')
    pages,specs=_pages();protected={str(p):C.sha(p) for p in (EVAL/'tasks.json',EVAL/'baseline.json',EVAL/'registration.json',
        C.ROOT/'r2/pilots/chair_multipart_seam_v1/gallery.py',C.ROOT/'r2/pilots/regular_patch_control_v1/depth_gallery.py',
        C.ROOT/'r2/tools/cref_layer_audit_v1/render.py')}
    for s in specs:_authenticate(s,protected)
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    dispatch=GALLERY/'dispatch'/str(time.time_ns());before=snapshot()
    C.write(dispatch/'before.json',dict(resources=before,workers=workers,threads_per_worker=1,pages=20,
        reason='Controller-selected CPU concurrency; same existing raster/section renderer; no GPU/model task'))
    definition=dict(version=VERSION,pages=pages,source_sha256=C.sha(__file__),protected=protected,
        camera='Inherited azimuth -60/120, elevation23, original center/span; no output-dependent fitting',
        section='Inherited front/side central finite plane positions, fixed display bounds',
        stages=list(STAGES),blocked='f495 has no EDGE_FLIP page because the input geometry gate blocked it')
    C.write(GALLERY/'definition.json',definition);start=time.perf_counter();results=[]
    if workers==1:
        for page in pages:
            result=_render_page(page);results.append(result)
            print(json.dumps(dict(page_id=result['page_id'],seconds=result['seconds'])),flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in pool.map(_render_page,pages):
                results.append(result);print(json.dumps(dict(page_id=result['page_id'],seconds=result['seconds'])),flush=True)
    records=sorted(results,key=lambda r:r['page_id'])
    blocks=['<!doctype html><meta charset="utf-8"><title>Bench C edge-flip candidates</title>',
        '<h1>EXPOSURE / PLAIN200 / FLIP200</h1><p>20 registered RAW/FINAL pages. Same physical input, saved noise, original cameras and sections. Final model epochs are fixed at local200. f495 has no legal flipped input. Actual AI_PROXY review is separate.</p>']
    for row in records:
        relative=row['page_id']+'/comparison.png'
        blocks.append('<h2>'+html.escape(row['page_id'])+'</h2><a href="'+html.escape(relative,quote=True)+'"><img style="max-width:100%" src="'+html.escape(relative,quote=True)+'"></a>')
    index=GALLERY/'index.html';content='\n'.join(blocks)
    if index.exists():
        if index.read_text(encoding='utf-8')!=content:raise ValueError('Frozen gallery HTML differs')
    else:
        with index.open('x',encoding='utf-8') as handle:handle.write(content)
    _verify(protected)
    C.write(dispatch/'complete.json',dict(status='COMPLETE',workers=workers,seconds=time.perf_counter()-start,resources_after=snapshot(),pages=20))
    artifacts=[GALLERY/'definition.json',index]+[Path(r['image_path']) for r in records]+[GALLERY/r['page_id']/'comparison.json' for r in records]
    result=dict(status='COMPLETE',review_status='PENDING_ACTUAL_AI_PROXY_REVIEW',pages=20,columns_per_page=3,
        stage_pages={'RAW':10,'CLEAN_FINAL_POST':10},workers=workers,seconds=time.perf_counter()-start,
        model_calls=0,scoring_calls=0,new_queries=0,ratings_generated=0,
        source_sha256=C.sha(__file__),protected=protected,files={str(p):C.sha(p) for p in artifacts})
    C.write(completion,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('collect','losses','render','all'))
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();results={}
    if args.action in ('collect','all'):results['metrics']=collect()
    if args.action in ('losses','all'):results['losses']=losses()
    if args.action in ('render','all'):results['render']=render(args.workers)
    print(json.dumps({k:{key:value for key,value in result.items() if key not in ('files','protected')} for k,result in results.items()},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
