"""Read-only extraction of frozen 32-vs-67 evidence. No model/geometry imports.
Run with Python stdlib only. Old OUT and ZIP are read-only. Output is MINI only.
CSV source cells are retained as strings, without float rounding or recoding.
Frozen scientific source snapshots are evidence, not executed by this program.
"""
from __future__ import annotations
import argparse,ast,collections,csv,hashlib,io,json,math,os
from pathlib import Path
import zipfile

DEFAULT_OLD=Path('/mnt/e/MeshFlow/experiments/r2_user_data_scaling_32_67_v1')
DEFAULT_OUT=Path('/mnt/e/MeshFlow/deliverables/R2_32_67_ANALYSIS_MINI_V1')
SHORT={'USER_SMALL':'SMALL','USER_EXPANDED_67':'EXP','HIST_USER_MIX_FROZEN':'HIST'}
KEY=('request_id','label','stage')
MISSING=[]
SOURCES={}
PRODUCTS={}
FROZEN_CODE={
 'code/new/scoring.py':'original/scoring.py',
 'code/new/report_results.py':'original/report_results.py',
 'code/new/evaluation.py':'original/evaluation.py',
 'code/new/reuse_data.py':'original/reuse_data.py',
 'code/dependencies/external/c01ac642573d8fe1/geometry.py':'original/scorers/geometry.py',
 'code/dependencies/external/c88b4946f12757f6/control.py':'original/scorers/control.py',
 'code/dependencies/external/4ba73455b21168fd/shape.py':'original/scorers/shape.py',
 'code/dependencies/external/5974fece95c385b0/pair_audit.py':'original/scorers/pair_audit.py',
 'code/dependencies/external/4a4b754b98c0b110/intersections.py':'original/scorers/intersections.py',
 'code/dependencies/external/910d7f21f733ef0b/seams.py':'original/scorers/seams.py',
 'code/dependencies/external/1bab2c227601f033/surface.py':'original/scorers/surface.py',
 'code/dependencies/r2/tools/avmr_pmr_v1/run.py':'original/avmr_pmr.py',
 'code/dependencies/r2/pilots/anchor_aware_denoiser_v1/evaluation.py':'original/valid_graph.py',
 'code/dependencies/r2/tools/c_restore_coupling_v1/evaluation.py':'original/opposite_reach.py',
 'code/dependencies/r2/src/meshflow_control/source_winding.py':'original/source_winding.py',
 'code/dependencies/r2/src/meshflow_control/winding_solver.py':'original/winding_solver.py',
 'inherited/code_snapshot/quality_scan.py':'original/source_quality_scan.py',
 'inherited/code_snapshot/data_freeze.py':'original/source_data_freeze.py',
 'code/inherited_data_dependencies/r2/data_tools/user_conditions_v1/quality.py':'original/source_quality_summary.py',
}
def sha(b):return hashlib.sha256(b).hexdigest()
def compact(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def key(r):return tuple(r[k] for k in KEY)
def scalar(x):return not isinstance(x,(dict,list))
def emit(path,data,rows=None,fields=None,sources=(),kind='PROJECTION',note=''):
    path=OUT/path
    path.resolve().relative_to(OUT.resolve())
    if path.exists():
        if path.read_bytes()!=data:raise RuntimeError('Refuse overwrite differing MINI file: '+str(path))
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    PRODUCTS[str(path.relative_to(OUT))]=dict(sha256=sha(data),bytes=len(data),rows=rows,fields=fields,
        source_refs=list(sources),transformation=kind,DERIVED=kind in ('DERIVED','DICTIONARY_DOCUMENTATION'),note=note)
def emitjson(name,value,**kw):emit(name,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode(),**kw)
def csvbytes(rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    s=io.StringIO(newline='');w=csv.DictWriter(s,fields);w.writeheader()
    for r in rows:w.writerow({k:compact(v)if isinstance(v,(dict,list))else v for k,v in r.items()})
    return s.getvalue().encode('utf-8-sig'),fields
def emitcsv(name,rows,**kw):
    b,f=csvbytes(rows);emit(name,b,rows=len(rows),fields=f,**kw)
def source_bytes(path):
    path=Path(path);ref=str(path)
    b=path.read_bytes();prior=BY_SOURCE.get(ref)
    h=sha(b)
    if prior and prior['sha256']!=h:raise ValueError('Frozen source changed: '+ref)
    SOURCES[ref]=dict(source_path=ref,existing_sha256=prior['sha256']if prior else None,
        existing_hash_source='old delivery PACKAGE_MANIFEST.json'if prior else 'NOT_IN_OLD_PACKAGE_MANIFEST',
        read_sha256=h,bytes=len(b),read_now=True,kind='FILE')
    return b
def source_json(rel):return json.loads(source_bytes(OLD/rel).decode('utf-8-sig'))
def source_csv(rel):
    b=source_bytes(OLD/rel);rr=list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
    return rr
def zip_member(name):
    b=ZIP.read(name);prior=BY_ARCHIVE.get(name);h=sha(b)
    if not prior or prior['sha256']!=h:raise ValueError('Missing/mismatched frozen ZIP source: '+name)
    ref='ZIP_MEMBER:'+name
    SOURCES[ref]=dict(source_path=prior['source_path'],zip_path=str(ZIP_PATH),zip_member=name,
        existing_sha256=prior['sha256'],existing_hash_source='old delivery PACKAGE_MANIFEST.json',
        read_sha256=h,bytes=len(b),read_now=True,kind='ZIP_MEMBER')
    return b,ref
def missing(field,reason,scope,status='NOT_MEASURED_OR_NOT_RECORDED'):
    r=dict(field=field,reason=reason,scope=scope,status=status)
    if r not in MISSING:MISSING.append(r)

def derive_source_parent_groups(rows):
    """DERIVED MINI helper, not a new historical worst-risk labeling rule.
    Original scoring.aggregate stratifies each condition's source_quality, then
    averages condition/noise values within each parent in that stratum.
    A parent with multiple labels belongs to multiple original strata. Only a
    unanimous label is emitted as parent_label; otherwise MIXED is explicit.
    """
    groups=collections.defaultdict(dict)
    for r in rows:
        groups[r['parent_id']][r['condition_id']]=r.get('source_quality','UNKNOWN')
    result=[]
    for parent,conds in sorted(groups.items()):
        labels=sorted(set(conds.values()))
        result.append(dict(parent_id=parent,parent_label=labels[0]if len(labels)==1 else 'MIXED',
            original_stratum_memberships=labels,condition_labels=conds,
            mutually_exclusive_parent_label=len(labels)==1,DERIVED=True))
    return result

def numerical_dictionary(names,directions,settings):
    units={};definitions={}
    def setdef(n,desc,unit='count',denom='none',domain='final free F',origin='INHERITED_ORIGINAL_SCORER',ref='original/scoring.py::score_one'):
        definitions[n]=dict(description=desc,unit=unit,denominator=denom,domain=domain,
            origin=origin,DERIVED_THIS_EXTRACTION=False,definition_ref=ref,
            missing_rule='Original CSV empty cell represents JSON null; never zero-fill. Consult score/status columns and original function for applicable empty/unknown cases.')
    for n in names:
        if n.startswith(('A_VMR__','PMR__')):
            part,setting=n.split('__',1);spec=settings[setting]
            setdef(n,'matched boundary vertices / all registered C open-boundary vertices'if part=='A_VMR'else 'Per-output indicator: all registered C open-boundary vertices match free vertices',
                'fraction_0_1'if part=='A_VMR'else 'indicator_0_1','boundary_vertex_count'if part=='A_VMR'else 'one output; aggregate over outputs via noise-condition-parent hierarchy',
                'ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY; F-only nearest vertex; strict distance < threshold',
                'INHERITED_ORIGINAL_MATCHER_EXISTING_WIDE_ALIAS','original/avmr_pmr.py::topology, nearest, metric')
            definitions[n]['threshold']=spec;continue
        if n.startswith(('CF_','FF_')):
            side=n[:2];kind=n[3:];ref='original/scorers/geometry.py::scan_geometry'
            if kind=='illegal_affected_free_fraction':
                setdef(n,'Fraction of F faces incident to >=1 proper or positive-area-coplanar illegal '+side+' pair','fraction_0_1','free_face_denominator',origin='EXISTING_ADAPTER_UNION'if side=='CF'else 'INHERITED_ORIGINAL_SCORER',ref='original/scoring.py::score_one'if side=='CF'else ref)
            elif kind=='proper_affected_free_fraction':setdef(n,'Fraction of F faces incident to >=1 proper transverse '+side+' pair','fraction_0_1','free_face_denominator',ref=ref)
            else:setdef(n,{'proper_pair_count':'Number of proper transverse penetration pairs','coplanar_pair_count':'Number of positive-area coplanar overlap pairs','illegal_pair_count':'Number of proper OR positive-area coplanar illegal pairs','unresolved_or_degenerate_pair_count':'Number of unresolved/degenerate pairs, not silently classified legal'}[kind],
                domain=side+' pairs',ref=ref)
            continue
        if n.startswith('ConflictCov_C'):
            suffix=n[len('ConflictCov_C'):]
            descriptions={'':'Weighted C-query fraction in cross OR dup (lower confirmed coverage)',
            '_upper':'Weighted C-query fraction in cross OR dup OR uncertain coverage',
            '_cross':'Weighted fraction within rho_cross*L of proper owner-C intersection segments',
            '_dup':'Weighted fraction in C interior with a free finite surface within rho_dup*L and unsigned normal angle <=15 degrees',
            '_cross_and_dup':'Weighted fraction in both cross and dup',
            '_uncertain_fraction':'Weighted unknown-only C-query coverage, excluding confirmed union',
            '_interior_weight_fraction':'Weighted C-query fraction at least 0.01L from C boundary',
            '_interior_samples':'Count of fixed C queries in the registered interior',
            '_unresolved_pairs':'Trusted CF code==3 pair count',
            '_coplanar_illegal_pairs':'Trusted CF code==2 positive-area coplanar pair count',
            '_proper_pairs':'Trusted CF code==1 proper pair count',
            '_segment_extraction_failures':'Proper intersection segment extraction failure count'}
            counted=suffix in ('_interior_samples','_unresolved_pairs','_coplanar_illegal_pairs','_proper_pairs','_segment_extraction_failures')
            setdef(n,descriptions[suffix],'count'if counted else 'fraction_0_1','none'if counted else 'C_query_weight_denominator','frozen 2048 area-weighted C queries; canonical owner metadata; same source-C geometric point set',ref='original/scorers/control.py::conflict_metrics')
            definitions[n]['missing_rule']='Original ConflictCov_C_status retained: PARTIAL for unknown-only coverage or extraction failures; LOW_SUPPORT for too few interior queries; PASS otherwise. Numerical value remains original lower/upper, not zero-filled.'
            continue
        if n.startswith('EXACT_COORD_'):
            what=n[len('EXACT_COORD_'):]
            desc={'vertices':'Number of exact FP32-bit coordinate-index vertices','edges':'Undirected indexed edges','boundary_edges':'Edges with one incident face','nonmanifold_edges':'Edges with >2 incident faces','same_direction_two_face_edges':'Exactly-two-incidence edges whose two original face traversals have equal direction','face_components':'Connected face components via any shared indexed edge; legality not required','CF_shared_edges':'Edges with >=1 C and >=1 F incident face; legality not required','closed_two_faces_per_edge':'Nonempty edge set and every edge has exactly two incidences; not a watertightness/manifold certificate'}[what]
            setdef(n,desc,'indicator_boolean'if what=='closed_two_faces_per_edge'else 'count',domain='Exact FP32-bit coordinate indexing of canonical-prefix triangles, before the independent source-winding graph diagnostic',ref='original/scorers/geometry.py::topology');continue
        if n.startswith('RAW_SOUP_'):
            setdef(n,'All triangles remain independently indexed: number of faces'if n.endswith('face_components')else 'Zero by independent triangle-soup construction, not measured attachment success',domain='descriptive triangle-soup view, also reported for final POST triangle coordinates',ref='original/scorers/geometry.py::topology');continue
        if n.startswith('boundary_distance_'):
            setdef(n,{'boundary_distance_mean_model':'Mean','boundary_distance_median_model':'Median','boundary_distance_p95_model':'np.quantile at 0.95','boundary_distance_max_model':'Maximum'}[n]+' of nearest F-vertex Euclidean distances over source-C open-boundary vertices','model_coordinate_length','boundary_vertex_count','same boundary domain as A-VMR',ref='original/scoring.py::score_one')
            definitions[n]['missing_rule']='NULL when empty boundary or nonfinite distances; boundary_distance_status distinguishes NA_EMPTY_BOUNDARY and POSITIVE_INFINITY_EMPTY_FREE.';continue
        if n.startswith('C_F_') or n in ('graph_valid_edges','reached_free_faces','reached_free_area','free_area','reached_free_area_fraction','exact_duplicate_faces','nonmanifold_edges','boundary_edges') or n.startswith('opposite_winding_reached_'):
            d={'C_F_has_valid_edge':'Indicator of >=1 valid C/F indexed edge','C_F_has_opposite_winding_edge':'Indicator of >=1 valid opposite-winding C/F indexed edge',
               'C_F_shared_edge_count':'All indexed edges incident to C and F, including rejected edges','C_F_shared_edge_total_length':'Total length of all indexed C/F shared edges',
               'C_F_valid_edge_count':'Eligible legal C/F indexed edge count','C_F_valid_edge_total_length':'Total length of eligible legal C/F indexed edges',
               'C_F_valid_opposite_winding_edge_count':'Valid C/F edges traversed in opposite directions by their two incident faces',
               'C_F_nonmanifold_edge_count':'C/F shared edges with >2 incidences','C_F_rejected_duplicate_edge_count':'C/F edges rejected because either incident face is in an indexed duplicate-face group',
               'C_F_rejected_illegal_edge_count':'C/F edges rejected for an illegal saved pair classification','C_F_rejected_unknown_edge_count':'C/F edges rejected for unknown or non-contact saved pair classification',
               'graph_valid_edges':'All valid graph edges incident to any F face, including F/F',
               'reached_free_faces':'Number of F faces reachable from all C nodes over legal graph edges',
               'reached_free_area':'Sum of reachable F triangle areas','free_area':'Total final F triangle area',
               'reached_free_area_fraction':'Reachable F area divided by total F area',
               'opposite_winding_reached_free_faces':'F faces reachable using only valid edges with opposite winding',
               'opposite_winding_reached_free_area':'Area of F faces reachable using only valid opposite-winding edges',
               'opposite_winding_reached_free_area_fraction':'Opposite-winding reachable F area divided by total F area',
               'exact_duplicate_faces':'Count of all faces in indexed duplicate-face groups, not excess-only count',
               'nonmanifold_edges':'Indexed edges with more than two incidences across C+F','boundary_edges':'Indexed edges with one incidence across C+F'}
            unit='indicator_0_1'if '_has_'in n else 'fraction_0_1'if n.endswith('fraction')else 'model_coordinate_length'if 'length'in n else 'model_coordinate_area'if 'area'in n else 'count'
            setdef(n,d[n],unit,'free_area'if n.endswith('fraction')else 'none','RAW: source-C restored, fixed-C oriented exact-coordinate index; POST: actual exported index with same source restoration/orientation',
                ref='original/opposite_reach.py::opposite_reach'if n.startswith('opposite_')else 'original/valid_graph.py::actual_valid_graph');continue
        if n=='C_preserved_exact':setdef(n,'C prefix exactly equals frozen canonical C in RAW or original-source C in POST','indicator_boolean',domain='saved output coordinate/order contract',origin='EXISTING_ADAPTER_HARD_CONTRACT');continue
        if n in ('actual_free_faces','deleted_free_faces'):
            setdef(n,'Number of F triangles actually scored'if n=='actual_free_faces'else 'Registered original N minus final C+F triangle count','count',domain='final current output; deletion count is descriptive',origin='EXISTING_ADAPTER_FIELD');continue
        if n=='free_exact_duplicate_excess_fraction':
            setdef(n,'(number of F faces minus unique unordered exact-coordinate F triangles)/number of F faces','fraction_0_1','free_face_denominator',ref='original/scorers/geometry.py::topology');continue
        prefix=next((x for x in ('free_','whole_','C_')if n.startswith(x)),None)
        if prefix:
            dom={'free_':'F','whole_':'C+F','C_':'C'}[prefix];tail=n[len(prefix):]
            desc={'minimum_angle_lt5_fraction':'Fraction of triangles whose minimum angle is <5 degrees','q_gt10_fraction':'Fraction with actual q>10, where q=sum(edge lengths squared)/(4 sqrt(3)*area)',
            'near_degenerate_count':'Number of triangles with normalized area <=1e-12','near_degenerate_fraction':'Fraction of triangles with normalized area <=1e-12',
            'exact_zero_area_count':'Number of triangles with exactly zero FP64 normalized area','exact_zero_area_fraction':'Fraction of triangles with exactly zero FP64 normalized area',
            'total_surface_area':'Total triangle area in original model-coordinate units','angle_p05_degrees':'Fifth percentile of all corner angles, not the fifth percentile of per-face minimum angles',
            'q_mean':'Mean actual q over finite q values only; nonfinite values have separate source per-face status'}[tail]
            unit='fraction_0_1'if tail.endswith('fraction')else 'model_coordinate_area'if 'surface_area'in tail else 'degrees'if 'degrees'in tail else 'dimensionless_q'if tail=='q_mean'else 'count'
            setdef(n,desc,unit,'free_face_denominator'if unit=='fraction_0_1'and dom=='F'else 'number of '+dom+' triangles'if unit=='fraction_0_1'else 'none',dom,ref='original/scorers/shape.py::shape_quality');continue
        raise ValueError('Unspecified registered metric '+n)
    for n in names:
        definitions[n]['direction']=directions[n]['direction']
        definitions[n]['direction_interpretation']=directions[n]['interpretation']
    return definitions

def main():
    global OLD,OUT,ZIP,ZIP_PATH,BY_SOURCE,BY_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--old',type=Path,default=DEFAULT_OLD);parser.add_argument('--out',type=Path,default=DEFAULT_OUT);args=parser.parse_args()
    OLD=args.old.resolve();OUT=args.out.resolve()
    if OUT.name!='R2_32_67_ANALYSIS_MINI_V1' or OLD==OUT or OLD in OUT.parents:raise ValueError('Wrong output root')
    ZIP_PATH=OLD/'delivery/R2_USER_DATA_SCALING_32_67_REVIEW.zip'
    manifest_path=OLD/'delivery/R2_USER_DATA_SCALING_32_67_REVIEW/PACKAGE_MANIFEST.json'
    manifest_bytes=manifest_path.read_bytes();manifest=json.loads(manifest_bytes)
    BY_SOURCE={r['source_path']:r for r in manifest['files']};BY_ARCHIVE={r['archive_path']:r for r in manifest['files']}
    SOURCES[str(manifest_path)]=dict(source_path=str(manifest_path),existing_sha256=None,existing_hash_source='not self-hashed',read_sha256=sha(manifest_bytes),bytes=len(manifest_bytes),read_now=True,kind='FILE')
    ZIP=zipfile.ZipFile(ZIP_PATH)
    registered=source_csv('report_tables/metric_directions.csv');names=[r['metric']for r in registered];directions={r['metric']:r for r in registered}
    if len(names)!=96 or len(set(names))!=96:raise ValueError('Metric registry not 96 unique fields')
    original=source_csv('evaluation/per_output.csv');qrows=source_csv('evaluation/quality_per_output.csv')
    topo=source_csv('evaluation/topology_per_output.csv');vm=source_csv('evaluation/AVMR_PMR_all8.csv')
    reg=source_json('evaluation/registration.json');requests={r['request_id']:r for r in reg['requests']}
    old_inventory_path=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1/parent_inventory.json')
    old_inventory_bytes=source_bytes(old_inventory_path);old_inventory=json.loads(old_inventory_bytes)
    old_parent_ids={r['parent_id']for r in original if r['panel']=='OLD6'}
    old_groups={r['uid']:r['group_id']for r in old_inventory['rows']if r['uid']in old_parent_ids}
    if len(old_groups)!=6 or not all(old_groups.values()):raise ValueError('Missing original OLD6 inventory identity')
    emitjson('protocol_and_code/old6_group_identity_projection.json',dict(projection=True,source_path=str(old_inventory_path),source_read_sha256=sha(old_inventory_bytes),existing_source_hash='NOT_RECORDED',source_fields=['uid','group_id'],rows=[dict(parent_uid=p,group_id=old_groups[p])for p in sorted(old_groups)],note='Inherited original inventory identity; no parent-to-group inference.'),rows=6,sources=[str(old_inventory_path)],kind='PROJECTION')
    complete=source_json('evaluation/score_completion.json')
    score_by_key={}
    for r in complete['receipts']:
        b=source_bytes(r['path']);s=json.loads(b);score_by_key[key(s)]=(s,str(r['path']))
    if len(original)!=840 or len(vm)!=6720 or len({key(r)for r in original})!=len(original):raise ValueError('Incomplete source identity set')
    if any(n not in original[0]for n in names):raise ValueError('Registered scalar missing from original table')
    qq={key(r):r for r in qrows};tt={key(r):r for r in topo}
    metrics=[]
    for r in original:
        k=key(r);s,p=score_by_key[k];q=requests[r['request_id']];out=dict(r)
        out.update(parent_uid=r['parent_id'],group_id=q.get('group_uid')or old_groups.get(r['parent_id'],'NA_NOT_RECORDED'),noise_id=r['candidate'],
            original_model_label=r['label'],model_short=SHORT[r['label']],condition_type=q.get('condition_type',r['region_type']+'_'+r['tier']),
            score_status=s['status'],score_file_path=p,score_record_key=compact(k),score_file_existing_sha256=SOURCES[p]['existing_sha256']or 'NA_NO_PRIOR_HASH',
            score_file_read_sha256=SOURCES[p]['read_sha256'],score_identity=compact(s['identity']),quality_C_orientation=s['quality_C_orientation'])
        out['group_id_source']='evaluation/registration.json::group_uid'if q.get('group_uid')else str(old_inventory_path)+'::rows[uid].group_id'
        out['group_id_status']='INHERITED_EXISTING_IDENTITY'
        for f in ('free_face_denominator','C_query_weight_denominator','I_status'):out[f]=qq[k][f]
        for f in ('C_query_count','CF_tested_pair_denominator','FF_tested_pair_denominator'):out[f]=s[f]
        out.update(topology_domain=tt[k]['domain'],topology_status=tt[k]['status'])
        for f,v in s['status_by_metric'].items():out['metric_status__'+f]=v
        for f,v in s['boundary_audit'].items():
            if scalar(v):out['C_boundary_audit__'+f]=v
        for f,v in s['topology']['direction_solver_summary'].items():
            if scalar(v):out['direction_solver__'+f]=v
        for f,v in s['source_C_connectivity_audit'].items():
            if scalar(v):out['source_C_connectivity__'+f]=v
        if out['group_id']=='NA_NOT_RECORDED':missing('group_id','No recorded source group in registration or original inventory.','OLD6')
        metrics.append(out)
    for a,b in zip(original,metrics):
        if any(a[n]!=b[n]for n in a):raise ValueError('Original CSV precision/content changed')
    emitcsv('metrics_per_output.csv',metrics,sources=['evaluation/per_output.csv','evaluation/quality_per_output.csv','evaluation/topology_per_output.csv','evaluation/registration.json','840 score.json records',str(old_inventory_path)],
        kind='LOSSLESS_JOIN',note='Every original cell preserved byte-for-byte as a parsed CSV string; appended identities/status/denominators are from existing receipts, not recomputed geometry.')
    emit('AVMR_PMR_all8.csv',source_bytes(OLD/'evaluation/AVMR_PMR_all8.csv'),rows=len(vm),fields=list(vm[0]),sources=[str(OLD/'evaluation/AVMR_PMR_all8.csv')],kind='COPY_ORIGINAL')
    settings={}
    for r in vm:
        v={k:r[k]for k in ('setting','scale','tau','threshold_model','domain')}
        if r['setting']in settings and settings[r['setting']]!=v:raise ValueError('Threshold differs across outputs')
        settings[r['setting']]=v
    if len(settings)!=8 or any(len({r['setting']for r in vm if key(r)==k})!=8 for k in score_by_key):raise ValueError('Missing threshold row')
    index=[]
    for member,dest in FROZEN_CODE.items():
        b,ref=zip_member(member);target='protocol_and_code/'+dest
        emit(target,b,sources=[ref],kind='COPY_FROZEN_SNAPSHOT',note='Original archived bytes; never imported or executed during MINI extraction.')
        funcs=[dict(name=n.name,line_start=n.lineno,line_end=n.end_lineno)for n in ast.parse(b.decode('utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        index.append(dict(package_path=target,zip_member=member,source_path=BY_ARCHIVE[member]['source_path'],source_sha256=sha(b),functions=funcs))
    emitjson('protocol_and_code/source_code_index.json',index,kind='DERIVED',sources=list(FROZEN_CODE),note='AST-only source location index, no code execution.')
    smallfiles=['registration.json','data/data_freeze.json','training_budget_lock.json','initialization/audit.json','initialization/common_step0000.json',
       'inference/USER_SMALL_generator_step6000.json','inference/USER_EXPANDED_67_generator_step6000.json',
       'checkpoints/USER_SMALL_step6000.json','checkpoints/USER_EXPANDED_67_step6000.json',
       'evaluation/execution_identity_audit.json','evaluation/execution_identity_audit_attempt1_metadata_assertion.json',
       'review_work/REVIEW_RULES.md','review_work/stdin_bom_fix.json','evaluation/gallery/output_review_receipt.json','PROTOCOL.md']
    for rel in smallfiles:
        if not (OLD/rel).is_file():missing(rel,'Small existing receipt absent','protocol_and_code');continue
        emit('protocol_and_code/'+rel,source_bytes(OLD/rel),sources=[str(OLD/rel)],kind='COPY_EXISTING_RECEIPT')
    projected=[]
    request_fields=['request_id','parent_id','condition_id','panel','candidate','region_type','tier','N','K','ratio','group_uid','condition_type','source_quality','structure_tags','input_path','input_sha256','C_sha256','Gaussian_sha256','noise_seed','noise_provenance','condition_source','condition_source_sha256','original_source_face_ids','official_pretrain_seen','conditional_pretrain_seen']
    for q in reg['requests']:projected.append({f:q[f]for f in request_fields if f in q})
    proj={k:v for k,v in reg.items()if k!='requests'}
    proj.update(requests=projected,MINI_PROJECTION=dict(source=str(OLD/'evaluation/registration.json'),source_existing_sha256=SOURCES[str(OLD/'evaluation/registration.json')]['existing_sha256'],
        source_read_sha256=SOURCES[str(OLD/'evaluation/registration.json')]['read_sha256'],retained_request_fields=request_fields,omitted='Other original source face arrays, repeated input-review metadata and nested CC_CF_FF; see conditions/source-quality tables. No noise arrays read.'))
    emitjson('protocol_and_code/evaluation_registration_projection.json',proj,rows=len(projected),sources=[str(OLD/'evaluation/registration.json')],kind='PROJECTION')
    raw=source_json('evaluation/raw_index.json');post=source_json('evaluation/post_index.json');post_by={(x['request_id'],x['label']):x for x in post['rows']}
    caches=[]
    rawfields=['request_id','parent_id','condition_id','panel','candidate','label','N','K','checkpoint_sha256','input_path','input_sha256','C_sha256','Gaussian_sha256','generation_status','new_rollouts','noise_provenance','raw_path','raw_sha256','result_path','result_sha256']
    for r in raw['rows']:
        p=post_by[(r['request_id'],r['label'])]
        row={f:r.get(f,'NA_NOT_RECORDED')for f in rawfields}
        for f in ['post_path','post_sha256','proposal_path','proposal_receipt','new_D0_F','GEO_calls','source_sha256']:row[f]=p.get(f,'NA_NOT_RECORDED')
        row['RAW_contracts']=r.get('contracts',{})
        row['source_restoration']=p.get('source_restoration',{})
        caches.append(row)
    emitjson('protocol_and_code/execution_cache_mapping.json',dict(rows=caches,projection=True,weight_and_array_hashes='EXISTING_RECEIPTS_ONLY; no weights/NPZ/PLY reread',raw_source=str(OLD/'evaluation/raw_index.json'),post_source=str(OLD/'evaluation/post_index.json')),rows=len(caches),sources=[str(OLD/'evaluation/raw_index.json'),str(OLD/'evaluation/post_index.json')],kind='PROJECTION')
    missing('standalone_geometry_recompute','MINI deliberately omits meshes, query arrays and weights. Frozen function definitions are complete evidence; aggregate tables can be recomputed without external files, geometry cannot.','all',status='INTENTIONALLY_OMITTED_ASSETS')
    entries=numerical_dictionary(names,directions,settings)
    definitions=dict(version='R2_32_67_ANALYSIS_MINI_V1_metric_dictionary',registered_metric_count=96,registered_metric_names=names,primary_key=list(KEY),
       original_csv_precision='All original metric strings kept exactly. Fractions are 0..1; multiply by100 only for explicit percent display. Counts/length/area never implicitly converted.',
       null_encoding='Original CSV empty string represents prior JSON null. NA/UNKNOWN/PARTIAL/LOW_SUPPORT/CHANGED states remain strings. No unavailable value is zero-filled.',
       fixed_L=9.208138264587333,L_definition='sqrt(3)*2/0.3762 canonical training cube diagonal; fixed across outputs, not per-output bbox',
       matching_settings=list(settings.values()),matching_rule='Strict Euclidean distance < threshold; empty boundary: NA_EMPTY_BOUNDARY with VMR/perfect null. F-only matching cannot use C itself. Paper-scale equivalence is not asserted.',
       legal_graph='Exactly two distinct nondegenerate incident faces, positive edge length/area, neither face in indexed duplicate group, saved pair classification in the original LEGAL set. C-C edges are excluded by actual_valid_graph because they have no F incident; all C nodes seed reachability. Unknown, non-contact, illegal and nonmanifold edges are rejected, not repaired. Opposite reach additionally requires opposite edge traversal.',
       I_definition='Weighted union cross OR dup; cross within0.005L proper-segment band; dup inside>=0.01L C boundary and finite F distance<0.005L with abs-normal alignment<=15deg. Upper adds uncertainty. No semantic duplicate certification.',
       row_domains='RAW and POST remain separate; canonical-coordinate exact topology fields differ in scope from source-oriented valid graph fields.',
       aliases={'parent_uid':'parent_id','noise_id':'candidate','original_model_label':'label','model_short':SHORT,'A_VMR__SETTING':'VMR of matching AVMR_PMR_all8 row','PMR__SETTING':'perfect of matching AVMR_PMR_all8 row','I':'ConflictCov_C','CF':'CF_illegal_affected_free_fraction (proper separate)','FF':'FF_illegal_affected_free_fraction','free_area_vs_free_total_surface_area':'Same geometric domain, computed by separate preserved original routines; do not force bitwise equality'},
       aggregation_original='protocol_and_code/original/scoring.py::aggregate. Noise→condition→parent; same parent equal weight. Missing strata statuses retained. Existing topology and detector statuses are not silently mapped to numeric zero.',
       source_quality_original='original/source_quality_scan.py::partition -> original/source_data_freeze.py::emit; ILLEGAL if FF confirmed-illegal count>0 else UNKNOWN if FF unknown>0 else NO_DETECTED_PAIR_PROBLEM. Parent strata use condition labels, not an invented worst-risk override.',
       source_parent_helper='protocol_and_code/extract_metrics.py::derive_source_parent_groups; DERIVED helper explicitly records mixed parent membership.',
       numerical_fields=entries,
       metadata_fields={f:dict(origin='EXISTING_ORIGINAL_FIELD_OR_IDENTITY_ALIAS',definition='See original name / prefix and score receipt; source scalar retained, not geometric recomputation')for f in metrics[0]if f not in names},
       extra_scalar_prefixes={'direction_solver__':'Existing direction_solver_summary scalars, not added to the registered96','C_boundary_audit__':'Existing source-C boundary audit scalars','source_C_connectivity__':'Existing source vertex-ID topology audit scalars','metric_status__':'Original status_by_metric entries'},
       code_index='protocol_and_code/source_code_index.json')
    emitjson('metric_dictionary.json',definitions,kind='DICTIONARY_DOCUMENTATION',sources=['frozen actual code snapshots','report_tables/metric_directions.csv','840 score receipts'])
    emit('protocol_and_code/metric_directions.csv',source_bytes(OLD/'report_tables/metric_directions.csv'),rows=96,sources=[str(OLD/'report_tables/metric_directions.csv')],kind='COPY_ORIGINAL')
    emitjson('protocol_and_code/label_mapping.json',dict(original_to_short=SHORT,comparison='USER_EXPANDED_67 - USER_SMALL',HIST_role='frozen historical reference, not strict scale-causal arm',panels={'NEW32':'32 parents x2 C x2 fixed noises per model','OLD6':'6 historical parents x1 C x2 fixed noises per model'},stages={'RAW':'original fixed Euler50 generated triangles','POST':'single official D0 + original GEO_C + source-C restoration + fixed-C free orientation'}),kind='IDENTITY_ALIAS')
    selfpath=Path(__file__).resolve();PRODUCTS[str(selfpath.relative_to(OUT))]=dict(sha256=sha(selfpath.read_bytes()),bytes=selfpath.stat().st_size,rows=None,source_refs=[],transformation='NEW_EXTRACTION_CODE',DERIVED=False,note='New stdlib-only extractor, not a replacement scientific scorer.')
    emitjson('_work/metrics_missing.json',dict(status='COMPLETE_WITH_EXPLICIT_MISSING_METADATA',items=MISSING),kind='DERIVED',sources=[])
    provenance=dict(status='COMPLETE',experiment='R2_USER_DATA_SCALING_32_67_V1',MINI='R2_32_67_ANALYSIS_MINI_V1',
        old_out_read_only=str(OLD),old_zip=str(ZIP_PATH),sources=list(SOURCES.values()),outputs=PRODUCTS,
        metrics=96,metric_rows=840,threshold_rows=6720,source_precision_preserved=True,
        new_model_calls=0,new_noise=0,new_geometry_scores=0,new_render=0,new_visual_reviews=0,
        source_group_recompute='derive_source_parent_groups is a new explicit projection helper; original per-condition partition/emit and original aggregate are provided verbatim.',
        snapshots_taken_from='OLD DELIVERED ZIP MEMBERS, SHA checked against frozen PACKAGE_MANIFEST; not current HEAD')
    (OUT/'_work').mkdir(parents=True,exist_ok=True)
    pp=OUT/'_work/metrics_provenance.json'
    pb=(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n').encode()
    if pp.exists()and pp.read_bytes()!=pb:raise RuntimeError('Refuse overwrite existing provenance')
    pp.write_bytes(pb)
    print(compact(dict(status='COMPLETE',metrics_rows=len(metrics),all8_rows=len(vm),scalar_fields=len(names),outputs=len(PRODUCTS),source_files_read=len(SOURCES),new_geometry_calls=0,provenance=str(pp))))
if __name__=='__main__':main()
