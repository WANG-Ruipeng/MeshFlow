"""Read-only metadata extraction for R2_32_67_ANALYSIS_MINI_V1.
No mesh coordinates, model, geometry function, renderer, or audit entrypoint are read/run.
New numeric fields are DERIVED solely by filtering/counting saved JSON records.
"""
import argparse, ast, collections, csv, datetime, hashlib, io, json, statistics, zipfile
from pathlib import Path

BASE = Path('/mnt/e/MeshFlow/experiments/r2_user_data_scaling_32_67_v1')
OLD = Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1')
OUT = Path('/mnt/e/MeshFlow/deliverables/R2_32_67_ANALYSIS_MINI_V1')
NA = 'NOT_MEASURED'
SOURCES, CACHE, REFERENCES, OUTPUTS = {}, {}, {}, {}
MISSING = []

def h(data):
    return hashlib.sha256(data).hexdigest()
def jtext(v):
    return json.dumps(v, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
def source(path, expected=None, parse=True):
    p = str(path)
    if p not in SOURCES:
        data = Path(p).read_bytes()
        sid = 'DATA_S%03d' % (len(SOURCES) + 1)
        SOURCES[p] = dict(source_id=sid,path=p,bytes=len(data),
            existing_sha256=expected or 'NOT_RECORDED',
            this_read_sha256=h(data),this_read=True,
            existing_hash_match=(h(data)==expected if expected else 'NOT_APPLICABLE'))
        if expected and h(data) != expected:
            raise AssertionError(('source hash mismatch',p,expected,h(data)))
        CACHE[p] = json.loads(data.decode('utf-8-sig')) if parse else data
    elif expected:
        assert SOURCES[p]['this_read_sha256']==expected, (p,expected)
        SOURCES[p]['existing_sha256']=expected
        SOURCES[p]['existing_hash_match']=True
    return CACHE[p], SOURCES[p]['source_id']
def reference(path, expected=None, kind='NPZ_ASSET_NOT_REREAD'):
    if path:
        REFERENCES[str(path)]=dict(path=str(path),existing_sha256=expected or 'NOT_RECORDED',
            this_read=False,this_read_sha256='NOT_REREAD',kind=kind)
def write_json(name, value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,separators=(',', ':'),allow_nan=False)+'\n',encoding='utf-8')
    OUTPUTS[name]=dict(path=name,sha256=h(p.read_bytes()),bytes=p.stat().st_size,rows=None)
def write_csv(name, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:
            w.writerow({k:jtext(v) if isinstance(v,(list,dict)) else 'NULL' if v is None else v for k,v in row.items()})
    OUTPUTS[name]=dict(path=name,sha256=h(p.read_bytes()),bytes=p.stat().st_size,rows=len(rows),fields=keys)
def value(row,*keys, default=NA):
    for k in keys:
        if k in row:return row[k]
    return default
def role(uid, small, train, dev):
    if uid in small:return 'COMMON_SMALL32'
    if uid in train:return 'EXP_ONLY35'
    if uid in dev:return 'NEW_DEV32'
    return 'OLD6'
def setpart(i,j,c):
    return 'CC' if i in c and j in c else 'CF' if (i in c)!=(j in c) else 'FF'
def risk(part):
    return 'ILLEGAL' if part['FF']['illegal'] else 'UNKNOWN' if part['FF']['unknown'] else 'NO_DETECTED_PAIR_PROBLEM'
def excerpt(z,name,functions):
    data=z.read(name);text=data.decode('utf-8-sig');lines=text.splitlines()
    chunks=[]
    for node in ast.walk(ast.parse(text)):
        if isinstance(node,ast.FunctionDef) and node.name in functions:
            chunks.append('# ZIP member: %s\n# Original full file SHA256: %s\n# Function %s, original lines %d-%d\n%s\n' %
                (name,h(data),node.name,node.lineno,node.end_lineno,'\n'.join(lines[node.lineno-1:node.end_lineno])))
    assert len(chunks)==len(functions)
    return '\n'.join(chunks)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train,ts=source(BASE/'data/train_manifest.json')
    panel,ps=source(BASE/'data/panel_manifest.json')
    reg,rs=source(BASE/'evaluation/registration.json')
    roles,ros=source(BASE/'data/parent_roles.json',train['parent_roles']['sha256'])
    inherit,ins=source(BASE/'data/raw_inheritance_hashes.json')
    freeze,fs=source(BASE/'data/data_freeze.json')
    for item in inherit['prepared_asset_hash_checks']:
        reference(item['path'],item['sha256'])
    oldconds,ocs=source(OLD/'conditions.json')
    oldparents,ops=source(OLD/'parent_inventory.json')
    oldregions,ors=source(OLD/'regions.json')
    oldactive,oas=source(OLD/'active_conditions.json')
    oldqsummary,oqs=source(OLD/'quality/summary.json')
    # The MINI metrics projection carries unchanged original score boundary_audit.
    # Read no output quality values; validate only C audit invariance over 12 records.
    mbytes,mbs=source(OUT/'metrics_per_output.csv',parse=False)
    mrows=list(csv.DictReader(io.StringIO(mbytes.decode('utf-8-sig'))))
    bf=['C_nonmanifold_edge_count','C_vertex_count','boundary_edge_count','boundary_length_model',
        'boundary_vertex_count','coincident_distinct_source_vertex_count','coordinate_boundary_vertex_count',
        'source_and_coordinate_boundary_locations_equal']
    boundary_groups=collections.defaultdict(list)
    for mr in mrows: boundary_groups[(mr['parent_uid'],mr['condition_id'])].append(mr)
    assert len(mrows)==840 and len(boundary_groups)==70
    boundary_map={}
    for key,mrs in boundary_groups.items():
        assert len(mrs)==12
        for field in bf:
            assert len({mr['C_boundary_audit__'+field] for mr in mrs})==1,(key,field,'saved C audits disagree')
        boundary_map[key]=mrs[0]
    oldcmap={c['condition_id']:c for c in oldconds['rows']}
    oldpmap={p['uid']:p for p in oldparents['rows']}
    oldrmap={r['region_id']:r for r in oldregions['rows']}
    small=set(train['pools']['USER_SMALL']); tr=set(train['pools']['USER_EXPANDED_67'])
    dp={p['uid']:p for p in panel['dev_parents']}; dev=set(dp)
    assert len(small)==32 and len(tr)==67 and small < tr and len(dev)==32 and not tr&dev
    parentmeta={**train['parents'],**dp}
    assert not {parentmeta[u]['group_uid'] for u in tr}&{parentmeta[u]['group_uid'] for u in dev}
    assert all(not dp[u].get('known_condition_train',False) and not dp[u].get('known_development',False) for u in dev)
    allc=[(c,'TRAIN',ts) for c in train['tasks'].values()]+[(c,'NEW32',ps) for c in panel['conditions']]
    oldrequests={r['condition_id']:r for r in reg['requests'] if r['panel']=='OLD6'}
    assert len(oldrequests)==6
    for cid,r in oldrequests.items():
        c=dict(oldcmap[cid]);c['parent_id']=c['uid'];c['condition_type']=c['region_type'].upper()+'_'+c['tier'].upper()
        c['condition_source']=c['npz'];c['condition_source_sha256']=c['sha256']
        assert c['npz']==r['condition_source'] and c['sha256']==r['condition_source_sha256']
        allc.append((c,'OLD6',ocs))
    conditions=[]; qualities=[]; parentcs=collections.defaultdict(list); events={}
    for c,panelname,manifest_sid in allc:
        uid=value(c,'uid','parent_id');cid=c['condition_id'];N=c['N'];K=c['K']
        r='OLD6' if panelname=='OLD6' else role(uid,small,tr,dev)
        p=oldpmap[uid] if panelname=='OLD6' else parentmeta[uid]
        qpath=OLD/'quality/parents'/f'{uid}.json' if panelname=='OLD6' else Path(c['quality_path'])
        qexpected=None if panelname=='OLD6' else c['quality_sha256']
        if panelname=='OLD6':
            cq,cqs=source(OLD/'quality/conditions'/f'{cid}.json')
            qexpected=cq.get('parent_quality_sha256')
        q,qs=source(qpath,qexpected)
        assert q['N']==N and len(q['basic_geometry']['per_face'])==N
        C=set(c['source_face_ids']);Y=set(c['free_source_ids'])
        assert len(C)==K and not C&Y and C|Y==set(range(N))
        original=q['original_face_ids']
        original_C=c.get('original_source_face_ids',[original[i] for i in c['source_face_ids']])
        if 'original_source_face_ids' in c:
            assert original_C==[original[i] for i in c['source_face_ids']]
        # Partition only saved detector records; no triangle/topology computation.
        parts={x:dict(illegal=0,unknown=0,proper=0,coplanar_positive_overlap=0,legal_contact=0,
                      shared_edge_contact=0,shared_vertex_contact=0,event_ids=[]) for x in ['CC','CF','FF']}
        if qs not in events:
            pf=q['basic_geometry']['per_face']
            saved=[]
            for event in q['pair_contacts']:
                if event['category'] in ('proper','coplanar_positive_overlap','UNKNOWN') or event['classification']=='shared_edge_contact':
                    i,j=event['pair'];eid=f'{qs}:{i}:{j}'
                    saved.append([eid,i,j,event.get('original_face_ids',[original[i],original[j]]),event['category'],event['classification']])
            events[qs]=dict(parent_uid=uid,variant_uid=c.get('variant_uid','HISTORICAL_RETAINED_R2_PARENT'),
                source_id=qs,source_path=str(qpath),source_existing_sha256=qexpected or 'NOT_RECORDED',
                source_this_read_sha256=SOURCES[str(qpath)]['this_read_sha256'],
                original_face_ids=original,
                per_face_columns=['source_face_id','original_face_id','minimum_angle_degrees','finite','exact_zero_area'],
                per_face=[[f['source_face_id'],f.get('original_face_id'),f.get('minimum_angle_degrees'),f.get('finite'),f.get('exact_zero_area')] for f in pf],
                exact_duplicate_groups=q['basic_geometry'].get('exact_duplicate_groups',[]),
                contact_columns=['event_id_DERIVED','source_i','source_j','original_face_ids','category','classification'],
                contacts=saved,
                projection='All saved problem contacts (proper/coplanar/UNKNOWN), all saved shared_edge_contact face pairs, and all saved per-face minimum angles. Other legal contact details, mesh coordinates, detector internals and area values omitted; complete original JSON path/hash retained.')
        for event in q['pair_contacts']:
            i,j=event['pair'];k=setpart(i,j,C);a=parts[k];cat=event['category']
            if cat in ('proper','coplanar_positive_overlap'):
                a['illegal']+=1;a[cat]+=1;a['event_ids'].append(f'{qs}:{i}:{j}')
            elif cat=='UNKNOWN':
                a['unknown']+=1;a['event_ids'].append(f'{qs}:{i}:{j}')
            elif cat=='legal_contact':
                a['legal_contact']+=1
            cls=event['classification']
            if cls in ('shared_edge_contact','shared_vertex_contact'):a[cls]+=1
        if panelname=='OLD6':
            originalparts={k:dict(illegal=cq['partitions'][k]['confirmed_illegal_pair_count'],
                unknown=cq['partitions'][k]['unknown_pair_count']) for k in parts}
            label=risk(originalparts);labelorigin='DERIVED_FROM_EXISTING_OLD6_FF_COUNTS; same rule as new pool; no original same-name label'
            annotation=oldrmap[c['region_id']]
            annpath=annotation.get('review_source',{}).get('path');annsha=annotation.get('review_source',{}).get('sha256')
            review_reason=annotation.get('reason','UNKNOWN')
            structure_tags=['UNKNOWN']; semstatus=c.get('region_status','UNKNOWN')
            region_declared=annotation.get('region_type','UNKNOWN')
            cp=C;cy_sid=cqs
        else:
            originalparts=c['CC_CF_FF'];label=c['source_quality'];labelorigin='INHERITED_CONDITION_MANIFEST; source FF only'
            assert label==risk(originalparts)
            annpath=c.get('annotation_source');annsha=c.get('annotation_sha256')
            review_reason=c.get('review_reason','UNKNOWN');structure_tags=c.get('structure_tags') or ['UNKNOWN']
            semstatus='FROZEN_QUALIFIED_AI_PROXY';region_declared=c.get('region_declared','UNKNOWN')
            cy_sid=manifest_sid
        for k in parts:
            assert parts[k]['illegal']==originalparts[k]['illegal'] and parts[k]['unknown']==originalparts[k]['unknown'],(cid,k)
        if panelname!='OLD6':
            assert all(originalparts[k]['illegal']==0 and originalparts[k]['unknown']==0 for k in ('CC','CF'))
        if annpath:
            _,anns=source(annpath,annsha)
        else:anns='NOT_RECORDED'
        reference(c['condition_source'],c['condition_source_sha256'],'CONDITION_NPZ_NOT_REREAD')
        reference(p['npz'],p['sha256'],'PARENT_NPZ_NOT_REREAD')
        common=dict(parent_uid=uid,group_id=value(p,'group_uid','group_id',default=uid),condition_id=cid,
            parent_role=r,panel=panelname,in_small=(panelname=='TRAIN' and uid in small),
            in_expanded=(panelname=='TRAIN'),condition_type=c['condition_type'],
            region_type=c['region_type'],tier=c['tier'],N=N,K=K,Y_face_count_DERIVED=N-K)
        crow=dict(common,K_over_N=value(c,'K_over_N','face_fraction'),
            C_over_M_area_ratio=value(c,'area_C_over_M','parent_area_fraction'),
            C_over_region_area_ratio=value(c,'area_C_over_region','region_area_fraction'),
            area_ratio_definition='sum of triangle areas; not exposed/union area',
            C_components=value(c,'C_components'),Y_components=value(c,'Y_components'),
            C_boundary_vertex_count=NA,C_boundary_edge_count=NA,Y_boundary_vertex_count=NA,Y_boundary_edge_count=NA,
            source_C_face_ids=c['source_face_ids'],source_Y_face_ids=c['free_source_ids'],
            original_source_C_face_ids=original_C,
            original_source_C_face_ids_origin=('INHERITED' if 'original_source_face_ids' in c else 'DERIVED_INDEXING_EXISTING_ORIGINAL_FACE_ID_MAP'),
            region_source_face_ids=c['region_source_face_ids'],
            source_corner_permutation_reference=c['condition_source']+'#C_source_corner_permutations',
            source_vertex_ids_reference=c['condition_source']+'#C_source_vertex_ids',
            condition_source=c['condition_source'],condition_existing_sha256=c['condition_source_sha256'],
            condition_this_read='NOT_REREAD',variant_uid=value(c,'variant_uid',default='HISTORICAL_RETAINED_R2_PARENT'),
            review_kind=c.get('review_kind','AI_PROXY'),semantic_status=semstatus,
            region_declared=region_declared,semantic_review_reason=review_reason,
            semantic_uncertainty='AI_PROXY; not human semantic certification',
            structure_tags=structure_tags,annotation_source_id=anns,annotation_source=annpath or 'NOT_RECORDED',
            annotation_existing_sha256=annsha or 'NOT_RECORDED',source_quality=label,
            source_quality_label_origin=labelorigin,source_quality_source_id=cy_sid,
            quality_source_id=qs,manifest_source_id=manifest_sid,
            missing_reason='Boundary counts absent in inherited data-construction/quality records; OLD6 connectivity counts absent in selected condition records.')
        br=boundary_map.get((uid,cid))
        crow['C_boundary_audit_origin']='NOT_MEASURED' if br is None else 'INHERITED_ORIGINAL_SCORE_SOURCE_C_AUDIT; all 12 model/noise/stage records equal'
        crow['C_boundary_audit_projection_source_id']=mbs if br is not None else 'NOT_MEASURED'
        for field in bf:
            crow['C_source_boundary_audit__'+field]=br['C_boundary_audit__'+field] if br is not None else NA
        if br is not None:
            crow['C_boundary_vertex_count']=br['C_boundary_audit__boundary_vertex_count']
            crow['C_boundary_edge_count']=br['C_boundary_audit__boundary_edge_count']
            crow['C_boundary_audit_score_file_path']=br['score_file_path']
            crow['C_boundary_audit_score_record_key']=br['score_record_key']
            crow['C_boundary_audit_score_existing_sha256']=br['score_file_existing_sha256']
            crow['C_boundary_audit_score_read_sha256_in_projection']=br['score_file_read_sha256']
            crow['missing_reason']='Source Y boundary counts not stored; OLD6 C/Y component counts absent. C boundary audit inherited from saved score, not source C/Y interface measurement.'
        else:
            crow['C_boundary_audit_score_file_path']=NA
            crow['C_boundary_audit_score_record_key']=NA
            crow['C_boundary_audit_score_existing_sha256']=NA
            crow['C_boundary_audit_score_read_sha256_in_projection']=NA
        conditions.append(crow);parentcs[(r,uid)].append(crow)
        bg=q['basic_geometry'];ypf=[f for f in bg['per_face'] if f['source_face_id'] in Y]
        angles=[f['minimum_angle_degrees'] for f in ypf if f.get('minimum_angle_degrees') is not None]
        badangle=[f['source_face_id'] for f in ypf if f.get('minimum_angle_degrees') is not None and f['minimum_angle_degrees']<10.0]
        zeros=[i for i in bg['exact_zero_area_source_face_ids'] if i in Y]
        nonfinite=[i for i in bg['nonfinite_source_face_ids'] if i in Y]
        duplicate_groups=[]
        for g in bg['exact_duplicate_groups']:
            ids=g if isinstance(g,list) else g.get('source_face_ids',g.get('face_ids'))
            if ids is None:raise ValueError(('unrecognized existing duplicate record',str(qpath),g))
            yy=[i for i in ids if i in Y]
            if len(yy)>=2:duplicate_groups.append(yy)
        qrow=dict(common,source_quality=label,source_quality_label_origin=labelorigin,
            source_quality_parent_rule='No forced parent severity; parents.csv preserves set of existing condition labels. DEV sets verified singletons.',
            quality_source_id=qs,quality_path=str(qpath),quality_existing_sha256=qexpected or 'NOT_RECORDED',
            quality_this_read_sha256=SOURCES[str(qpath)]['this_read_sha256'],
            original_partition_counts_source_id=cy_sid,face_event_table_reference='source_events.json#'+qs)
        for k,a in parts.items():
            qrow.update({k+'_confirmed_illegal_pairs':originalparts[k]['illegal'],
                k+'_unknown_pairs':originalparts[k]['unknown'],
                k+'_proper_pairs_DERIVED':a['proper'],
                k+'_coplanar_positive_overlap_pairs_DERIVED':a['coplanar_positive_overlap'],
                k+'_problem_event_ids_DERIVED':a['event_ids']})
        qrow.update(Y_angle_valid_denominator_DERIVED=len(angles),
            Y_minimum_angle_min_degrees_DERIVED=min(angles) if angles else 'NULL',
            Y_minimum_angle_mean_degrees_DERIVED=statistics.mean(angles) if angles else 'NULL',
            Y_small_angle_lt10_faces_DERIVED=len(badangle),Y_small_angle_lt10_fraction_DERIVED=len(badangle)/(N-K),
            Y_small_angle_source_face_ids_DERIVED=badangle,
            Y_exact_zero_area_count_DERIVED=len(zeros),Y_exact_zero_area_face_ids_DERIVED=zeros,
            Y_nonfinite_count_DERIVED=len(nonfinite),Y_nonfinite_face_ids_DERIVED=nonfinite,
            Y_exact_duplicate_groups_DERIVED=duplicate_groups,
            Y_exact_duplicate_group_count_DERIVED=len(duplicate_groups),
            Y_duplicate_involved_face_count_DERIVED=len(set(i for g in duplicate_groups for i in g)),
            Y_other_degeneracy_count=NA,Y_q_gt10_count=NA,
            source_CY_geometric_legal_contact_face_pairs_DERIVED=parts['CF']['legal_contact'],
            source_CY_geometric_shared_edge_contact_face_pairs_DERIVED=parts['CF']['shared_edge_contact'],
            source_CY_geometric_shared_vertex_contact_face_pairs_DERIVED=parts['CF']['shared_vertex_contact'],
            source_CY_original_index_shared_edges=NA,
            source_CY_coordinate_merged_shared_edges=NA,
            source_CY_legality_checked_unique_interface_edges=NA,
            source_CY_legality_checked_interface_length=NA,
            source_C_boundary_vertices_matching_Y=NA,
            source_interface_problem_label=NA,
            source_problem_contact_involves_C_DERIVED=bool(parts['CC']['illegal']+parts['CC']['unknown']+parts['CF']['illegal']+parts['CF']['unknown']),
            interface_limitation='Detector face-pair contact class is not indexed connectivity, welded unique edges, valid-interface length, or a user-required seam.',
            derivation='DERIVED fields filter/count saved per-face/contact records. Angle threshold strictly <10 degrees; denominator N-K. No geometry executed.')
        qualities.append(qrow)
    assert len(conditions)==264 and len({(c['parent_role'],c['condition_id']) for c in conditions})==264
    assert sum(c['in_small'] for c in conditions)==91 and sum(c['in_expanded'] for c in conditions)==194
    parents=[]
    for (r,uid),cs in sorted(parentcs.items()):
        p=oldpmap[uid] if r=='OLD6' else parentmeta[uid]
        labels=sorted(set(c['source_quality'] for c in cs));tags=sorted(set(t for c in cs for t in c['structure_tags']))
        parents.append(dict(parent_uid=uid,group_id=value(p,'group_uid','group_id',default=uid),
            parent_role=r,official_split=p.get('official_split','UNKNOWN'),
            variant_uid=p.get('variant_uid','HISTORICAL_RETAINED_R2_PARENT'),
            N=p['N'],condition_count_DERIVED=len(cs),condition_ids_DERIVED=[c['condition_id'] for c in cs],
            structure_tags_DERIVED=tags,structure_label_basis='Inherited per-condition input AI_PROXY tags; UNKNOWN where absent, never reclassified.',
            source_qualification_status='FROZEN_QUALIFIED_INPUT_SUBSET' if r!='OLD6' else 'HISTORICAL_FROZEN_REGRESSION',
            source_quality_labels_DERIVED=labels,
            source_quality_parent_label_DERIVED=labels[0] if len(labels)==1 else 'HETEROGENEOUS_NO_FORCED_LABEL',
            source_quality_parent_rule='Set of all actually used condition labels; singleton only if identical.',
            official_pretrain_seen=p.get('official_pretrain_seen','UNKNOWN'),
            official_generator_pretraining_seen=p.get('official_generator_pretraining_seen',p.get('official_pretrain_seen','UNKNOWN')),
            official_denoiser_pretraining_seen=p.get('official_denoiser_pretraining_seen','UNKNOWN'),
            known_historical_condition_train=p.get('known_condition_train',p.get('conditional_pretrain_seen','UNKNOWN')),
            known_historical_development=p.get('known_development',True if r=='OLD6' else 'UNKNOWN'),
            source_npz=p['npz'],source_existing_sha256=p['sha256'],source_this_read='NOT_REREAD',
            source_original_asset=value(p,'original_source_file','source_file',default='NOT_RECORDED'),
            source_original_existing_sha256=value(p,'original_source_sha256','source_sha256',default='NOT_RECORDED'),
            metadata_source_id=ops if r=='OLD6' else ps if r=='NEW_DEV32' else ts,
            historical_qualification_order='Existing C-region screening preceded original DEV assignment; new 32/67 protocol accepted frozen roles before training.'))
    assert collections.Counter(p['parent_role'] for p in parents)==dict(COMMON_SMALL32=32,EXP_ONLY35=35,NEW_DEV32=32,OLD6=6)
    devlabels=collections.Counter(p['source_quality_parent_label_DERIVED'] for p in parents if p['parent_role']=='NEW_DEV32')
    assert devlabels==dict(ILLEGAL=20,NO_DETECTED_PAIR_PROBLEM=11,UNKNOWN=1)
    write_csv('parents.csv',parents);write_csv('conditions.csv',conditions);write_csv('source_quality_and_interfaces.csv',qualities)
    write_json('source_events.json',dict(schema='saved_source_diagnostics_projection_v1',status='DERIVED_PROJECTION_NO_NEW_GEOMETRY',
        source_face_id_domain='retained source M row; original_face_ids maps saved raw source row',
        event_id_definition='source_quality_id:source_i:source_j; new identifier for an existing saved detector face pair',
        records=events))
    summary={}
    for r in ['COMMON_SMALL32','EXP_ONLY35','NEW_DEV32','OLD6']:
        qr=[x for x in qualities if x['parent_role']==r]
        pr=[x for x in parents if x['parent_role']==r]
        byp=collections.defaultdict(list)
        for x in qr:byp[x['parent_uid']].append(x)
        eventset={k:set() for k in ['CC','CF','FF']}
        for x in qr:
            for k in eventset:eventset[k].update(x[k+'_problem_event_ids_DERIVED'])
        summary[r]=dict(parents=len(pr),conditions=len(qr),
            condition_quality_counts=dict(collections.Counter(x['source_quality'] for x in qr)),
            parent_quality_counts=dict(collections.Counter(x['source_quality_parent_label_DERIVED'] for x in pr)),
            unique_existing_problem_event_counts_DERIVED={k:len(v) for k,v in eventset.items()},
            condition_mean_Y_small_angle_fraction_DERIVED=statistics.mean(x['Y_small_angle_lt10_fraction_DERIVED'] for x in qr),
            parent_equal_mean_Y_small_angle_fraction_DERIVED=statistics.mean(statistics.mean(x['Y_small_angle_lt10_fraction_DERIVED'] for x in xs) for xs in byp.values()),
            parents_with_Y_exact_zero_DERIVED=sum(any(x['Y_exact_zero_area_count_DERIVED'] for x in xs) for xs in byp.values()),
            parents_with_Y_duplicate_groups_DERIVED=sum(any(x['Y_exact_duplicate_group_count_DERIVED'] for x in xs) for xs in byp.values()),
            parents_with_Y_nonfinite_DERIVED=sum(any(x['Y_nonfinite_count_DERIVED'] for x in xs) for xs in byp.values()),
            condition_mean_CF_geometric_shared_edge_contact_pairs_DERIVED=statistics.mean(x['source_CY_geometric_shared_edge_contact_face_pairs_DERIVED'] for x in qr),
            source_indexed_or_legal_unique_interfaces=NA)
    answer=dict(status='DESCRIPTIVE_ONLY_DERIVED',groups=summary,
        evidence='Existing source Y contains confirmed illegal/UNKNOWN face pairs and small-angle faces; COMMON32 and EXP_ONLY35 composition is measurable. Condition repetitions are not independent defect events.',
        limitations=['No measured original-index or coordinate-merged unique source C/Y interfaces, legal interface count/length, or boundary-vertex matching in these inherited source records.',
        'Saved geometric shared_edge_contact face-pair classification is not a unique topology/interface edge.',
        'Source Y is one reference completion, not the required output shape.',
        'These are composition summaries, not exposure-weighted gradient contribution or a causal explanation of result changes.',
        'Exact-zero/duplicate/nonfinite diagnostics do not cover every possible degeneracy.'],
        dev_quality_parent_rule=dict(rule='Each of each parent two registered conditions has identical inherited FF-only label; no maximum/first-row fallback.',counts=dict(devlabels),unknown_parent_ids=[p['parent_uid'] for p in parents if p['parent_role']=='NEW_DEV32' and p['source_quality_parent_label_DERIVED']=='UNKNOWN']),
        independent_checks=dict(parent_roles=dict(collections.Counter(p['parent_role'] for p in parents)),unique_condition_count=264,small_conditions=91,expanded_conditions=194,new_dev_conditions=64,old_conditions=6,
            train_dev_parent_disjoint=True,train_dev_group_disjoint=True,dev_known_historical_train_or_development=False,
            old6_uid_overlap_with_train=sorted(set(oldrequests[c]['parent_id'] for c in oldrequests)&tr),
            original_partition_counts_equal_saved_event_recount=True,evaluation_C_boundary_audits_inherited=70,C_audit_records_checked=840,models=0,geometry_calls=0,mesh_coordinate_reads=0))
    write_json('_work/data_answer6.json',answer)
    MISSING.extend([
        dict(fields=['source_CY_original_index_shared_edges','source_CY_coordinate_merged_shared_edges','source_CY_legality_checked_unique_interface_edges','source_CY_legality_checked_interface_length','source_C_boundary_vertices_matching_Y','source_interface_problem_label'],rows=264,status=NA,reason='No corresponding indexed/welded/valid interface measurements in inherited source geometry records. Existing detector face-pair labels are preserved separately. No new mesh/topology computation authorized.'),
        dict(fields=['C_boundary_vertex_count','C_boundary_edge_count'],rows=194,scope='TRAIN',status=NA,reason='Not stored in training condition/source-quality records. The 70 DEV/OLD conditions inherit saved score C-only boundary audits after verifying all 12 records agree.'),
        dict(fields=['Y_boundary_vertex_count','Y_boundary_edge_count'],rows=264,status=NA,reason='No source Y boundary records. Generated free boundary counts are not source Y counts.'),
        dict(fields=['C_components','Y_components'],rows=6,scope='OLD6',status=NA,reason='Absent in old condition manifest; no new topology computation.'),
        dict(fields=['Y_other_degeneracy_count','Y_q_gt10_count'],rows=264,status=NA,reason='Only existing exact-zero/nonfinite/duplicate and saved minimum-angle diagnostics available; no new metric.'),
        dict(fields=['structure_tags'],rows=6,scope='OLD6',status='UNKNOWN',reason='No matching structure labels in selected old source metadata; semantic region review retained.'),
        dict(fields=['official_pretrain_seen'],rows=105,status='UNKNOWN',reason='Official split membership does not prove generator/denoiser pretraining exposure.'),
    ])
    write_json('_work/data_missing.json',MISSING)
    # Read actual frozen source snapshots from the completed ZIP, never current HEAD.
    z=zipfile.ZipFile(BASE/'delivery/R2_USER_DATA_SCALING_32_67_REVIEW.zip')
    chunks=[excerpt(z,'inherited/code_snapshot/quality_scan.py',{'partition'}),
            excerpt(z,'inherited/code_snapshot/data_freeze.py',{'emit'})]
    snap=OUT/'protocol_and_code/source_quality_rule_functions.txt'
    snap.write_text('# Exact complete functions from prior frozen experiment code. This is source evidence, not a runnable substitute.\n\n'+'\n'.join(chunks),encoding='utf-8')
    OUTPUTS['protocol_and_code/source_quality_rule_functions.txt']=dict(path='protocol_and_code/source_quality_rule_functions.txt',sha256=h(snap.read_bytes()),bytes=snap.stat().st_size,rows=None,
        projection='Complete partition and emit functions with original file hashes and line numbers; no reimplementation.')
    write_json('_work/data_provenance.json',dict(schema='mini_data_extraction_provenance_v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources=list(SOURCES.values()),reference_assets_not_reread=list(REFERENCES.values()),outputs=OUTPUTS,
        source_fields='Original JSON numeric precision preserved; new fields named DERIVED except explicitly documented CSV identity aliases/projections. C_boundary_audit is inherited from the MINI metrics projection of saved original score records, not newly measured.',
        projections=dict(parents='Selected 99 new pool+DEV plus 6 OLD6 identities; original condition label sets, structure tags and version/pretraining fields.',
                         conditions='All 194 train +64 new DEV +6 OLD6. Source face ID lists and source permutation paths retained; no arrays/geometry included.',
                         source_quality_and_interfaces='Original partition counts/labels plus arithmetic filtering of saved per-face/contact records. No geometry evaluation.',
                         source_events='Deduplicated saved contacts and minimum angles; omitted fields declared in each source record.'),
        runtime=dict(model_calls=0,geometry_calls=0,mesh_coordinate_reads=0,noise_draws=0,render_calls=0,new_visual_reviews=0,weight_hashes=0),
        script_path=str(Path(__file__).resolve()),script_sha256=h(Path(__file__).read_bytes())))
    print(json.dumps(dict(status='COMPLETE',outputs=OUTPUTS,groups=summary,source_files=len(SOURCES)),ensure_ascii=False))
if __name__=='__main__':
    main()
