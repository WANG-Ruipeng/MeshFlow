"""CPU input/teacher collision audit; diagnostic only, no training or model change."""
from pathlib import Path
import argparse,json
import numpy as np
from ..io_utils import array_hash,file_hash,write_json,new_dir

def run(source,out):
    source=Path(source).resolve();out=new_dir(out);p=json.loads((source/'RUN_PROTOCOL.json').read_text())
    path=source/'train_labels.npz'
    if file_hash(path)!=p['hashes']['training_labels_sha256']:raise ValueError('Frozen training archive changed')
    with np.load(path,allow_pickle=False) as z:
        uid=z['train_uids'].tolist();features=np.array(z['point_features']);base=np.array(z['directions0']);teacher=np.array(z['directions1']);logr=np.array(z['logr'])
    if len(uid)!=12 or uid!=p['train_uids']:raise ValueError('Training parent set differs')
    canonical=np.stack([f[np.lexsort(f.T[::-1])] for f in features]); assert np.array_equal(canonical[0],features[0][::-1][np.lexsort(features[0][::-1].T[::-1])])
    records=[{'uid':u,'feature_typed_hash':array_hash(features[i]),'permutation_canonical_feature_hash':array_hash(canonical[i]),'base_typed_hash':array_hash(base[i]),'teacher_typed_hash':array_hash(teacher[i]),'logradius_hash':array_hash(logr[i])} for i,u in enumerate(uid)]
    pairs=[]
    for i in range(12):
        for j in range(i+1,12):
            same_base=base[i].dtype==base[j].dtype and np.array_equal(base[i],base[j])
            same_features=features[i].dtype==features[j].dtype and np.array_equal(features[i],features[j])
            same_multiset=np.array_equal(canonical[i],canonical[j]);same_teacher=np.array_equal(teacher[i],teacher[j])
            pairs.append({'first':uid[i],'second':uid[j],'identical_base':same_base,'identical_condition_features':same_features,'identical_condition_multiset':same_multiset,'identical_teacher':same_teacher,
                'exact_identical_input_conflicting_target':same_base and same_multiset and not same_teacher,
                'row_order_dependent_feature_RMS_not_a_shape_distance':float(np.sqrt(np.square(features[i].astype(np.float64)-features[j]).mean())),
                'mainfield_logradius_RMS':float(np.sqrt(np.square(logr[i].astype(np.float64)-logr[j]).mean())),
                'teacher_direction_RMS':float(np.sqrt(np.square(teacher[i]-teacher[j]).mean()))})
    result={'scope':'READ_ONLY_TRAIN12_EXACT_INPUT_COLLISION_DIAGNOSIS','parents':records,'pairs':pairs,'pair_count':66,
        'exact_condition_duplicate_pairs':sum(r['identical_condition_features'] for r in pairs),
        'exact_conflicting_pairs':[r for r in pairs if r['exact_identical_input_conflicting_target']],
        'permutation_canonical_duplicate_pairs':sum(r['identical_condition_multiset'] for r in pairs),
        'same_base_all_pairs':all(r['identical_base'] for r in pairs),
        'closest_five_mainfield_pairs':sorted(pairs,key=lambda r:r['mainfield_logradius_RMS'])[:5],
        'source_training_archive_sha256':file_hash(path),'source_protocol_sha256':file_hash(source/'RUN_PROTOCOL.json'),'source_code_sha256':file_hash(__file__),
        'input_uniqueness_does_not_prove_sufficient_information_or_learnability':True,'raw_order_and_permutation_invariant_input_identity_both_checked':True,'FP32_reduction_order_effects_not_treated_as_scientific_shape_information':True,'no_network_forward_or_training':True,'no_torch_import_or_GPU':True}
    write_json(out/'RESULTS.json',result);(out/'EXECUTED_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ['pair_count','exact_condition_duplicate_pairs','exact_conflicting_pairs','same_base_all_pairs','permutation_canonical_duplicate_pairs','closest_five_mainfield_pairs']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.source,a.out)