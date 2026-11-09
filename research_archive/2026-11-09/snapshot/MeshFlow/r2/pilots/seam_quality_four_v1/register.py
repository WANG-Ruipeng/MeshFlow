"""Freeze the bounded four-direction experiment without model calls."""
from pathlib import Path
import ast
import datetime
import json
from . import eval_common as C

def main():
    root=C.OUT; base=C.ROOT/'experiments/r2_seam_slot_c200_v1'
    original=C.read(base/'registration.json')
    manifest=C.read(base/'eval_manifest.json')
    manifest.update(schema='seam_quality_four_eval_v1',arms=list(C.ARMS),
        original_manifest_path=str(base/'eval_manifest.json'),original_manifest_sha256=C.sha(base/'eval_manifest.json'))
    C.write(root/'eval_manifest.json',manifest)
    source=base/'checkpoints/SEAM_SLOT_C200_step200.pt'
    assert C.sha(source)=='c0c26ad26d2123062d3fd38039227d6cd05e269b5ca16e312602c45266fb3d65'
    checkpoint=original['d0_checkpoint']
    assert C.sha(checkpoint['path'])==checkpoint['sha256']
    code={}
    for name in ('generate.py','postprocess.py','evaluate.py','eval_common.py','register.py'):
        p=Path(__file__).parent/name
        ast.parse(p.read_text(encoding='utf-8'))
        code[str(p)]=C.sha(p)
    for folder in ('r2/prototypes/seam_slot_c200_v1','r2/src/meshflow_control/seam'):
        code.update({str(p):C.sha(p) for p in (C.ROOT/folder).glob('*.py')})
    inputs={str(base/'eval_manifest.json'):C.sha(base/'eval_manifest.json')}
    for q in manifest['requests']:
        for key in ('inference_npz','metadata_npz','noise_npz'):
            inputs[q[key]]=C.sha(q[key])
        inputs[q['query_descriptor']['path']]=q['query_descriptor']['sha256']
    for q in manifest['conditions']:
        inputs[q['labels_npz']]=C.sha(q['labels_npz'])
    reg=dict(schema='R2_SEAM_QUALITY_FOUR_V1',status='FROZEN',utc=C.utc(),
        arms=list(C.ARMS),updates_per_arm=800,cumulative_seam_endpoint=1000,
        source_global_step=12200,final_global_step=13000,batch=8,microbatch=1,
        source_checkpoint=str(source),source_checkpoint_sha256=C.sha(source),
        same_live_generator_matching_AdamW_head_optimizer_RNG=True,
        d0_checkpoint=checkpoint,eval_manifest_sha256=C.sha(root/'eval_manifest.json'),
        new_endpoint_rollouts=48,generator_Euler_steps=50,new_D0_forwards=48,
        all_RAW_before_D0=True,CPU_antiflatten_POST=16,CPU_redirection_POST=16,
        training=dict(LONG='unchanged SEAM continuation',ANTI_COVER='same continuation plus fixed finite-C loss lambda=1; normalized squared distance; t>=0.5; tau=.02L; two probes',
            LOCAL_NOISE='same continuation; occupied seam third corner x0 += corresponding C-edge midpoint after OT; sigma1; unchanged bulk and matching'),
        occupancy='Continue the same head and AdamW in all arms; C-only predicted occupancy at inference; zero threshold; no oracle or all-on substitution',
        CPU_controls='Original predicted-SEAM200 fixed caches only. Anti-flatten and safe redirection run separately; no combination with trained endpoints.',
        metrics=['RAW/POST original CF illegal affected-free fraction','RAW/POST original FF illegal affected-free fraction','I union/cross/duplicate original frozen C queries','role-partitioned FF counts','legal CF count/length','free-only seam-to-bulk reachable area','oriented reachable free area','small angles and q','deleted faces/reasons and retained area','actual AI_PROXY whole-body/fragment/layer/join review'],
        comparison='ANTI_COVER and LOCAL_NOISE versus equal-budget LONG; LONG versus cached SEAM200 is continuation effect including common head continuation. Original FULL4_C200 is historical unequal-budget reference.',
        success='No guaranteed promotion or p-value threshold. Inspect full paired FF/I means, medians, tails, per-parent/noise changes and connectivity/area/visual costs. No unique-GT style ranking.',
        boundary='Four seen TRAIN C, four reused fixed noises each. Not DEV or new-parent generalization; no claim that published scorer scores certify visual quality.',
        stops='Only three fixed800-update arms and registered endpoint panel. Preserve failure ledgers, no silent retry or budget extension. No default switch, cleanup, commit or push.',
        resources=dict(generator_estimate_mib=2600,D0_estimate_mib=3000),
        previous_routes=['R31 shared slots: bounded connection gains with FF/I costs','CREF: conditional-origin shifts are not an untested general idea; this test shifts only structured seam tips','R10/partial edge binding: avoid arbitrary-radius clusters, index conflicts and unrecorded tearing','RBF/co-restoration: no guarantee D0 proposal is conflict-free','60deg/135deg normal gates: rejected; finite projected coverage is separately source-audited'])
    C.write(root/'registration.json',reg)
    C.write(root/'execution_freeze.json',dict(status='FROZEN',utc=C.utc(),code_hashes=code,input_hashes=inputs,
        registration_sha256=C.sha(root/'registration.json'),scope='Endpoint inference/scoring source and fixed evaluation inputs; training has separate execution freeze',
        protected_weights={str(source):reg['source_checkpoint_sha256'],checkpoint['path']:checkpoint['sha256']}))
    print(json.dumps(dict(status='FROZEN',new_RAW=48,new_D0=48,training_updates=2400)),flush=True)

if __name__=='__main__':main()
