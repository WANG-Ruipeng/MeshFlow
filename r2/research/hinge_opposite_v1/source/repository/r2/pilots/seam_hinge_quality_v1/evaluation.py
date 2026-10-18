"""Private adapters around preserved CPU scorers; no model dispatch."""
from pathlib import Path
from r2.pilots.seam_quality_wave1_wave2_v1.evaluation import (
    C, private_functions, private_module, dispatch_cpu)
ROOT=C.ROOT
OUT=ROOT/'experiments/r2_seam_hinge_quality_v1'
ARMS=('CLEAN_EDGE_COH','SMOOTH_PRIOR')
WAVES={'main':ARMS}
HIST=ROOT/'experiments/r2_seam_quality_wave1_wave2_v1'

def requests():
    qq=C.read(OUT/'protocol.json')['evaluation']['requests']
    assert len(qq)==28 and len({q['request_id'] for q in qq})==28
    assert sum(q['split']=='TRAIN' for q in qq)==16
    return qq

def source_paths():return [Path(__file__)]

def evalroot(label):return HIST if label in ('PLAIN_S1000','PLAIN_S2000') else OUT

def bindroot(label):return HIST/'binding/wave1' if label in ('PLAIN_S1000','PLAIN_S2000') else OUT/'binding/main'

def mesh_path(label,rid,stage):
    if stage=='BOUND015':return bindroot(label)/'cells'/(label+'_BIND_015')/rid/'post.npz'
    folder=evalroot(label)/'evaluation'/label/rid
    return folder/'raw/raw.npz' if stage=='RAW' else folder/'post/proposal.npz' if stage=='D0' else folder/'post/pre_geo.npz' if stage=='RESTORED' else folder/'post/post.npz'

def score_path(label,rid,stage):
    if stage=='BOUND015':return bindroot(label)/'cells'/(label+'_BIND_015')/rid/'score.json'
    return evalroot(label)/'evaluation/scores'/label/rid/stage/'score.json'
