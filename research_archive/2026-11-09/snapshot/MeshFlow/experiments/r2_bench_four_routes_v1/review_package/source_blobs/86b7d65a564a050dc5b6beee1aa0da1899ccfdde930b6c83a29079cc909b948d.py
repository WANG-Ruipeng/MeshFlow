"""Create the independent exposure continuation implementation, before freezing."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'r2/pilots/bench_group_weight_match_v1'
NEW=Path(__file__).parent/'exposure'


def save(name,text):
    p=NEW/name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_text(encoding='utf-8')!=text:
        raise ValueError('Refuse overwriting prepared source: '+str(p))
    if not p.exists():p.write_text(text,encoding='utf-8',newline='\n')


def replace(text,before,after):
    if before not in text:raise ValueError('Source drift: '+before[:100])
    return text.replace(before,after)


def main():
    save('__init__.py','"""M plus3000, independently registered additional-compute experiment."""\n')
    save('runtime.py','from r2.pilots.bench_group_weight_match_v1.runtime import (build_model, native_execution, sample_for_training, verify_sample, VERSION)\n')
    t=(OLD/'training.py').read_text(encoding='utf-8')
    for before,after in [
        ('parents[3]','parents[4]'),
        ('bench_group_weight_match_training_v1','bench_four_routes_exposure_training_v1'),
        ('bench_group_weight_match_generator_v1','bench_four_routes_exposure_generator_v1'),
        ('1500,BASE_STEP,22500','3000,BASE_STEP,25500'),
        ('full_steps\'] != [1500]','full_steps\'] != [3000]'),
        ('dict(updates=1500,full_steps=[1500]','dict(updates=3000,full_steps=[3000]'),
        ('0 <= step <= 1500','0 <= step <= 3000'),
        ('curriculum_step=3000+step','curriculum_step=4500+step'),
        ('native_total_step=9000+step','native_total_step=10500+step'),
        ('train_forward=12000,train_backward=12000,optimizer_update=1500','train_forward=24000,train_backward=24000,optimizer_update=3000'),
        ("('bench_exposure_allocation_training_v1','B_75_25','NATIVE',1500,21000,False,False)","('bench_group_weight_match_training_v1','W_MATCHED','NATIVE',1500,22500,False,False)"),
        ("p['native_total_step'] != 9000","p['native_total_step'] != 10500"),
        ("'bench_group_weight_match_'+arm","'bench_four_routes_exposure_'+arm"),
        ('Exactly 1500 new updates from exposure B21000 to22500 required','Exactly3000 new updates from M22500 to25500 required'),
        ('Only retained exposure B21000 live/matching optimizer/RNG is allowed','Only retained M22500 live/matching optimizer/RNG is allowed'),
        ('Retained exposure B endpoint identity differs','Retained M endpoint identity differs'),
        ('Source is not the retained Native-only exposure B endpoint','Source is not the retained Native-only M endpoint'),
        ('Exact frozen W data stream','Registered continuation of the same W exposure policy'),
        ('original immutable exposure B file','original immutable M file'),
    ]:t=replace(t,before,after)
    save('training.py',t)


if __name__=='__main__':main()
