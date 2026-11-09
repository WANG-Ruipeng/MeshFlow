"""Presentation-only repair of an inherited W/M caption; geometry is unchanged."""
import argparse, types
from . import prepare_review as P

R=P.R
OLD=' | Blue C / gray BULK / green BRIDGE\nW original wide HIGH; M group-weight matched. Same frozen noise, frame and sections.'
NEW=' | Blue C / gray BULK / green BRIDGE\nM frozen; EXPOSURE +3000 Native updates. Same frozen noise, frame and sections.'
assert R.one.__code__.co_consts.count(OLD)==1
constants=tuple(NEW if x==OLD else x for x in R.one.__code__.co_consts)
corrected=types.FunctionType(R.one.__code__.replace(co_consts=constants),vars(R),R.one.__name__)

def one(job):
    return corrected(job)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    P.prepare()
    R.E.immutable(R.OUT/'visual_review/caption_correction_v2.json',dict(
        status='PRESENTATION_ONLY',old_caption=OLD,new_caption=NEW,
        method='Replace exactly one string constant in the frozen rendering function; all image geometry, bounds, colors and sections unchanged',
        original_renderer_sha256=R.E.sha(R.__file__),adapter_sha256=R.E.sha(__file__),
        changed_code_bytes=False,changed_constant_count=1,model_calls=0,scoring_calls=0))
    R.one=one
    result=P.render(a.workers)
    print({k:v for k,v in result.items() if k not in ('files','images')})

if __name__=='__main__':main()
