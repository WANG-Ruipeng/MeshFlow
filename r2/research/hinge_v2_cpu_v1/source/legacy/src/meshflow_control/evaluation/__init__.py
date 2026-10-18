"""CPU-only frozen metric adapters. No model is imported here."""
def evaluate(manifest,out,workers=2):
    from .evaluate import evaluate as run
    return run(manifest,out,workers=workers)
def render(manifest,out,**kwargs):
    from .render import render as run
    return run(manifest,out,**kwargs)
