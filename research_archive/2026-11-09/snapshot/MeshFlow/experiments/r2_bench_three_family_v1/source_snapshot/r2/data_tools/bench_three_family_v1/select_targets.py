"""Freeze three outcomes before task/head assessment; failed targets retained."""
from .common import *

def main():
    protocol=freeze();rows=[]
    for p in protocol['parents']:
        uid=p['parent_id'];method='shared_boundary' if uid.startswith('945214') else 'part_rebuild'
        r=read(OUT/'targets'/uid/method/'receipt.json')
        for path,h in r['hashes'].items():assert sha(path)==h
        rows.append(dict(r,proposed_condition_count=2,target_version_id=VERSION+'_'+method,
          source_parent=p['source'],inference_or_training_ready=False,
          outcome=('PENDING_VISUAL_AND_PRODUCTION' if r['geometry_pass'] and r['current_N_pass'] else 'BLOCKED_N' if r['geometry_pass'] else 'HELD_GEOMETRY')))
    save(OUT/'part_recipes/945214_not_executed.json',dict(parent_prefix='945214',status='NO_STRUCTURE_SAFE_PART_RECIPE_FROZEN',
      evidence='All448 source faces axis-aligned;216 patch junction vertices; strict shared-boundary method removes0; source has8 open boundary loops. Independent box decomposition would alter connected assembly and loops.',
      rejected_recipe='Coordinate plane snapping could only combine Y=.254858/.261928 at the conservative per-axis range .0025L/sqrt3; these planes may define thickness or assembly gap. No supported192-face reduction demonstrated.',
      action='Retain the actual448-face unchanged-surface proposal, markNblocked, do not erase parts/holes or run a speculative destructive fallback.',
      interpretation='Not a proof that a skilled retopology or another representation cannot fit256. No parameter search performed.',
      model_calls=0))
    save(OUT/'selected_targets.json',dict(rows=rows,parents=3,max_conditions=6,selection_rule='Fixed source/recipe outcomes before conditions/head; no selection on production mask or generation results.',
      no_production_N_change=True,training_ready=False))
    event('THREE_TARGET_OUTCOMES_FROZEN',rows=[dict(parent=r['parent_id'],N=r['N'],outcome=r['outcome']) for r in rows])
    print(json.dumps([dict(parent=r['parent_id'][:6],N=r['N'],outcome=r['outcome']) for r in rows]))

if __name__=='__main__':main()
