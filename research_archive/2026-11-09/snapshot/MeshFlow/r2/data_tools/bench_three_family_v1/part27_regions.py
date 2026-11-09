"""Region, not exact-face, ancestry for the fixed 27 shell collapse proposal."""
from .common import OUT, read, save, sha


def run():
    folder=OUT/'fallback27'
    recipe=read(folder/'RECIPE.json');lineage=read(folder/'lineage.json')
    if lineage['recipe_sha256']!=sha(folder/'RECIPE.json'):raise ValueError('Recipe identity differs')
    regions={i:{i} for i in range(recipe['source_N'])}
    shell=set(recipe['component_source_face_ids'])
    for action in lineage['audit']:
        if action['status']!='ACCEPT':continue
        removed=action['removed_source_face_ids'];receivers=action['affected_source_face_ids']
        if not receivers:raise ValueError('No same-component recipient region')
        assert set(removed+receivers)<=shell
        inherited=set().union(*(regions[i] for i in removed))
        for i in receivers:regions[i]|=inherited
        for i in removed:regions.pop(i)
    ids=lineage['output_face_to_source_face']
    assert set(ids)==set(regions)
    mapped=[sorted(regions[i]) for i in ids]
    assert set().union(*(set(v) for v in mapped))==set(range(recipe['source_N']))
    for i,r in zip(ids,mapped):
        assert (set(r)<=shell) if i in shell else r==[i]
    result=dict(schema='BENCH27_COLLAPSE_REGION_LINEAGE_V1',
        output_face_source_region_ids=mapped,
        interpretation='Every source face accounted in same-part neighborhoods. Regions overlap and are NOT exact face identity, a geometric area partition, or exact surface correspondence.',
        all_original_faces_covered=True,cross_component_mapping=False,
        collapsed_face_transfer='Each removed face lineage transfers to every surviving directly affected shell face; subsequent collapse propagates it.',
        protected={str(folder/'RECIPE.json'):sha(folder/'RECIPE.json'),str(folder/'lineage.json'):sha(folder/'lineage.json'),__file__:sha(__file__)})
    save(folder/'region_lineage.json',result)
    return result


if __name__=='__main__':
    result=run();print('PASS: all source faces covered by explicitly non-exact same-component regions')
