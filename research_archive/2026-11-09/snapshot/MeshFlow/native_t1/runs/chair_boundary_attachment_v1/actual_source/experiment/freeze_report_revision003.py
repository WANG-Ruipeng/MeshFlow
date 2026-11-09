"""Update only the recorded postprocessing identities after audited revision."""
import shutil
from .common import *


def run():
    folder=OUT/'engineering/report_revision003'
    manifest=OUT/'protocol/evaluation_sources_before.json'
    original=folder/'evaluation_sources_before.root_original.json'
    if original.exists():raise RuntimeError('Already frozen; no replay')
    shutil.copyfile(manifest,original)
    expected={'aggregation.py':'98f27a865061f3d551f9765ed70c0f65c17bf1a70d12a77aaadfdda73053ae06',
              'reporting.py':'a5a91bfb3527dd3ce8daf57379658c47e783be1fb05930165efa04c55f6389dd'}
    values=read_json(manifest);changed=[]
    for key in values:
        name=Path(key).name
        if name in expected:
            observed=digest(CODE/name)
            assert observed==expected[name]
            changed.append(dict(path=key,before=values[key],after=observed))
            values[key]=observed
    assert len(changed)==2
    atomic_json(manifest,values)
    atomic_json(folder/'evaluation_manifest_update.json',dict(changed=changed,model_calls=0,
        training_protocol_unchanged=True,metric_definitions_unchanged=True))
    print(changed)


if __name__=='__main__':run()
