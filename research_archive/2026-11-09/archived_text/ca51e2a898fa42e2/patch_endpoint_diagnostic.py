from pathlib import Path
p=Path('/mnt/e/MeshFlow/.worktrees/a-layout-value-20260924/experiments/a_radial/layout_endpoint/smoke.py')
s=p.read_text(encoding='utf-8-sig')
s=s.replace('from .train import train_run,read_json,write_json','from .train import train_run,read_json,write_json,load_train_archive')
anchor='\n\ndef main():'
helper='''

def single_object_diagnostic(run,device='cuda'):
    """Fixed 200-update C1 diagnostic; its weights never initialize formal arms."""
    from ..exporter import RadialSupport
    from ..legacy.networks import DomainVelocity
    from ..layout_student.deployment import deploy
    from ..io_utils import array_hash
    run=Path(run).resolve();protocol=read_json(run/'RUN_PROTOCOL.json');d=protocol['diagnostic']
    assert d['arm']=='c1' and d['seed']==10 and d['updates']==200
    assert not (run/'diagnostics/SINGLE_OBJECT_DIAGNOSTIC.json').exists()
    uids,data,scale,_=load_train_archive(run,protocol);index=uids.index(d['uid'])
    initial_path=run/protocol['paths']['initial_checkpoints']['10']
    support=RadialSupport();snapshots={}
    torch.set_num_threads(1)
    def measure(checkpoint,label):
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=DomainVelocity().cpu();model.load_state_dict(state['model'],strict=True)
        model.eval().requires_grad_(False)
        deployed=deploy(model,data['logr'][index],support,data['directions0'][index],data['faces'])
        mesh=deployed['raw_mesh'];u=mesh['u'];target=data['directions1'][index]
        endpoint_mse=float(np.mean((u-target)**2))
        angle=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u,target),axis=-1),(u*target).sum(-1)))
        out=run/'diagnostics'/('single_object_deployment_'+label+'.npz')
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('xb') as stream:np.savez_compressed(stream,**mesh)
        snapshots[label]={'checkpoint_sha256':file_sha256(checkpoint),'model_sha256':model_sha256(state['model']),
            'endpoint_direction_mse':endpoint_mse,'normalized_endpoint_loss':endpoint_mse/scale,
            'endpoint_angular_rmse_deg':float(np.sqrt(np.mean(angle**2))),
            'quality':deployed['raw_quality'],'qualified':deployed['raw_qualification']['qualified'],
            'failure_reasons':deployed['raw_qualification']['failure_reasons'],
            'audit':deployed['raw_audit'],'timing':deployed['timing'],
            'mesh_file':out.relative_to(run).as_posix(),'mesh_npz_sha256':file_sha256(out),
            'stored_array_hash':array_hash(mesh['vertices'],mesh['faces'])}
    measure(initial_path,'initial')
    report=train_run(run,'c1',10,200,device=device,checkpoint_every=250,diagnostic_uid=d['uid'])
    measure(Path(report['checkpoint']),'final')
    result={'scope':'SINGLE_TRAINING_OBJECT_DIAGNOSTIC_ONLY','uid':d['uid'],'arm':'c1','seed':10,
        'updates':200,'selection_rule':d['selection_rule'],'endpoint_scale_squared':scale,
        'scale_from_all_training_parents':True,'training_report':report,'snapshots':snapshots,
        'final_over_initial_deployment_endpoint_mse':snapshots['final']['endpoint_direction_mse']/snapshots['initial']['endpoint_direction_mse'],
        'deployed_geometry_measured_with_original_cpu_32_guard':True,
        'diagnostic_weights_used_for_formal_training':False,'additional_updates_after_200':0}
    write_json(run/'diagnostics/SINGLE_OBJECT_DIAGNOSTIC.json',result)
    print(json.dumps({'diagnostic_uid':d['uid'],'updates':200,
        'initial_endpoint_loss':snapshots['initial']['normalized_endpoint_loss'],
        'final_endpoint_loss':snapshots['final']['normalized_endpoint_loss'],
        'final_over_initial_deployment_endpoint_mse':result['final_over_initial_deployment_endpoint_mse'],
        'training_seconds':report['elapsed_training_seconds']},indent=2),flush=True)
    return result
'''
assert anchor in s;s=s.replace(anchor,helper+anchor)
s=s.replace("p.add_argument('--out',required=True,type=Path);a=p.parse_args();smoke_run(a.source_run,a.out)", "p.add_argument('--out',type=Path);p.add_argument('--diagnostic',action='store_true');a=p.parse_args()\n    if a.diagnostic:single_object_diagnostic(a.source_run)\n    else:\n        if a.out is None:p.error('--out required for isolated resume smoke')\n        smoke_run(a.source_run,a.out)")
p.write_text(s,encoding='utf-8')
