"""Fixed CPU FP64 quadrature diagnostics, never a replacement training loss.

Production n=5 calls native_t1.losses.edge.edge_loss unchanged. The denser
257/1025 trapezoidal references are diagnostic only; analytic witnesses remain
authoritative if these fixed grids have not converged. No model or OT exists.
"""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from native_t1.losses.edge import prepare_edge_target, edge_loss, centered_segment_distance_sq

RESOLUTIONS=(5,257,1025)
FINITE_DIFFERENCE_STEP=1e-6
POLICY=dict(dtype='CPU_FP64',production_points=5,dense_points=[257,1025],
    derivative='torch.autograd.grad with respect to the named scalar configuration parameter',
    finite_difference_step=FINITE_DIFFERENCE_STEP,finite_difference_atol=3e-7,finite_difference_rtol=2e-6,
    analytic_five_point_atol=1e-12,analytic_five_point_rtol=1e-12,
    dense_convergence_atol=1e-12,dense_convergence_rtol=1e-3,
    finite_difference_projection_switch='skip if either direction changes clipping region across +/-h; no step retuning',
    dense_nonconvergence='record only; analytic examples remain valid, never change training loss',
    predicted_segment_denominator_epsilon=1e-12,production_loss_modified=False,
    model_calls=0,OT_calls=0,optimizer_updates=0)
CASES=tuple(
 [dict(name=f'collinear_delta_{x}',kind='collinear',parameter_name='delta',parameter=x) for x in (.001,.01,.05,.1,.2)]
 +[dict(name=f'perpendicular_h_{x}',kind='perpendicular',parameter_name='h',parameter=x) for x in (.001,.01,.05,.1,.2)]
 +[dict(name=f'centered_length_ratio_{x}',kind='length',parameter_name='predicted_length_over_ell',parameter=x) for x in (.6,.8,1.2,1.4)]
 +[dict(name=f'centered_rotation_{x}',kind='rotation',parameter_name='theta_radians',parameter=x) for x in (.1,.3,.7)]
 +[dict(name='rotation_endpoint_exchange',kind='rotation',parameter_name='theta_radians',parameter=.3,exchange=True)]
 +[dict(name='symmetric_midpoint_collapse',kind='collapse',parameter_name='vertical_midpoint_translation_h',parameter=0.)])


def new_counts():
    return dict(loss_attempts=0,loss_returns=0,production_five_loss_calls=0,dense_loss_calls=0,
        parameter_autograd_attempts=0,parameter_autograd_returns=0,finite_difference_evaluations=0,
        finite_difference_comparisons=0,finite_difference_skipped_at_switch=0,projection_regime_checks=0,
        target_preparations=0,analytic_reference_evaluations=0,model_forwards=0,OT_calls=0,optimizer_updates=0)


def _inc(ledger,key,n=1):
    if ledger is not None:ledger[key]=ledger.get(key,0)+n


def fixed_target(ell=1.,bbox_L=1.,ledger=None):
    _inc(ledger,'target_preparations')
    vertices=torch.tensor([[0.,0.,0.],[ell,0.,0.],[0.,ell,0.],[.3*ell,-ell,.2*ell]],dtype=torch.float64)
    free_ids=torch.tensor([[1,0,3]]);known_ids=torch.tensor([[0,1,2]])
    return prepare_edge_target(dict(free_gt=vertices[free_ids],known=vertices[known_ids],free_ids=free_ids,
                                   known_ids=known_ids,bbox_L=bbox_L))


def prediction(case,parameter,ell=1.):
    zero=parameter*0.;kind=case['kind']
    if kind=='collinear':
        left=torch.stack((parameter,zero,zero));right=torch.stack((parameter+ell,zero,zero))
    elif kind=='perpendicular':
        left=torch.stack((zero,parameter,zero));right=torch.stack((zero+ell,parameter,zero))
    elif kind=='length':
        left=torch.stack((.5*ell*(1-parameter),zero,zero));right=torch.stack((.5*ell*(1+parameter),zero,zero))
    elif kind=='rotation':
        direction=torch.stack((parameter.cos(),parameter.sin(),zero))*.5*ell
        midpoint=torch.stack((zero+.5*ell,zero,zero));left=midpoint-direction;right=midpoint+direction
    elif kind=='collapse':
        left=torch.stack((zero+.5*ell,parameter,zero));right=left
    else:raise ValueError('Unknown fixed diagnostic case')
    if case.get('exchange',False):left,right=right,left
    third=torch.stack((zero+.3*ell,zero-ell,zero+.2*ell))
    return torch.stack((right,left,third))[None]


def _queries(pred,target,n):
    free=pred/target['bbox_L']
    segment=free[target['free_faces'][:,None],target['free_corners']]
    known=target['GT_endpoints']
    s=torch.linspace(0.,1.,n,dtype=pred.dtype,device=pred.device)[None,:,None]
    return segment,known,(1-s)*known[:,:1]+s*known[:,1:],(1-s)*segment[:,:1]+s*segment[:,1:]


def loss_at(case,parameter,n,target,ledger=None,ell=1.):
    """FP64 scalar; production five points call the frozen public loss directly."""
    _inc(ledger,'loss_attempts')
    if parameter.dtype!=torch.float64 or parameter.device.type!='cpu' or n not in RESOLUTIONS:
        raise ValueError('Only registered CPU FP64 resolutions')
    pred=prediction(case,parameter,ell)
    if n==5:
        _inc(ledger,'production_five_loss_calls');value=edge_loss(pred,target)['loss']
    else:
        _inc(ledger,'dense_loss_calls')
        segment,known,qknown,qpred=_queries(pred,target,n)
        weights=pred.new_ones(n)/(n-1);weights[0]*=.5;weights[-1]*=.5
        a=(centered_segment_distance_sq(qknown,segment)*weights[None]).sum(-1)
        b=(centered_segment_distance_sq(qpred,known)*weights[None]).sum(-1)
        lengths=target['GT_lengths']
        value=.5*((lengths*a).sum()/lengths.sum()+(lengths*b).sum()/lengths.sum())
    if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite CPU quadrature')
    _inc(ledger,'loss_returns')
    return value


def _projection_regions(case,value,n,target,ledger,ell=1.):
    _inc(ledger,'projection_regime_checks')
    p=torch.tensor(value,dtype=torch.float64)
    segment,known,qknown,qpred=_queries(prediction(case,p,ell),target,n)
    answer=[]
    for points,edges in ((qknown,segment),(qpred,known)):
        midpoint=(edges[:,0]+edges[:,1])*.5;direction=edges[:,1]-edges[:,0]
        parameter=((points-midpoint[:,None])*direction[:,None]).sum(-1)/direction.square().sum(-1).clamp_min(1e-12)[:,None]
        answer.append(torch.where(parameter<-.5,-1,torch.where(parameter>.5,1,0)))
    return torch.cat(answer,dim=1)


def analytic(case,ell=1.,bbox_L=1.):
    """Independent closed-form continuous and five-point witnesses."""
    x=float(case['parameter']);kind=case['kind'];scale=bbox_L*bbox_L
    if kind=='collinear':
        if not 0<x<ell/4:raise ValueError('Registered collinear analytic regime required')
        continuous=x**3/(3*ell*scale);derivative=x*x/(ell*scale)
        five=x*x/(8*scale);five_derivative=x/(4*scale)
        formula='D_cont=delta^3/(3*ell*L^2); dD/ddelta=delta^2/(ell*L^2); D5=delta^2/(8*L^2)'
    elif kind=='perpendicular':
        continuous=five=x*x/scale;derivative=five_derivative=2*x/scale
        formula='D_cont=D5=h^2/L^2; dD/dh=2h/L^2'
    elif kind=='length':
        factor=ell*ell/scale
        if x<1:
            continuous=factor*(1-x)**3/24;derivative=-factor*(1-x)**2/8
        else:
            continuous=factor*(x-1)**3/(24*x);derivative=factor*(x-1)**2*(2*x+1)/(24*x*x)
        if not .5<x<2:raise ValueError('Registered five-point length regime required')
        five=factor*(x-1)**2/32;five_derivative=factor*(x-1)/16
        formula='centered k<1: ell^2*(1-k)^3/(24L^2); k>1: ell^2*(k-1)^3/(24kL^2); D5=ell^2*(k-1)^2/(32L^2)'
    elif kind=='rotation':
        factor=ell*ell/scale
        continuous=factor*math.sin(x)**2/12;derivative=factor*math.sin(2*x)/12
        five=factor*3*math.sin(x)**2/32;five_derivative=factor*3*math.sin(2*x)/32
        formula='centered equal lengths: D_cont=ell^2*sin(theta)^2/(12L^2); D5=3ell^2*sin(theta)^2/(32L^2)'
    elif kind=='collapse':
        continuous=ell*ell/(24*scale)+x*x/scale;five=3*ell*ell/(64*scale)+x*x/scale
        derivative=five_derivative=2*x/scale
        formula='centered point: D_cont=ell^2/(24L^2)+h^2/L^2; D5=3ell^2/(64L^2)+h^2/L^2; dD/dh=0 at h=0'
    else:raise ValueError(kind)
    return dict(loss=continuous,parameter_derivative=derivative,five_point_loss=five,
                five_point_parameter_derivative=five_derivative,formula=formula)


def _relative(actual,reference):
    return None if reference==0 else abs(actual-reference)/abs(reference)


def _close(a,b,atol=1e-12,rtol=1e-3):return abs(a-b)<=atol+rtol*abs(b)


def analyze_case(case,target,ledger,ell=1.,bbox_L=1.):
    _inc(ledger,'analytic_reference_evaluations');reference=analytic(case,ell,bbox_L)
    records={};h=FINITE_DIFFERENCE_STEP
    for n in RESOLUTIONS:
        parameter=torch.tensor(case['parameter'],dtype=torch.float64,requires_grad=True)
        value=loss_at(case,parameter,n,target,ledger,ell)
        _inc(ledger,'parameter_autograd_attempts');derivative=torch.autograd.grad(value,parameter)[0]
        _inc(ledger,'parameter_autograd_returns')
        if not bool(torch.isfinite(derivative)):raise FloatingPointError('Nonfinite CPU quadrature derivative')
        lo=_projection_regions(case,case['parameter']-h,n,target,ledger,ell)
        hi=_projection_regions(case,case['parameter']+h,n,target,ledger,ell)
        switches=int(torch.count_nonzero(lo!=hi))
        fd=dict(status='NOT_RUN_PROJECTION_SWITCH',switching_quadrature_features=switches,step=h)
        if switches==0:
            plus=loss_at(case,torch.tensor(case['parameter']+h,dtype=torch.float64),n,target,ledger,ell)
            minus=loss_at(case,torch.tensor(case['parameter']-h,dtype=torch.float64),n,target,ledger,ell)
            _inc(ledger,'finite_difference_evaluations',2);_inc(ledger,'finite_difference_comparisons')
            numeric=float((plus-minus)/(2*h));error=abs(numeric-float(derivative))
            if not _close(numeric,float(derivative),atol=3e-7,rtol=2e-6):raise AssertionError('Finite difference mismatch')
            fd.update(status='PASS',derivative=numeric,absolute_error=error)
        else:_inc(ledger,'finite_difference_skipped_at_switch')
        loss_value=float(value.detach());derivative_value=float(derivative)
        if n==5 and (not _close(loss_value,reference['five_point_loss'],atol=1e-12,rtol=1e-12) or not _close(derivative_value,reference['five_point_parameter_derivative'],atol=1e-12,rtol=1e-12)):
            raise AssertionError('Production five-point result differs from independent closed form')
        records[str(n)]=dict(loss=loss_value,parameter_derivative=derivative_value,
            loss_absolute_error_to_continuous=abs(loss_value-reference['loss']),
            loss_relative_error_to_continuous=_relative(loss_value,reference['loss']),
            derivative_absolute_error_to_continuous=abs(derivative_value-reference['parameter_derivative']),
            derivative_relative_error_to_continuous=_relative(derivative_value,reference['parameter_derivative']),
            finite_difference=fd)
    coarse,fine=records['257'],records['1025']
    loss_converged=_close(fine['loss'],reference['loss']) and _close(fine['loss'],coarse['loss'])
    derivative_converged=_close(fine['parameter_derivative'],reference['parameter_derivative']) and _close(fine['parameter_derivative'],coarse['parameter_derivative'])
    convergence=dict(status='CONVERGED_AT_FIXED_TOLERANCE' if loss_converged and derivative_converged else 'NOT_CONVERGED_AT_FIXED_GRID',
        loss_converged=loss_converged,parameter_derivative_converged=derivative_converged,
        loss_1025_minus_257=fine['loss']-coarse['loss'],derivative_1025_minus_257=fine['parameter_derivative']-coarse['parameter_derivative'],
        no_training_change=True)
    return dict(**case,ell=ell,bbox_L=bbox_L,coordinate_normalization_count=1,
                analytic=reference,quadrature=records,dense_convergence=convergence)


def make_figure(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for column,(kind,title) in enumerate((('collinear','Collinear equal-length translation'),('perpendicular','Perpendicular translation'),('rotation','Centered rotation'))):
        group=[row for row in rows if row['kind']==kind and not row.get('exchange',False)]
        x=[row['parameter'] for row in group]
        for row_index,key in enumerate(('loss','parameter_derivative')):
            ax=axes[row_index,column]
            ax.plot(x,[row['analytic'][key] for row in group],'k-',label='Continuous analytic',linewidth=2)
            for n,style in ((5,'o--'),(257,'s:'),(1025,'^-.')):
                ax.plot(x,[row['quadrature'][str(n)][key] for row in group],style,label=f'{n} points',markersize=4)
            if kind in ('collinear','perpendicular'):ax.set_xscale('log');ax.set_yscale('log')
            ax.set_xlabel(group[0]['parameter_name']);ax.set_ylabel('Loss' if row_index==0 else 'Parameter derivative')
            ax.grid(True,alpha=.25)
            if row_index==0:ax.set_title(title)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Fixed FP64 quadrature witnesses; production five-point loss is unchanged')
    fig.savefig(path,dpi=160);plt.close(fig)


def run_diagnostics(out):
    """One explicit bounded offline execution; refuses existing artifacts."""
    out=Path(out);json_path=out/'quadrature_diagnostic.json';figure=out/'quadrature_loss_derivative.png'
    if json_path.exists() or figure.exists():raise FileExistsError('Quadrature diagnostic already exists')
    if not out.is_dir():raise FileNotFoundError('Root must create the isolated run directory first')
    if torch.cuda.is_initialized():raise RuntimeError('This diagnostic must not initialize/use CUDA')
    started=time.perf_counter();counts=new_counts();target=fixed_target(ledger=counts)
    rows=[analyze_case(case,target,counts) for case in CASES]
    normal=next(row for row in rows if row['name']=='centered_rotation_0.3')
    exchanged=next(row for row in rows if row['name']=='rotation_endpoint_exchange')
    exchange={str(n):dict(loss_absolute_difference=abs(normal['quadrature'][str(n)]['loss']-exchanged['quadrature'][str(n)]['loss']),
        derivative_absolute_difference=abs(normal['quadrature'][str(n)]['parameter_derivative']-exchanged['quadrature'][str(n)]['parameter_derivative'])) for n in RESOLUTIONS}
    if any(value>1e-12 for record in exchange.values() for value in record.values()):raise AssertionError('Endpoint exchange changed loss/parameter derivative')
    report=dict(status='PASS_ANALYTIC_WITNESSES',policy=POLICY,cases=rows,counts=counts,endpoint_exchange=exchange,
        dense_not_converged=[row['name'] for row in rows if row['dense_convergence']['status']!='CONVERGED_AT_FIXED_TOLERANCE'],
        interpretation='Five-point and continuous finite-edge losses can differ substantially; no conclusion about denser-loss generation quality is established.',
        collapse_limitation='At exact symmetric midpoint collapse, positive loss and zero gradient remain; no epsilon/anchor modification.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('edge_metric.py'),Path(__file__).with_name('test_math.py'))},
        CUDA_initialized=False,geometry_seconds=time.perf_counter()-started)
    figure_start=time.perf_counter();make_figure(rows,figure)
    report['figure']=dict(path=str(figure.resolve()),sha256=hashlib.sha256(figure.read_bytes()).hexdigest(),seconds=time.perf_counter()-figure_start)
    report['total_seconds']=time.perf_counter()-started
    with json_path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    return report
