"""Synchronous microbatch OT concurrency with the original sequential RNG stream."""
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from ..io_utils import array_hash

_CAPTURE_LOCK=threading.RLock()


class OTContractError(RuntimeError):pass


def require(ok,message):
    if not ok:raise OTContractError(message)


def load_microbatch(ds,indices,pool:ThreadPoolExecutor):
    """Return original collated data after exactly one original OT per parent.

    Dataset sampling remains sequential in this calling thread. Only the pure,
    deterministic optimal_sum_numpy calls run in workers. All futures finish
    before this function returns; no prefetched data or RNG state crosses a step.
    """
    import datasets.mesh_dataset as dataset_module
    import utils.ot_utils as ot_module
    require(threading.current_thread() is threading.main_thread(),'Sequential dataset capture must run in the main thread')
    require(len(indices)==4 and len(set(indices))==4,'Exactly four distinct parents per microbatch required')
    require(getattr(pool,'_max_workers',None)==4,'This candidate has exactly four OT workers')
    require(ds.noise_sort=='ot','Only the original OT noise sorting mode is supported')
    original=ot_module.optimal_sum_numpy;captures=[];items=[];slot=[None]
    def capture(*args,**kwargs):
        require(slot[0] is not None,'OT outside a parent getitem')
        require(len(captures)==slot[0],'Multiple OT calls within one parent are not supported')
        require(len(args)==2 and set(kwargs)=={'optimal'} and kwargs['optimal'] is True,
                'Special OT call signature is unsupported')
        data,noise=args
        require(isinstance(data,np.ndarray) and isinstance(noise,np.ndarray),'OT inputs must be original NumPy arrays')
        require(data.shape==noise.shape==(512,3,3),'Only exact512 triangle OT is supported')
        require(np.issubdtype(data.dtype,np.floating) and np.issubdtype(noise.dtype,np.floating)
                and np.isfinite(data).all() and np.isfinite(noise).all(),'Invalid original OT arrays')
        captures.append({'slot':slot[0],'data':data.copy(order='K'),'noise':noise.copy(order='K'),'kwargs':kwargs.copy()})
        return noise
    with _CAPTURE_LOCK:
        require(dataset_module.optimal_sum_numpy is original,'Dataset OT binding was already replaced')
        dataset_module.optimal_sum_numpy=capture
        try:
            for position,index in enumerate(indices):
                slot[0]=position;item=ds[int(index)]
                require(len(captures)==position+1,'Missing OT call within one parent')
                require(item['num_faces']==512 and item['len']==512,'Parent face contract changed')
                require(np.array_equal(item['noise'],captures[position]['noise'].reshape(512,9)),
                        'Dataset changed noise after OT; deferred execution is unsupported')
                items.append(item)
        finally:
            dataset_module.optimal_sum_numpy=original;slot[0]=None
    # Futures are collected in submission order, never completion order.
    futures=[]
    try:
        for capture_row in captures:
            futures.append(pool.submit(original,capture_row['data'],capture_row['noise'],**capture_row['kwargs']))
        results=[future.result() for future in futures]
    finally:
        # Even failures leave no background data work crossing the caller boundary.
        for future in futures:
            try:future.result()
            except Exception:pass
    audit=[]
    for position,(item,result,capture_row) in enumerate(zip(items,results,captures)):
        require(capture_row['slot']==position and result.shape==(512,3,3),'OT result association/shape changed')
        require(np.isfinite(result).all(),'Nonfinite original OT output')
        item['noise']=result.reshape(512,9)
        audit.append({'slot':position,'index':int(indices[position]),'uid':str(np.asarray(item['uid']).item()),
            'OT_input_data_sha256':array_hash(capture_row['data']),
            'unaligned_noise_sha256':array_hash(capture_row['noise']),'aligned_noise_sha256':array_hash(result),
            'OT_kwargs':capture_row['kwargs']})
    require(all(f.done() for f in futures),'Unfinished worker at microbatch boundary')
    batch=dataset_module.collate_fn(items,max_seq_length=800)
    return batch,audit
