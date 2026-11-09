from .orchestrate import host_ready


def test_host_gate_requires_compilation_finished_and_memory_and_idle():
    p = {'host_gate':{'minimum_free_physical_GiB':12,'minimum_free_commit_GiB':12,'maximum_host_cpu_percent':25}}
    good = {'compiler_process_count':0,'free_physical_GiB':40,'free_commit_GiB':40,'cpu_percent':2}
    assert host_ready(good,p)
    for key,value in [('compiler_process_count',1),('free_physical_GiB',11),('free_commit_GiB',11),('cpu_percent',25)]:
        assert not host_ready({**good,key:value},p)