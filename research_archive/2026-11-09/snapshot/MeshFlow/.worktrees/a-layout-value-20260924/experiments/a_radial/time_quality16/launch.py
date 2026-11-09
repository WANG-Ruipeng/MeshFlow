"""Record the Python startup-to-ready interval without importing numeric libraries."""
import os
import runpy
import time
os.environ['TTQ_PROCESS_START_MONOTONIC'] = str(time.perf_counter())
os.environ['CUDA_VISIBLE_DEVICES'] = ''
runpy.run_module('experiments.a_radial.time_quality16.orchestrate', run_name='__main__')