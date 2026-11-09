# Frozen dispatch plan

Two independent authorized suffix jobs, CU and UC, each 400 forwards. Start CU first; measure its real suffix workload before considering UC. No synthetic probe, extra sampling, or training.

Preparation: RTX 5080, 16,303 MiB total; 1,561 MiB used, 14,419 MiB free; GPU utilization 16%. WSL: 24 logical CPU, 65.84 GB available RAM, load average 0.154/0.033/0.011. WSL compute-process list is empty despite Windows graphics processes; it does not certify an unused GPU. Windows query near preparation: 1,555 MiB used, 33% utilization. No unrelated process will be stopped.

Comparable prior inference: allocated 592,043,008 bytes, reserved 648,019,968 bytes; whole-device increment about 0.8–1.0 GiB, process VRAM N/A. Conservatively budget 1.5 GiB per worker, with 1.2 multiplier and at least 2,048 MiB extra headroom. Incremental slots floor((free-headroom)/(1.2*1536)), capped at one remaining authorized GPU job and measured compute/CPU load. Tensor peaks are not process VRAM.

Recheck after CU's first completed request, before UC dispatch, and at completion. If compute saturation or a fault removes the benefit/headroom, keep the remainder serial and record why. GPU time under contention is not an isolated benchmark.

CPU: first actual scoring request on one worker, then up to three independent scoring processes if observed RAM/load permits; two rendering workers. One numerical thread per process. No source Y or new score queries. GPU producer outputs remain separate and have one owner each.
