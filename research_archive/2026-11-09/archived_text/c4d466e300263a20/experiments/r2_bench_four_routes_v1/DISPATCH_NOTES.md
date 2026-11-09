# Actual dispatch decisions

- Initial Windows sandbox query could not expose CIM or process VRAM. The actual
  WSL resource snapshot was used for dispatch: RTX5080 16,303MiB, free13,360MiB,
  24 logical CPUs, roughly61GiB RAM available. No unrelated processes stopped.
- M+3000 started alone with conservative estimated worker5,200MiB and required
  free8,288MiB including2,048MiB extra headroom. Representative updates1..27
  include backward/AdamW. Device usage rose from~2,606 to7,155MiB. This difference
  is observational with background processes, not certified per-process VRAM.
- After27 updates, free8,825MiB and GPU utilization20% permitted one SEAM worker;
  it used the same5,200MiB estimate. Both run with separate input/optimizer/RNG.
- After both trained >100 updates, device usage~10,827MiB, free5,153MiB and GPU
  utilization64–71%. The comparable completed M46 Euler50 observer recorded
  peak3,619MiB total against initial~2,546MiB background. For Bestof sampling use
  conservative2,200MiB incremental estimate (not inferred from weight size),
  requiring4,688MiB free including20% margin and2,048MiB headroom. Start at most
  one short sampling worker and remeasure; no additional GPU task while it runs.
- Hybrid scorer actually selected1 CPU worker through the inherited resource
  helper. Its first case used~122MiBRSS. Do not report the requested cap2 as an
  actual worker count. Rendering can overlap independently under bounded threads.

- Bestof's46 new RAW outputs finished in247.12 seconds while both training
  workers were active. Its observed whole-device peak was below the16GiB limit
  with the2GiB reserved headroom retained. This is concurrent end-to-end timing,
  not a standalone sampling benchmark. Selection requires13 new D0 outputs.
- At10:14UTC, both training workers were active; freeVRAM5,062MiB, utilization56%,
  RAM available~50.9GiB and CPU usage12.4%. The single-D0 estimate3,500MiB requires
  6,248MiB including margin/headroom. D0 was queued rather than forced to launch.
- HYBRID's one-worker choice was traced to resource-schema incompatibility:
  the inherited helper expects memory/cpu_logical/loadavg while the new caller
  supplies RAM_available/CPU_cores/load. It fell back to one worker, not a measured
  shortage. The running frozen pool was left intact to avoid duplicate writers
  or partial CPU replays; the separate receipt records this limitation.

- SEAM training completed at1,500 updates without failures/replays. Its active
  wall time was2,399.23s, concurrent with M and a short Bestof sampling wave;
  whole-device peak12,157MiB and minimum free3,823MiB. The independently checked
  final state and full accounting are under seam/closeout/.
- At10:35UTC after SEAM released CUDA, free8,637MiB permitted one official D0
  worker at the conservative3,500MiB estimate. All13 new Bestof winners completed
  single D0/GEO in55.632s (scoped processor timing; validation/startup separately).
- At10:37UTC, free8,636MiB, utilization22%, CPU8.8% and availableRAM~57.1GiB
  permitted the one SEAM46 sampling worker at2,800MiB estimate, while M continues.
  No training changes or additional sampling seeds were introduced.

- SEAM46 Euler50 completed in183.384s with2,300 generator forwards and21 C-only
  head forwards; two closed C skipped the head. At10:46UTC, free8,627MiB,
  GPU utilization26%, CPU16.1%, availableRAM~55.9GiB permitted its single D0
  worker at3,500MiB estimate. All46 D0/GEO outputs completed in55.452s.
  RAW CPU scoring and POST CPU processing use separate3-worker pools with
  one numerical thread per worker, overlapping the remaining M training.
- EXPOSURE completed exactly3,000 updates with active wall4,320.491s. Its whole-device
  peak12,156MiB/minimum free3,824MiB includes concurrent work, not its process peak.
  The isolated allocator peak was4,034,676,224 allocated/4,368,367,616 reserved bytes.
- After its training exit, availableVRAM13,142MiB and CPU4.2% permitted one
  EXPOSURE46 sampler (estimate2,800MiB): all46 RAW,2,300 generator forwards,
  zero head forwards in142.828s. At11:13UTC free13,141MiB, GPU1%, CPU4.2%,
  RAM available65.53GB permitted one D0 worker(3,500MiB estimate); all46 single
  D0/GEO calls completed in46.803s. Three-worker RAW and POST CPU pools overlap.
  These timings include the declared concurrent workloads and are not isolated
  performance comparisons with SEAM or Bestof.

- At12:14:40UTC the HYBRID scheduling limitation was addressed without changing
  frozen scientific code or replaying a request. After validating PID/start-time,
  parentage, absence of child processes and unopened output paths, the experiment's
  original CPU worker619 (including its waiting thread622) and renderer1547 were
  SIGSTOP-paused with their memory retained. The original pool and its queue remain
  intact. A separate controller invokes the original `one(task, registration_sha)`
  only for the five still-unstarted registered requests; it does not write the
  global completion receipt. Af/n1 started at12:14:41UTC. Its first complete BASE
  scan and a fresh resource measurement preceded bcb/n0 at12:14:59UTC. At most two
  geometry workers are active, each with one numerical thread; the paused worker's
  memory remains included in resource accounting. After all five inner and outer
  receipts verify, the controller resumes the original worker and renderer. The
  original queue then reads those completion receipts without new geometry/scoring.
  Exact events are in `hybrid/dispatch/preserved_worker_recovery_v1/`. The original
  af/n0 wall time includes the pause; report the pause interval separately and do
  not label pause-adjusted wall time as measured CPU compute time. A preliminary
  overly strict thread-count assertion exited before any signal or scientific call;
  its replacement verifies that the entire thread group is paused, without treating
  a zero-CPU futex-waiting thread as a second numerical worker.
- The five redistributed requests all finished and passed both receipt/hash checks.
  At13:20:30UTC the controller resumed the original CPU worker and renderer;
  both recorded zero CPU-time increase during their approximately3,949.59s pause.
  See `hybrid/dispatch/preserved_worker_recovery_v1/complete.json` for the exact
  lower/upper pause bounds and actual worker peaks. At this scheduling milestone
  45/46 requests were complete; af/n0 continued from its original in-memory state.
