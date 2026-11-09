# R2 terminal-reward NFT pilot

Independent finite experiment; original models, assets, samplers, scorers and defaults remain protected.

`resume.ps1` defaults to read-only status. `-Stage pipeline -Round 0` verifies completed stages and continues authorized work until the next mandatory actual-image review gate. It never fills visual labels automatically. Completed matching cells are reused; failed/incomplete scientific calls require explicit documented engineering recovery, not replay.

Scientific registration: `experiments/r2_terminal_reward_nft_v1/registration.json`. Inputs and all training shuffle/re-noise/time identities are frozen before rollout. Code receipts and amendments identify instrumentation/dispatch changes.

Frozen FULL_S6000 -> 16 rounds of 4 conditions x 8 online samples -> single official D0 -> unchanged GEO_C/source winding -> geometry score and actual AI_PROXY review -> within-C terminal reward -> 16 LoRA optimizer updates per arm. NFT samples feed both NFT and reward-free SELF_REFLOW. This control is off-policy and changes the re-noising path relative to historical OT/HYBRID; it is not ordinary original-data FM continuation. The reference penalty is velocity MSE, not exact distribution KL.

512 training + 84 final evaluation new RAW; 596 single-D0 proposals; 29,800 Euler forwards plus one explicitly accounted zero-adapter validation forward. Final evaluation keeps old24 watched requests separate from new12, and TRAIN separate from DEV. No best-checkpoint selection. Best-of8 frozen-reference evidence only covers the four round0 conditions; later rounds belong to the online policy.

Run CPU score/render with `python -m r2.pilots.terminal_reward_nft_v1.cpu_worker --round 0 --stage all --workers 2`. Review `training/round_00/gallery/index.html` and preserve real review evidence in `reviews_actual.json`; no metric-based auto-labeling. `prepare_rewards` creates the immutable training stream only after complete actual reviews, with UNKNOWN abstention.

Default entry does not commit, push, clean assets, change default models or add trials. Final checkpoint is round15; first-round lack of informative reward is a predeclared stop, not a successful training run.
