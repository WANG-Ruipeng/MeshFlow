# CPU contract check receipt

Executed by implementation sub-agent before freeze: 7/7 unittest cases PASS. Checks cover FP32 C area center, C-only reference, preserved Gaussian (no demeaning/translation), 51-state world C byte-copy, world/local geometry descriptor difference, pre-OT half-scale conversion, and real Native model meta-device parameter/state identity. Learned model forwards=0; updates=0; CUDA initialized=false.

Command: python -B -m r2.pilots.c_reference_frame_v1.test_model_contract

Actual source model.py SHA256: 0c7f461315a29de304be22a91623b53b7836a7e8522fcb694460236edf482659.