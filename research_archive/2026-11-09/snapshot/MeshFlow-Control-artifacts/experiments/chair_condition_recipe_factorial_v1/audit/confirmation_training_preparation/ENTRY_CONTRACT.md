# Independent confirmation trainer preparation

Code prepared and CPU tests passed; no execution authorization was created.

Entry: tools/train_recipe_confirmation.py. Required arguments: --recipe, --hybrid-schedule (explicit for a course), --official, --data-manifest, --stream-root, --out, --attempt-id, --registration. Optional --resume accepts only a matching completed confirmation checkpoint; --device defaults to cuda for a future owned worker.

The new authorization schema is chair_recipe_confirmation_training_authorization_v1. It must specify experiment CHAIR_CONDITION_RECIPE_FACTORIAL_V1, stage independent_confirmation, a new confirmation namespace, exactly two distinct full selected_pair labels, the identical authorized_recipes list, a hybrid_schedules mapping exactly covering the pair, updates_per_recipe=6000, effective_batch=8, microbatch=1, max_training_recipes=2, experiment_root, a separate out directory beneath that root, and complete source_hashes. Source registration includes this trainer, its new data module, the new CLI and the original model/data/training dependencies returned by source_identity(). No pair is selected by this preparation.

Data plan: meshflow_recipe_confirmation_plan_v1, namespace CHAIR_CONDITION_RECIPE_FACTORIAL_V1/confirmation/<suffix>. Every recipe, including R1, must use its own new-stream paired table bound to the same new plan. Historical base_stream_root/base plan/old paired hashes are explicitly rejected. The data module preserves the course MIX multiset and last 2000 update suffix.

The initializer, optimizer validation and forward/backward/update arithmetic are direct aliases of the frozen recipe worker. This is the same official EMA/Geo1010 initialization and fresh AdamW, with independently repeated input randomness. No auxiliary B or teacher is constructed. Batch/microbatch, BF16 forward, FP32 weights/integration, strict deterministic math SDPA and no TF32 are preserved.

Each worker owns <out>/training/<recipe>/attempts/<attempt>/{calls.jsonl,events.jsonl,training_log.csv,exposure.csv,diagnostics.jsonl,final_status.json}. Complete resumable latest checkpoints are atomic every100 updates. The immutable endpoint is <out>/checkpoints/<recipe>_step6000.pt and pure generator export is <out>/inference/<recipe>_generator_step6000.pt. Checkpoint schema is chair_recipe_confirmation_training_v1; generator schema is chair_recipe_confirmation_generator_v1. load_generator(path, device='cpu', expected_sha256=None) returns (model, identity) for the future evaluation adapter. Export receipt lives in completed.json.inference_export; no export sidecar is promised.

OOMs are recorded and rethrown; the worker never retries or dispatches. The unchanged external scheduling/replay contract still requires measured resources, at most two workers, one downgrade, at most one authorized serial retry per affected task, and no more than99 replayed completed updates. No registration or dispatcher wave for this future stage was created here.

Not executed: pair selection, genuine new-plan preparation, OT precomputation, official checkpoint loading, model construction, training, sampling, scoring or promotion. This preparation changes no defaults or existing frozen source.
