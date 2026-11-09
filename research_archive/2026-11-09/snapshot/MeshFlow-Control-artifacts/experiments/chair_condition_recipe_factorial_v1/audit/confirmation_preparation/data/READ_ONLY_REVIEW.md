# Read-only review of confirmation-data preparation

Status: PASS for isolated code and small synthetic CPU tests. No real confirmation plan, OT preparation, evaluation-noise materialization, model call, or training was executed. Existing stage1 jobs were not changed. All97 files in registrations/source_hashes.json match their frozen hashes.

## Test-output provenance

The dedicated command was executed once during implementation:

`wsl -d Ubuntu --exec /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m pytest -q /mnt/e/MeshFlow-Control/tests/test_recipe_confirmation_data.py -p no:cacheprovider`

Tool result chunk aa963f reported exit0 and11 PASS in4.85 seconds (command wall5.6516057 seconds). tests_stdout_from_tool.txt is a text copy of that displayed tool output, not an original shell redirection or independently recaptured raw byte stream. No JUnit XML was requested or captured. No test was rerun merely to create these receipts. Tests use synthetic six-update fixtures and small pure planning examples, including synthetic OT; those fixture calls are not real experiment OT or model execution.

## Namespace actually changes computation

- recipe_confirmation.py:44 derives each seed from the validated confirmation namespace and its independent domain.
- Lines77-109 instantiate fresh parent/task40/alpha/slot-shuffle RandomStates and use them to choose actual records; Gaussian, epsilon, permutation and time seeds are stored per origin.
- Frozen data/stream.py:140-147 consumes those exact per-record seeds in actual permutation, Gaussian, epsilon and training-time construction. They are not metadata-only values.
- recipe_confirmation.py:129 and136 pass the new namespace into ratio assignment and new20 task selection. No old paired table, old plan or old OT path is loaded.
- test_recipe_confirmation_data.py:51 checks deterministic reproduction, changed domains/records versus the old schedule, actual changed Gaussian/epsilon arrays and independence from which pair is selected.

## R1 identity is the new plan

recipe_confirmation.py:176 selects R1 records from the new base_records. Lines232-254 bind the stream to confirmation_plan.json and create paired_path under the new stream root. Lines377-381 write the R1 table with the same new plan_sha256 as the stream. The parametrized test at test_recipe_confirmation_data.py:124 includes R1/R3 and R1/R2 and verifies actual hashes, new-plan identity and selected-union cache counts. There is no stage1 old-plan hash exception.

## Course multiset and final suffix

recipe_confirmation.py:152-156 constructs both courses from the same new MIX origin multiset and appends the unchanged final suffix. Lines176-191 change optimizer order/position while preserving origin-based inputs. Tests at lines78 and192 verify distinct parents,4clean/4aug, exact pure-ratio stages, complete multiset, actual input hashes and identical final MIX suffix. For the real6000-update plan the segment length is2000; small fixtures use the same proportions.

## Preserved boundaries

The plan explicitly retains condition_pool_policy=EXISTING_HETEROGENEOUS_POOLS_UNCHANGED and strict_same_candidate_selection_policy=NOT_SATISFIED. Geo_seed remains1010 and independent_model_initialization is false. New training data and evaluation noise remain gated by a later concrete candidate/control decision; no pair is chosen by this preparation receipt.
