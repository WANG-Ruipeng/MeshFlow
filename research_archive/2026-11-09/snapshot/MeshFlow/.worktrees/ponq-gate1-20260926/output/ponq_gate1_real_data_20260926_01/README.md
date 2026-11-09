# Real-object Gate 1 data selection

Selected before any PoNQ fitting or extraction. Source files were read-only; no mesh repair was performed.

|Slot|UID|Raw faces|Betti|Prior status|
|---|---|---:|---|---|
|loudspeaker_local_1|7f1be3e7b6adeeae9bb399e812556da2|106|[1, 0, 1]|closed16_development_parent|
|loudspeaker_local_2|736820a85063a4371e5b68ee10358ea3|76|[1, 0, 1]|closed16_development_parent|
|bottle_local_1|ab6792cddc7c4c83afbf338b16b43f53|718|[1, 0, 1]|ray_reject|
|bottle_local_2|194f4eb1707aaf674c8b72e8da0e65c5|642|[1, 0, 1]|ray_reject|

Audited 6 candidates; rejected 2. All attempted failures are retained in SELECTION.json and candidates/.

This is not a new qualification rate for all 225 bottles. The preferred 61-object pool previously passed a genus-zero screen, so this selection cannot establish nonzero-genus bottle coverage.

References are already bbox-centered with diagonal 2; consumers must not normalize them again.

The speaker raw NPZ identity is bound to the earlier assets/parents.json. This does not newly certify the raw archive's historical preprocessing or make these unseen test subjects.
