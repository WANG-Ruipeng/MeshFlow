# Eight-label saved-table comparison

512 original primary score rows: main384 plus course128 on the identical64 requests. No scoring, model, geometry or checkpoint reads. All scalar fields, statuses, NA values and original nested rows remain in all_metrics.json. No source table is changed.

Each mean uses four noises per parent, followed by equal parent weights. C20 and C40 each retain ALL8, ORIGINAL6 and EXTENDED2; at C20 these denote parent membership. Missing numeric values keep their original denominators; the main mean stays blank. Nonnumeric/all-NA fields retain rows and are explicitly numeric_summary=False.

The only six contrasts are each curriculum minus START5000, R1_40_H_ALL and HIST_FM6000. Improved/worsened/tied parent counts use the existing individual metric direction and exact sign, with no new threshold or composite score. Descriptive directions have NA improvement counts. lower_uncertainty concerns uncertainty, not certified quality. Counts and means from single training realizations on observed DEV parents are descriptive; they establish neither significance nor automatic selection.

Reference GT/S/E and diversity retain their separate original analysis artifacts; this supplement compares the full primary metric table only.
