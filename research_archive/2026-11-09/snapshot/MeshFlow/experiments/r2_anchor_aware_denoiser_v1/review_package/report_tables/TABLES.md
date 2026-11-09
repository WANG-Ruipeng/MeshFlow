# Completed-score tables

All percentages use macro per-output means unless explicitly weighted; differences are percentage points. No added metric or scorer calls.

| denoiser | label | outputs | valid_connected_outputs | parents_with_valid_connection | valid_CF_edge_sum | reachable_free_area_pct | strict_AVMR_pct | strict_PMR_pct | CF_pct | FF_pct | I_pct | small_angle_pct | deletion_weighted_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D0_OFFICIAL | R2_CONT_OLD | 12 | 1 | 1 | 1 | 0.035858 | 3.9021 | 0 | 7.3276 | 15.3 | 7.9386 | 11.387 | 9.8011 |
| D0_OFFICIAL | R2_CONT_USER | 12 | 0 | 0 | 0 | 0 | 3.5053 | 0 | 7.1867 | 15.36 | 12.415 | 17.586 | 12.027 |
| D0_OFFICIAL | R2_FROZEN | 12 | 0 | 0 | 0 | 0 | 1.8519 | 0 | 8.1316 | 16.113 | 9.2611 | 10.072 | 11.6 |
| D0_OFFICIAL | R2_USER_MIX | 12 | 1 | 1 | 1 | 0.013701 | 1.3889 | 0 | 4.8784 | 12.019 | 3.6865 | 17.062 | 15.388 |
| D_ANCHOR | R2_CONT_OLD | 12 | 2 | 2 | 3 | 1.1015 | 3.9021 | 0 | 7.0803 | 17.103 | 7.7026 | 10.759 | 9.233 |
| D_ANCHOR | R2_CONT_USER | 12 | 0 | 0 | 0 | 0 | 2.4471 | 0 | 6.378 | 14.839 | 12.374 | 16.524 | 11.648 |
| D_ANCHOR | R2_FROZEN | 12 | 0 | 0 | 0 | 0 | 1.8519 | 0 | 7.9652 | 18.648 | 8.8501 | 10.408 | 10.89 |
| D_ANCHOR | R2_USER_MIX | 12 | 2 | 1 | 2 | 6.6365 | 2.3148 | 0 | 4.6987 | 13.411 | 3.658 | 16.753 | 14.725 |
| D_CONT_NO_ROLE | R2_CONT_OLD | 12 | 2 | 2 | 3 | 1.1102 | 3.9021 | 0 | 6.9995 | 17.286 | 7.7026 | 10.759 | 8.9962 |
| D_CONT_NO_ROLE | R2_CONT_USER | 12 | 0 | 0 | 0 | 0 | 2.9101 | 0 | 6.3594 | 13.945 | 12.451 | 16.584 | 11.553 |
| D_CONT_NO_ROLE | R2_FROZEN | 12 | 0 | 0 | 0 | 0 | 1.8519 | 0 | 8.0397 | 18.62 | 8.964 | 10.294 | 10.843 |
| D_CONT_NO_ROLE | R2_USER_MIX | 12 | 2 | 1 | 2 | 6.6901 | 2.3148 | 0 | 4.6485 | 13.534 | 3.658 | 16.911 | 14.347 |

## Synthetic same-shape diagnostics (denoiser coordinate units)

| denoiser | input_kind | count | free_mse_macro | supported_interface_mse_macro | free_movement_corner_l2_mean_macro |
|---|---|---|---|---|---|
| D0_OFFICIAL | clean | 6 | 5.186979670224501e-06 | 3.449041289355358e-06 | 0.0032649230238899173 |
| D0_OFFICIAL | noisy | 12 | 1.3972843640233797e-05 | 9.062997812746e-06 | 0.031238348554303852 |
| D_CONT_NO_ROLE | clean | 6 | 9.833608437581209e-06 | 5.930453710277796e-06 | 0.004664031886236516 |
| D_CONT_NO_ROLE | noisy | 12 | 1.7137881621273444e-05 | 1.0918885050213738e-05 | 0.031527330981057686 |
| D_ANCHOR | clean | 6 | 9.86313603626825e-06 | 6.038469229398797e-06 | 0.0046750373269229305 |
| D_ANCHOR | noisy | 12 | 1.710212460763829e-05 | 1.0995809692966055e-05 | 0.03153241591575789 |
