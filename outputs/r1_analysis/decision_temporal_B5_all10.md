
## Alarm rates, tau = 0.2, all10, c* = 3.023 (20 seeds, macro over bearings)

| Policy | False alarm % | Missed detection % |
|---|---|---|
| point | 3.6 ± 0.9 | 79.4 ± 3.2 |
| uncalibrated | 6.7 ± 1.0 | 66.8 ± 4.8 |
| calibrated | 22.7 ± 4.2 | 34.7 ± 5.8 |

## Alarm timing, tau = 0.30 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -123.0 | 53.0 | 35.5 | 47.0 | 941.5 | 5/10 |
| pava | +195.3 | 73.5 | 60.0 | 26.5 | 1423.7 | 7/10 |
| rtrm | -123.0 | 53.0 | 35.5 | 47.0 | 941.5 | 5/10 |
| ces | +13.3 | 61.5 | 41.0 | 38.5 | 1017.4 | 6/10 |
| lb_raw | -659.8 | 20.0 | 9.5 | 80.0 | 739.0 | 2/10 |
| lb_ces | -633.7 | 20.0 | 10.5 | 80.0 | 724.7 | 2/10 |
decision-optimal by r = c_late/c_early: pava [0.50, 0.86]; ces [0.86, 2.86]; raw [2.86, 3.23]; lb_ces [3.23, 12.29]; lb_raw [12.29, 100.00]

## Alarm timing, tau = 0.20 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -78.1 | 48.5 | 47.0 | 51.5 | 787.7 | 5/10 |
| pava | +206.4 | 90.0 | 79.0 | 10.0 | 1261.7 | 9/10 |
| rtrm | -78.1 | 48.5 | 47.0 | 51.5 | 787.7 | 5/10 |
| ces | +49.3 | 60.5 | 58.5 | 39.5 | 931.8 | 6/10 |
| lb_raw | -689.8 | 20.0 | 14.0 | 80.0 | 750.8 | 2/10 |
| lb_ces | -649.5 | 20.5 | 19.0 | 79.5 | 765.2 | 2/10 |
decision-optimal by r = c_late/c_early: pava [0.50, 0.94]; ces [0.94, 1.82]; raw [1.82, 4.69]; lb_raw [4.69, 100.00]

## Alarm timing, tau = 0.10 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -14.0 | 68.0 | 67.0 | 32.0 | 602.6 | 7/10 |
| pava | +144.8 | 96.0 | 90.5 | 4.0 | 734.6 | 10/10 |
| rtrm | -14.0 | 68.0 | 67.0 | 32.0 | 602.6 | 7/10 |
| ces | +52.5 | 76.0 | 73.0 | 24.0 | 642.7 | 7/10 |
| lb_raw | -668.2 | 20.5 | 20.5 | 79.5 | 737.1 | 2/10 |
| lb_ces | -561.6 | 23.5 | 23.5 | 76.5 | 664.9 | 2/10 |
decision-optimal by r = c_late/c_early: pava [0.50, 2.02]; ces [2.02, 2.76]; raw [2.76, 5.79]; lb_ces [5.79, 17.66]; lb_raw [17.66, 100.00]

written outputs\r1_analysis\decision_temporal_B5_all10.json
