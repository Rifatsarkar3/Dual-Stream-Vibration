
## Alarm rates, tau = 0.2, new6, c* = 3.023 (20 seeds, macro over bearings)

| Policy | False alarm % | Missed detection % |
|---|---|---|
| point | 5.0 ± 1.5 | 83.3 ± 4.5 |
| uncalibrated | 9.3 ± 1.5 | 66.4 ± 7.5 |
| calibrated | 27.6 ± 4.5 | 39.2 ± 4.0 |

## Alarm timing, tau = 0.30 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -360.9 | 40.8 | 35.0 | 59.2 | 677.5 | 2/6 |
| pava | +56.5 | 65.8 | 55.0 | 34.2 | 994.3 | 4/6 |
| rtrm | -360.9 | 40.8 | 35.0 | 59.2 | 677.5 | 2/6 |
| ces | -193.2 | 46.7 | 36.7 | 53.3 | 576.8 | 3/6 |
| lb_raw | -609.4 | 33.3 | 15.8 | 66.7 | 741.4 | 2/6 |
| lb_ces | -588.8 | 33.3 | 17.5 | 66.7 | 740.6 | 2/6 |
decision-optimal by r = c_late/c_early: pava [0.50, 1.25]; ces [1.25, 8.94]; lb_raw [8.94, 100.00]

## Alarm timing, tau = 0.20 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -242.4 | 47.5 | 45.0 | 52.5 | 562.9 | 3/6 |
| pava | +152.2 | 90.0 | 83.3 | 10.0 | 1122.2 | 5/6 |
| rtrm | -242.4 | 47.5 | 45.0 | 52.5 | 562.9 | 3/6 |
| ces | -44.3 | 65.0 | 64.2 | 35.0 | 796.3 | 4/6 |
| lb_raw | -669.5 | 33.3 | 23.3 | 66.7 | 771.1 | 2/6 |
| lb_ces | -633.8 | 33.3 | 30.8 | 66.7 | 802.7 | 2/6 |
decision-optimal by r = c_late/c_early: pava [0.50, 1.27]; ces [1.27, 1.76]; raw [1.76, 10.76]; lb_raw [10.76, 100.00]

## Alarm timing, tau = 0.10 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -48.5 | 80.0 | 78.3 | 20.0 | 599.2 | 5/6 |
| pava | +136.2 | 100.0 | 100.0 | 0.0 | 681.2 | 6/6 |
| rtrm | -48.5 | 80.0 | 78.3 | 20.0 | 599.2 | 5/6 |
| ces | +47.9 | 90.0 | 88.3 | 10.0 | 649.9 | 5/6 |
| lb_raw | -736.0 | 31.7 | 31.7 | 68.3 | 816.0 | 2/6 |
| lb_ces | -645.8 | 33.3 | 33.3 | 66.7 | 737.8 | 2/6 |
decision-optimal by r = c_late/c_early: pava [0.50, 3.18]; raw [3.18, 6.86]; lb_ces [6.86, 43.93]; lb_raw [43.93, 100.00]

written outputs\r1_analysis\decision_temporal_B5_new6.json
