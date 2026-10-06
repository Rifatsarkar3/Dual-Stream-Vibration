
## Alarm rates, tau = 0.2, zs3, c* = 3.023 (20 seeds, macro over bearings)

| Policy | False alarm % | Missed detection % |
|---|---|---|
| point | 10.2 ± 4.8 | 81.2 ± 7.3 |
| uncalibrated | 26.9 ± 6.8 | 47.9 ± 10.7 |
| calibrated | 59.1 ± 4.4 | 8.3 ± 3.5 |

## Alarm timing, tau = 0.30 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -307.8 | 6.7 | 0.0 | 93.3 | 314.7 | 0/3 |
| pava | -176.6 | 26.7 | 8.3 | 73.3 | 339.8 | 1/3 |
| rtrm | -307.8 | 6.7 | 0.0 | 93.3 | 314.7 | 0/3 |
| ces | -246.4 | 18.3 | 0.0 | 81.7 | 274.3 | 1/3 |
| lb_raw | -479.8 | 0.0 | 0.0 | 100.0 | 479.8 | 0/3 |
| lb_ces | -469.3 | 0.0 | 0.0 | 100.0 | 469.3 | 0/3 |
decision-optimal by r = c_late/c_early: pava [0.50, 2.11]; ces [2.11, 16.60]; raw [16.60, 100.00]

## Alarm timing, tau = 0.20 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -232.1 | 3.3 | 1.7 | 96.7 | 267.0 | 0/3 |
| pava | +96.6 | 80.0 | 80.0 | 20.0 | 728.9 | 3/3 |
| rtrm | -232.1 | 3.3 | 1.7 | 96.7 | 267.0 | 0/3 |
| ces | -157.4 | 13.3 | 13.3 | 86.7 | 259.0 | 0/3 |
| lb_raw | -516.1 | 0.0 | 0.0 | 100.0 | 516.1 | 0/3 |
| lb_ces | -496.5 | 0.0 | 0.0 | 100.0 | 496.5 | 0/3 |
decision-optimal by r = c_late/c_early: pava [0.50, 1.11]; ces [1.11, 5.74]; raw [5.74, 44.71]; lb_ces [44.71, 100.00]

## Alarm timing, tau = 0.10 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -51.8 | 50.0 | 50.0 | 50.0 | 315.6 | 2/3 |
| pava | +80.5 | 100.0 | 100.0 | 0.0 | 402.4 | 3/3 |
| rtrm | -51.8 | 50.0 | 50.0 | 50.0 | 315.6 | 2/3 |
| ces | +25.6 | 63.3 | 63.3 | 36.7 | 360.5 | 2/3 |
| lb_raw | -532.1 | 0.0 | 0.0 | 100.0 | 532.1 | 0/3 |
| lb_ces | -497.6 | 0.0 | 0.0 | 100.0 | 497.6 | 0/3 |
decision-optimal by r = c_late/c_early: pava [0.50, 2.41]; ces [2.41, 2.81]; raw [2.81, 9.18]; lb_ces [9.18, 100.00]

written outputs\r1_analysis\decision_temporal_B5_zs3.json
