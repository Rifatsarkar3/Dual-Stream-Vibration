
## Alarm rates, tau = 0.2, orig4, c* = 3.023 (20 seeds, macro over bearings)

| Policy | False alarm % | Missed detection % |
|---|---|---|
| point | 1.5 ± 0.2 | 73.4 ± 6.7 |
| uncalibrated | 2.7 ± 0.7 | 67.5 ± 7.1 |
| calibrated | 15.4 ± 4.2 | 27.9 ± 10.4 |

## Alarm timing, tau = 0.30 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | +233.8 | 71.2 | 36.2 | 28.7 | 1337.3 | 3/4 |
| pava | +403.4 | 85.0 | 67.5 | 15.0 | 2067.8 | 3/4 |
| rtrm | +233.8 | 71.2 | 36.2 | 28.7 | 1337.3 | 3/4 |
| ces | +323.1 | 83.8 | 47.5 | 16.2 | 1678.2 | 3/4 |
| lb_raw | -735.4 | 0.0 | 0.0 | 100.0 | 735.4 | 0/4 |
| lb_ces | -700.9 | 0.0 | 0.0 | 100.0 | 700.9 | 0/4 |
decision-optimal by r = c_late/c_early: raw [0.50, 2.59]; lb_ces [2.59, 100.00]

## Alarm timing, tau = 0.20 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | +168.2 | 50.0 | 50.0 | 50.0 | 1124.9 | 2/4 |
| pava | +287.6 | 90.0 | 72.5 | 10.0 | 1471.0 | 4/4 |
| rtrm | +168.2 | 50.0 | 50.0 | 50.0 | 1124.9 | 2/4 |
| ces | +189.6 | 53.8 | 50.0 | 46.2 | 1135.0 | 2/4 |
| lb_raw | -720.3 | 0.0 | 0.0 | 100.0 | 720.3 | 0/4 |
| lb_ces | -672.9 | 1.2 | 1.2 | 98.8 | 708.9 | 0/4 |
decision-optimal by r = c_late/c_early: ces [0.50, 3.04]; lb_ces [3.04, 6.92]; lb_raw [6.92, 100.00]

## Alarm timing, tau = 0.10 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | +37.7 | 50.0 | 50.0 | 50.0 | 607.7 | 2/4 |
| pava | +157.7 | 90.0 | 76.2 | 10.0 | 814.7 | 4/4 |
| rtrm | +37.7 | 50.0 | 50.0 | 50.0 | 607.7 | 2/4 |
| ces | +59.6 | 55.0 | 50.0 | 45.0 | 631.9 | 2/4 |
| lb_raw | -566.5 | 3.8 | 3.8 | 96.2 | 618.9 | 0/4 |
| lb_ces | -435.3 | 8.8 | 8.8 | 91.2 | 555.4 | 0/4 |
decision-optimal by r = c_late/c_early: pava [0.50, 1.10]; ces [1.10, 1.85]; raw [1.85, 4.41]; lb_ces [4.41, 10.67]; lb_raw [10.67, 100.00]

written outputs\r1_analysis\decision_temporal_B5_orig4.json
