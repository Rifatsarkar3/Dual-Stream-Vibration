
## Alarm rates, tau = 0.2, all10, c* = 3.947 (20 seeds, macro over bearings)

| Policy | False alarm % | Missed detection % |
|---|---|---|
| point | 2.2 ± 0.6 | 79.5 ± 4.0 |
| uncalibrated | 5.1 ± 1.0 | 67.5 ± 3.9 |
| calibrated | 25.2 ± 4.1 | 29.9 ± 4.7 |

## Alarm timing, tau = 0.30 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -255.7 | 45.0 | 20.0 | 55.0 | 722.1 | 4/10 |
| pava | +223.0 | 81.5 | 60.5 | 18.5 | 1451.6 | 8/10 |
| rtrm | -255.7 | 45.0 | 20.0 | 55.0 | 722.1 | 4/10 |
| ces | +56.0 | 64.5 | 43.0 | 35.5 | 1058.0 | 7/10 |
| lb_raw | -764.7 | 12.0 | 6.5 | 86.5 | 795.8 | 1/10 |
| lb_ces | -725.3 | 16.0 | 9.0 | 83.5 | 779.9 | 2/10 |
decision-optimal by r = c_late/c_early: pava [0.50, 0.79]; ces [0.79, 1.90]; raw [1.90, 5.85]; lb_ces [5.85, 9.10]; lb_raw [9.10, 100.00]

## Alarm timing, tau = 0.20 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -217.9 | 42.5 | 41.0 | 57.5 | 839.6 | 4/10 |
| pava | +267.1 | 93.5 | 84.0 | 6.5 | 1373.6 | 9/10 |
| rtrm | -217.9 | 42.5 | 41.0 | 57.5 | 839.6 | 4/10 |
| ces | +107.7 | 74.0 | 63.0 | 26.0 | 1068.9 | 8/10 |
| lb_raw | -762.4 | 16.5 | 10.0 | 83.5 | 787.3 | 2/10 |
| lb_ces | -685.9 | 19.5 | 15.5 | 80.5 | 755.7 | 2/10 |
decision-optimal by r = c_late/c_early: pava [0.50, 1.07]; ces [1.07, 2.52]; raw [2.52, 4.10]; lb_ces [4.10, 9.26]; lb_raw [9.26, 100.00]

## Alarm timing, tau = 0.10 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -233.9 | 60.0 | 56.0 | 40.0 | 677.1 | 6/10 |
| pava | +131.8 | 95.5 | 85.5 | 4.5 | 712.2 | 10/10 |
| rtrm | -233.9 | 60.0 | 56.0 | 40.0 | 677.1 | 6/10 |
| ces | +109.9 | 88.5 | 81.0 | 11.5 | 696.7 | 9/10 |
| lb_raw | -800.4 | 10.0 | 10.0 | 90.0 | 810.5 | 1/10 |
| lb_ces | -646.4 | 21.0 | 20.5 | 79.0 | 708.5 | 2/10 |
decision-optimal by r = c_late/c_early: pava [0.50, 2.52]; ces [2.52, 4.69]; raw [4.69, 5.50]; lb_ces [5.50, 16.90]; lb_raw [16.90, 100.00]

written E:\Yolo-Thermal\Dual-Stream Vibration-Vision\outputs\r1_analysis\decision_final20_all10.json
