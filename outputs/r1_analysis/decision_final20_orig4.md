
## Alarm rates, tau = 0.2, orig4, c* = 3.947 (20 seeds, macro over bearings)

| Policy | False alarm % | Missed detection % |
|---|---|---|
| point | 1.3 ± 0.0 | 69.3 ± 10.5 |
| uncalibrated | 1.8 ± 0.4 | 61.6 ± 9.5 |
| calibrated | 17.8 ± 5.8 | 19.3 ± 8.6 |

## Alarm timing, tau = 0.30 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | +89.9 | 60.0 | 18.8 | 40.0 | 803.9 | 2/4 |
| pava | +383.3 | 97.5 | 55.0 | 2.5 | 1925.4 | 4/4 |
| rtrm | +89.9 | 60.0 | 18.8 | 40.0 | 803.9 | 2/4 |
| ces | +353.9 | 93.8 | 50.0 | 6.2 | 1789.8 | 4/4 |
| lb_raw | -849.6 | 0.0 | 0.0 | 100.0 | 849.6 | 0/4 |
| lb_ces | -809.1 | 0.0 | 0.0 | 100.0 | 809.1 | 0/4 |
decision-optimal by r = c_late/c_early: raw [0.50, 5.07]; lb_ces [5.07, 100.00]

## Alarm timing, tau = 0.20 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | +154.6 | 48.8 | 47.5 | 51.2 | 1087.2 | 2/4 |
| pava | +280.4 | 85.0 | 62.5 | 15.0 | 1451.0 | 3/4 |
| rtrm | +154.6 | 48.8 | 47.5 | 51.2 | 1087.2 | 2/4 |
| ces | +237.1 | 73.8 | 50.0 | 26.2 | 1265.5 | 3/4 |
| lb_raw | -821.9 | 0.0 | 0.0 | 100.0 | 821.9 | 0/4 |
| lb_ces | -693.9 | 0.0 | 0.0 | 100.0 | 693.9 | 0/4 |
decision-optimal by r = c_late/c_early: ces [0.50, 0.90]; raw [0.90, 3.12]; lb_ces [3.12, 100.00]

## Alarm timing, tau = 0.10 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | +43.9 | 57.5 | 50.0 | 42.5 | 626.3 | 2/4 |
| pava | +130.9 | 88.8 | 63.7 | 11.2 | 787.1 | 4/4 |
| rtrm | +43.9 | 57.5 | 50.0 | 42.5 | 626.3 | 2/4 |
| ces | +89.2 | 73.8 | 55.0 | 26.2 | 730.4 | 3/4 |
| lb_raw | -777.6 | 0.0 | 0.0 | 100.0 | 777.6 | 0/4 |
| lb_ces | -510.9 | 3.8 | 2.5 | 96.2 | 541.5 | 0/4 |
decision-optimal by r = c_late/c_early: pava [0.50, 1.11]; raw [1.11, 4.21]; lb_ces [4.21, 51.51]; lb_raw [51.51, 100.00]

written E:\Yolo-Thermal\Dual-Stream Vibration-Vision\outputs\r1_analysis\decision_final20_orig4.json
