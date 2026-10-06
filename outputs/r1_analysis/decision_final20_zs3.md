
## Alarm rates, tau = 0.2, zs3, c* = 3.947 (20 seeds, macro over bearings)

| Policy | False alarm % | Missed detection % |
|---|---|---|
| point | 5.8 ± 1.3 | 80.1 ± 4.0 |
| uncalibrated | 13.7 ± 3.4 | 63.4 ± 5.9 |
| calibrated | 49.1 ± 5.8 | 9.1 ± 4.5 |

## Alarm timing, tau = 0.30 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -278.8 | 11.7 | 0.0 | 88.3 | 290.9 | 0/3 |
| pava | +45.5 | 48.3 | 48.3 | 51.7 | 665.4 | 1/3 |
| rtrm | -278.8 | 11.7 | 0.0 | 88.3 | 290.9 | 0/3 |
| ces | -121.5 | 33.3 | 21.7 | 66.7 | 289.1 | 1/3 |
| lb_raw | -513.6 | 0.0 | 0.0 | 100.0 | 513.6 | 0/3 |
| lb_ces | -471.4 | 0.0 | 0.0 | 100.0 | 471.4 | 0/3 |
decision-optimal by r = c_late/c_early: pava [0.50, 0.85]; ces [0.85, 5.07]; raw [5.07, 94.00]; lb_ces [94.00, 100.00]

## Alarm timing, tau = 0.20 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -245.8 | 10.0 | 0.0 | 90.0 | 264.4 | 0/3 |
| pava | +130.6 | 76.7 | 76.7 | 23.3 | 727.9 | 2/3 |
| rtrm | -245.8 | 10.0 | 0.0 | 90.0 | 264.4 | 0/3 |
| ces | +53.8 | 53.3 | 53.3 | 46.7 | 516.5 | 2/3 |
| lb_raw | -540.4 | 0.0 | 0.0 | 100.0 | 540.4 | 0/3 |
| lb_ces | -474.6 | 0.0 | 0.0 | 100.0 | 474.6 | 0/3 |
decision-optimal by r = c_late/c_early: pava [0.50, 0.60]; ces [0.60, 2.27]; raw [2.27, 73.38]; lb_ces [73.38, 100.00]

## Alarm timing, tau = 0.10 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | +6.9 | 53.3 | 53.3 | 46.7 | 299.3 | 2/3 |
| pava | +77.2 | 100.0 | 100.0 | 0.0 | 386.0 | 3/3 |
| rtrm | +6.9 | 53.3 | 53.3 | 46.7 | 299.3 | 2/3 |
| ces | +44.9 | 66.7 | 66.7 | 33.3 | 366.0 | 2/3 |
| lb_raw | -545.9 | 0.0 | 0.0 | 100.0 | 545.9 | 0/3 |
| lb_ces | -434.3 | 0.0 | 0.0 | 100.0 | 434.3 | 0/3 |
decision-optimal by r = c_late/c_early: pava [0.50, 1.69]; raw [1.69, 7.69]; lb_ces [7.69, 100.00]

written E:\Yolo-Thermal\Dual-Stream Vibration-Vision\outputs\r1_analysis\decision_final20_zs3.json
