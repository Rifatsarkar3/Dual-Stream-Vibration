
## Alarm rates, tau = 0.2, new6, c* = 3.947 (20 seeds, macro over bearings)

| Policy | False alarm % | Missed detection % |
|---|---|---|
| point | 2.8 ± 1.1 | 86.3 ± 3.1 |
| uncalibrated | 7.3 ± 1.6 | 71.4 ± 5.4 |
| calibrated | 30.1 ± 3.2 | 36.9 ± 2.4 |

## Alarm timing, tau = 0.30 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -486.2 | 35.0 | 20.8 | 65.0 | 667.5 | 2/6 |
| pava | +116.2 | 70.8 | 64.2 | 29.2 | 1135.8 | 4/6 |
| rtrm | -486.2 | 35.0 | 20.8 | 65.0 | 667.5 | 2/6 |
| ces | -142.6 | 45.0 | 38.3 | 55.0 | 570.2 | 3/6 |
| lb_raw | -708.1 | 20.0 | 10.8 | 77.5 | 759.9 | 1/6 |
| lb_ces | -669.4 | 26.7 | 15.0 | 72.5 | 760.4 | 2/6 |
decision-optimal by r = c_late/c_early: pava [0.50, 0.89]; ces [0.89, 7.42]; raw [7.42, 9.34]; lb_raw [9.34, 100.00]

## Alarm timing, tau = 0.20 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -466.3 | 38.3 | 36.7 | 61.7 | 674.6 | 2/6 |
| pava | +258.2 | 99.2 | 98.3 | 0.8 | 1322.0 | 6/6 |
| rtrm | -466.3 | 38.3 | 36.7 | 61.7 | 674.6 | 2/6 |
| ces | +21.4 | 74.2 | 71.7 | 25.8 | 937.8 | 5/6 |
| lb_raw | -722.8 | 27.5 | 16.7 | 72.5 | 764.2 | 2/6 |
| lb_ces | -680.6 | 32.5 | 25.8 | 67.5 | 796.9 | 2/6 |
decision-optimal by r = c_late/c_early: pava [0.50, 1.29]; ces [1.29, 2.91]; raw [2.91, 8.25]; lb_raw [8.25, 100.00]

## Alarm timing, tau = 0.10 (dt in windows of 10 s)

| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |
|---|---|---|---|---|---|---|
| raw | -419.2 | 61.7 | 60.0 | 38.3 | 711.0 | 4/6 |
| pava | +132.4 | 100.0 | 100.0 | 0.0 | 662.2 | 6/6 |
| rtrm | -419.2 | 61.7 | 60.0 | 38.3 | 711.0 | 4/6 |
| ces | +123.6 | 98.3 | 98.3 | 1.7 | 674.2 | 6/6 |
| lb_raw | -815.5 | 16.7 | 16.7 | 83.3 | 832.4 | 1/6 |
| lb_ces | -736.7 | 32.5 | 32.5 | 67.5 | 819.9 | 2/6 |
decision-optimal by r = c_late/c_early: pava [0.50, 5.59]; raw [5.59, 7.69]; lb_raw [7.69, 100.00]

written E:\Yolo-Thermal\Dual-Stream Vibration-Vision\outputs\r1_analysis\decision_final20_new6.json
