# XJTU-SY interval calibration, validation data only (revision R1, Amendment 1, E5)

Coverage of the 90 % intervals on the six XJTU-SY test bearings (pooled over 20 seeds and all windows). Two-sided: mu ± z c s; one-sided lower bound: mu − k s (k = k*) or mu − z c* s.

| Version | c* | k* | uncalibrated two-sided | calibrated two-sided | late life [0, 0.2) | lower bound k* | lower bound c* | per bearing, calibrated two-sided |
|---|---|---|---|---|---|---|---|---|
| adopted | 5.414 | 4.844 | 25.8 % | 90.2 % | 94.4 % | 89.5 % | 98.2 % | Bearing1_3 97, Bearing1_4 77, Bearing1_5 69, Bearing2_3 99, Bearing2_4 86, Bearing2_5 82 |

| submitted | 4.892 | 4.980 | 23.3 % | 86.6 % | 90.7 % | 90.4 % | 97.0 % | Bearing1_3 95, Bearing1_4 67, Bearing1_5 70, Bearing2_3 97, Bearing2_4 81, Bearing2_5 76 |

Replay check (final-epoch validation MAE of the repeated selection runs against the originals): adopted: max |difference| 0.00e+00 (deterministic original, must be 0); submitted: max |difference| 1.99e-02 (original run with non-deterministic kernels).
