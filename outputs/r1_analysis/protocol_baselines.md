# In-bearing early stopping for every baseline (revision R1, Amendment 1, E3)

Reference: each baseline's final runs (fixed budget selected on validation bearings). Comparator: in-bearing early stopping (every 10th window of each training bearing as validation, patience 30, at most 120 epochs, best-validation checkpoint). Five shared seeds. Δ = (early stopping − final) / early stopping on bearing-averaged MAE; positive: early stopping worse. p: exact sign-flip test over bearings; Holm over the seven baselines (all10).

## all10 (10 bearings; smallest attainable p = 0.0020)

| Model | final, CES bearing avg | early stopping, CES | Δ CES | early stopping worse on | p | Holm | Δ raw | p raw | Holm raw |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1639 | 0.1713 | +4.3 % | 7/10 | 0.059 | 0.305 | +4.2 % | 0.053 | 0.264 |
| BiLSTM | 0.1795 | 0.1951 | +8.0 % | 7/10 | 0.027 | 0.191 | +8.1 % | 0.018 | 0.123 |
| CNN-LSTM | 0.1657 | 0.1852 | +10.5 % | 7/10 | 0.051 | 0.305 | +10.5 % | 0.041 | 0.246 |
| 1D ResNet | 0.1933 | 0.2030 | +4.8 % | 6/10 | 0.303 | 0.621 | +5.4 % | 0.242 | 0.650 |
| Transformer | 0.2616 | 0.2325 | -12.5 % | 4/10 | 0.207 | 0.621 | -10.2 % | 0.217 | 0.650 |
| TSMC-CNN [21] | 0.2357 | 0.2333 | -1.0 % | 5/10 | 0.889 | 0.889 | -1.9 % | 0.746 | 0.746 |
| MSCNN [7] | 0.2121 | 0.1824 | -16.3 % | 3/10 | 0.121 | 0.484 | -15.6 % | 0.125 | 0.500 |
| DSAFLite+PGMC, submitted architecture (plan Section 8) | 0.1505 | 0.1763 | +14.6 % | 8/10 | 0.031 | – | | | |
| DSAFLite+PGMC with temporal context (adopted) (plan Section 8) | 0.1553 | 0.1857 | +16.4 % | 8/10 | 0.010 | – | | | |

In-bearing early stopping gives the higher all10 CES error for 4 of the 7 baselines and for the submitted architecture: 5 of 8 models.

## new6 (6 bearings; smallest attainable p = 0.0312)

| Model | final, CES bearing avg | early stopping, CES | Δ CES | early stopping worse on | p | Holm | Δ raw | p raw | Holm raw |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1866 | 0.1889 | +1.2 % | 3/6 | 0.562 | 1.000 | +1.5 % | 0.531 | 1.000 |
| BiLSTM | 0.2018 | 0.2171 | +7.0 % | 4/6 | 0.125 | 0.875 | +7.2 % | 0.062 | 0.438 |
| CNN-LSTM | 0.1968 | 0.2107 | +6.6 % | 4/6 | 0.312 | 1.000 | +7.2 % | 0.250 | 1.000 |
| 1D ResNet | 0.2174 | 0.2250 | +3.4 % | 3/6 | 0.688 | 1.000 | +3.9 % | 0.594 | 1.000 |
| Transformer | 0.2610 | 0.2492 | -4.7 % | 3/6 | 0.656 | 1.000 | -4.5 % | 0.594 | 1.000 |
| TSMC-CNN [21] | 0.2136 | 0.2193 | +2.6 % | 3/6 | 0.688 | 1.000 | +2.5 % | 0.688 | 1.000 |
| MSCNN [7] | 0.2186 | 0.1844 | -18.6 % | 1/6 | 0.125 | 0.875 | -18.0 % | 0.062 | 0.438 |
| DSAFLite+PGMC, submitted architecture (plan Section 8) | 0.1822 | 0.1937 | +5.9 % | 5/6 | 0.219 | – | | | |
| DSAFLite+PGMC with temporal context (adopted) (plan Section 8) | 0.1866 | 0.2031 | +8.1 % | 4/6 | 0.156 | – | | | |

## orig4 (4 bearings; smallest attainable p = 0.1250)

| Model | final, CES bearing avg | early stopping, CES | Δ CES | early stopping worse on | p | Holm | Δ raw | p raw | Holm raw |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1298 | 0.1448 | +10.3 % | 4/4 | 0.125 | 0.875 | +9.6 % | 0.125 | 0.875 |
| BiLSTM | 0.1460 | 0.1623 | +10.0 % | 3/4 | 0.375 | 1.000 | +9.9 % | 0.375 | 1.000 |
| CNN-LSTM | 0.1191 | 0.1469 | +18.9 % | 3/4 | 0.250 | 1.000 | +17.5 % | 0.250 | 1.000 |
| 1D ResNet | 0.1571 | 0.1701 | +7.6 % | 3/4 | 0.375 | 1.000 | +8.3 % | 0.375 | 1.000 |
| Transformer | 0.2627 | 0.2075 | -26.6 % | 1/4 | 0.250 | 1.000 | -19.9 % | 0.375 | 1.000 |
| TSMC-CNN [21] | 0.2688 | 0.2542 | -5.8 % | 2/4 | 0.875 | 1.000 | -7.9 % | 0.625 | 1.000 |
| MSCNN [7] | 0.2024 | 0.1794 | -12.9 % | 2/4 | 0.625 | 1.000 | -12.0 % | 0.625 | 1.000 |
| DSAFLite+PGMC, submitted architecture (plan Section 8) | 0.1030 | 0.1501 | +31.4 % | 3/4 | 0.250 | – | | | |
| DSAFLite+PGMC with temporal context (adopted) (plan Section 8) | 0.1083 | 0.1597 | +32.2 % | 4/4 | 0.125 | – | | | |

## zs3 (3 bearings; smallest attainable p = 0.2500)

| Model | final, CES bearing avg | early stopping, CES | Δ CES | early stopping worse on | p | Holm | Δ raw | p raw | Holm raw |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1493 | 0.1462 | -2.1 % | 1/3 | 0.500 | 1.000 | -2.5 % | 0.250 | 1.000 |
| BiLSTM | 0.1496 | 0.1600 | +6.5 % | 2/3 | 0.500 | 1.000 | +6.8 % | 0.500 | 1.000 |
| CNN-LSTM | 0.1518 | 0.1751 | +13.3 % | 2/3 | 0.500 | 1.000 | +12.6 % | 0.500 | 1.000 |
| 1D ResNet | 0.1740 | 0.1950 | +10.7 % | 2/3 | 0.500 | 1.000 | +11.5 % | 0.500 | 1.000 |
| Transformer | 0.3155 | 0.3368 | +6.3 % | 3/3 | 0.250 | 1.000 | +4.8 % | 0.250 | 1.000 |
| TSMC-CNN [21] | 0.1551 | 0.1507 | -2.9 % | 1/3 | 0.500 | 1.000 | -1.9 % | 0.750 | 1.000 |
| MSCNN [7] | 0.2248 | 0.2011 | -11.8 % | 1/3 | 0.500 | 1.000 | -11.1 % | 0.500 | 1.000 |
| DSAFLite+PGMC, submitted architecture (plan Section 8) | 0.1636 | 0.1349 | -21.2 % | 1/3 | 0.750 | – | | | |
| DSAFLite+PGMC with temporal context (adopted) (plan Section 8) | 0.1863 | 0.1084 | -71.9 % | 1/3 | 0.500 | – | | | |
