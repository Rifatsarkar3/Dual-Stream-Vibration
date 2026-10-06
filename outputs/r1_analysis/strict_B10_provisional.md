# Strict bearing-level inference (reference: outputs/r1_final_B10/final20)

Status: PROVISIONAL: baseline family incomplete, missing MSCNN [7].

Baselines used: 1D CNN = outputs/r1_reeval/baseline_cnn; BiLSTM = outputs/r1_reeval/baseline_bilstm; CNN-LSTM = outputs/r1_reeval/baseline4f_cnnlstm; 1D ResNet = outputs/r1_reeval/baseline_resnet; Transformer = outputs/r1_reeval/baseline4f_transformer; TSMC-CNN [21] = outputs/r1_reeval/published_tsmccnn

Δ% = (comparator − reference) / comparator on bearing-averaged MAE; positive: reference better. flip p: exact two-sided sign-flip permutation test over bearings. Holm: adjusted within each family (baselines, ablations, protocol). Bootstrap interval shown for continuity only.

## all10 (10 bearings), CES MAE (smallest attainable flip p = 0.0020)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1516 | 0.1616 | +6.1 | 7/10 | 0.0762 | 0.1523 | 0.3438 | [+0.0014, +0.0202] | 20 |
| BiLSTM | 0.1516 | 0.1630 | +7.0 | 7/10 | 0.1426 | 0.1523 | 0.3438 | [+0.0003, +0.0237] | 20 |
| CNN-LSTM | 0.1516 | 0.1667 | +9.0 | 9/10 | 0.0059 | 0.0293 | 0.0215 | [+0.0063, +0.0246] | 20 |
| 1D ResNet | 0.1516 | 0.1929 | +21.4 | 9/10 | 0.0059 | 0.0293 | 0.0215 | [+0.0200, +0.0645] | 20 |
| Transformer | 0.1516 | 0.2685 | +43.5 | 10/10 | 0.0020 | 0.0117 | 0.0020 | [+0.0823, +0.1514] | 20 |
| TSMC-CNN [21] | 0.1516 | 0.2304 | +34.2 | 9/10 | 0.0098 | 0.0293 | 0.0215 | [+0.0295, +0.1316] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1516 | 0.1493 | -1.6 | 2/10 | 0.1348 | 0.8086 | 0.1094 | [-0.0056, -0.0000] | 20 |
| temporal stream only | 0.1516 | 0.1525 | +0.6 | 3/10 | 0.9160 | 1.0000 | 0.3438 | [-0.0112, +0.0160] | 20 |
| spectral stream only | 0.1516 | 0.2318 | +34.6 | 9/10 | 0.0078 | 0.0547 | 0.0215 | [+0.0377, +0.1231] | 20 |
| without PGMC | 0.1516 | 0.1504 | -0.9 | 4/10 | 0.3672 | 1.0000 | 0.7539 | [-0.0037, +0.0014] | 20 |
| without DANN | 0.1516 | 0.1499 | -1.1 | 3/10 | 0.3828 | 1.0000 | 0.3438 | [-0.0052, +0.0017] | 20 |
| without PGMC and DANN | 0.1516 | 0.1490 | -1.8 | 4/10 | 0.2754 | 1.0000 | 0.7539 | [-0.0068, +0.0014] | 20 |
| PGMC batch ordering | 0.1516 | 0.1498 | -1.2 | 3/10 | 0.1562 | 0.8086 | 0.3438 | [-0.0040, +0.0005] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1505 | 0.1763 | +14.6 | 8/10 | 0.0312 | 0.0625 | 0.1094 | [+0.0076, +0.0457] | 5 |
| lambda by validation argmin | 0.1516 | 0.1480 | -2.5 | 6/10 | 0.7734 | 0.7734 | 0.7539 | [-0.0278, +0.0191] | 20 |

## all10 (10 bearings), RAW MAE (smallest attainable flip p = 0.0020)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1536 | 0.1646 | +6.7 | 8/10 | 0.0391 | 0.0625 | 0.1094 | [+0.0026, +0.0208] | 20 |
| BiLSTM | 0.1536 | 0.1651 | +7.0 | 8/10 | 0.0312 | 0.0625 | 0.1094 | [+0.0026, +0.0217] | 20 |
| CNN-LSTM | 0.1536 | 0.1676 | +8.4 | 9/10 | 0.0039 | 0.0195 | 0.0215 | [+0.0064, +0.0221] | 20 |
| 1D ResNet | 0.1536 | 0.1923 | +20.1 | 9/10 | 0.0059 | 0.0234 | 0.0215 | [+0.0186, +0.0608] | 20 |
| Transformer | 0.1536 | 0.2742 | +44.0 | 10/10 | 0.0020 | 0.0117 | 0.0020 | [+0.0881, +0.1529] | 20 |
| TSMC-CNN [21] | 0.1536 | 0.2300 | +33.2 | 9/10 | 0.0098 | 0.0293 | 0.0215 | [+0.0322, +0.1233] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1536 | 0.1511 | -1.6 | 2/10 | 0.1152 | 0.6914 | 0.1094 | [-0.0057, -0.0001] | 20 |
| temporal stream only | 0.1536 | 0.1538 | +0.1 | 3/10 | 0.9824 | 0.9824 | 0.3438 | [-0.0117, +0.0156] | 20 |
| spectral stream only | 0.1536 | 0.2395 | +35.9 | 9/10 | 0.0039 | 0.0273 | 0.0215 | [+0.0457, +0.1268] | 20 |
| without PGMC | 0.1536 | 0.1526 | -0.7 | 4/10 | 0.4473 | 0.8945 | 0.7539 | [-0.0034, +0.0015] | 20 |
| without DANN | 0.1536 | 0.1511 | -1.7 | 3/10 | 0.1934 | 0.7617 | 0.3438 | [-0.0058, +0.0008] | 20 |
| without PGMC and DANN | 0.1536 | 0.1503 | -2.2 | 3/10 | 0.1523 | 0.7617 | 0.3438 | [-0.0073, +0.0006] | 20 |
| PGMC batch ordering | 0.1536 | 0.1519 | -1.1 | 3/10 | 0.1836 | 0.7617 | 0.3438 | [-0.0038, +0.0005] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1525 | 0.1769 | +13.8 | 8/10 | 0.0371 | 0.0742 | 0.1094 | [+0.0071, +0.0432] | 5 |
| lambda by validation argmin | 0.1536 | 0.1501 | -2.3 | 6/10 | 0.7734 | 0.7734 | 0.7539 | [-0.0265, +0.0179] | 20 |

## new6 (6 bearings), CES MAE (smallest attainable flip p = 0.0312)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1815 | 0.1855 | +2.2 | 4/6 | 0.4375 | 0.5000 | 0.6875 | [-0.0027, +0.0117] | 20 |
| BiLSTM | 0.1815 | 0.1937 | +6.3 | 5/6 | 0.2500 | 0.5000 | 0.2188 | [-0.0011, +0.0285] | 20 |
| CNN-LSTM | 0.1815 | 0.1948 | +6.9 | 5/6 | 0.0625 | 0.2500 | 0.2188 | [+0.0033, +0.0247] | 20 |
| 1D ResNet | 0.1815 | 0.2163 | +16.1 | 6/6 | 0.0312 | 0.1875 | 0.0312 | [+0.0168, +0.0536] | 20 |
| Transformer | 0.1815 | 0.2740 | +33.8 | 6/6 | 0.0312 | 0.1875 | 0.0312 | [+0.0539, +0.1317] | 20 |
| TSMC-CNN [21] | 0.1815 | 0.2112 | +14.1 | 5/6 | 0.1562 | 0.4688 | 0.2188 | [-0.0056, +0.0664] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1815 | 0.1814 | -0.0 | 2/6 | 0.9688 | 1.0000 | 0.6875 | [-0.0008, +0.0010] | 20 |
| temporal stream only | 0.1815 | 0.1851 | +2.0 | 2/6 | 0.9688 | 1.0000 | 0.6875 | [-0.0101, +0.0260] | 20 |
| spectral stream only | 0.1815 | 0.2446 | +25.8 | 5/6 | 0.0938 | 0.6562 | 0.2188 | [+0.0131, +0.1174] | 20 |
| without PGMC | 0.1815 | 0.1821 | +0.4 | 4/6 | 0.7500 | 1.0000 | 0.6875 | [-0.0030, +0.0040] | 20 |
| without DANN | 0.1815 | 0.1812 | -0.1 | 3/6 | 0.9062 | 1.0000 | 1.0000 | [-0.0060, +0.0048] | 20 |
| without PGMC and DANN | 0.1815 | 0.1813 | -0.1 | 4/6 | 0.9688 | 1.0000 | 0.6875 | [-0.0064, +0.0049] | 20 |
| PGMC batch ordering | 0.1815 | 0.1809 | -0.3 | 3/6 | 0.7188 | 1.0000 | 1.0000 | [-0.0039, +0.0025] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1822 | 0.1937 | +5.9 | 5/6 | 0.2188 | 0.4375 | 0.2188 | [-0.0019, +0.0296] | 5 |
| lambda by validation argmin | 0.1815 | 0.1837 | +1.2 | 4/6 | 0.9375 | 0.9375 | 0.6875 | [-0.0323, +0.0362] | 20 |

## new6 (6 bearings), RAW MAE (smallest attainable flip p = 0.0312)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1824 | 0.1875 | +2.7 | 5/6 | 0.2188 | 0.3750 | 0.2188 | [-0.0015, +0.0125] | 20 |
| BiLSTM | 0.1824 | 0.1941 | +6.0 | 5/6 | 0.1250 | 0.3750 | 0.2188 | [+0.0012, +0.0239] | 20 |
| CNN-LSTM | 0.1824 | 0.1944 | +6.2 | 5/6 | 0.0625 | 0.2500 | 0.2188 | [+0.0040, +0.0205] | 20 |
| 1D ResNet | 0.1824 | 0.2151 | +15.2 | 6/6 | 0.0312 | 0.1875 | 0.0312 | [+0.0155, +0.0511] | 20 |
| Transformer | 0.1824 | 0.2795 | +34.7 | 6/6 | 0.0312 | 0.1875 | 0.0312 | [+0.0605, +0.1357] | 20 |
| TSMC-CNN [21] | 0.1824 | 0.2140 | +14.7 | 5/6 | 0.1562 | 0.3750 | 0.2188 | [+0.0019, +0.0647] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1824 | 0.1823 | -0.1 | 2/6 | 0.8750 | 1.0000 | 0.6875 | [-0.0010, +0.0010] | 20 |
| temporal stream only | 0.1824 | 0.1854 | +1.6 | 2/6 | 0.9688 | 1.0000 | 0.6875 | [-0.0110, +0.0257] | 20 |
| spectral stream only | 0.1824 | 0.2514 | +27.4 | 5/6 | 0.0625 | 0.4375 | 0.2188 | [+0.0219, +0.1192] | 20 |
| without PGMC | 0.1824 | 0.1831 | +0.3 | 4/6 | 0.7812 | 1.0000 | 0.6875 | [-0.0030, +0.0041] | 20 |
| without DANN | 0.1824 | 0.1814 | -0.6 | 3/6 | 0.6875 | 1.0000 | 1.0000 | [-0.0065, +0.0037] | 20 |
| without PGMC and DANN | 0.1824 | 0.1816 | -0.5 | 3/6 | 0.8125 | 1.0000 | 1.0000 | [-0.0070, +0.0041] | 20 |
| PGMC batch ordering | 0.1824 | 0.1818 | -0.4 | 3/6 | 0.7188 | 1.0000 | 1.0000 | [-0.0038, +0.0024] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1831 | 0.1942 | +5.7 | 5/6 | 0.2812 | 0.5625 | 0.2188 | [-0.0022, +0.0287] | 5 |
| lambda by validation argmin | 0.1824 | 0.1836 | +0.6 | 4/6 | 0.9375 | 0.9375 | 0.6875 | [-0.0328, +0.0339] | 20 |

## orig4 (4 bearings), CES MAE (smallest attainable flip p = 0.1250)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1069 | 0.1257 | +14.9 | 3/4 | 0.2500 | 0.7500 | 0.6250 | [+0.0025, +0.0370] | 20 |
| BiLSTM | 0.1069 | 0.1168 | +8.5 | 2/4 | 0.7500 | 0.7500 | 1.0000 | [-0.0054, +0.0329] | 20 |
| CNN-LSTM | 0.1069 | 0.1246 | +14.2 | 4/4 | 0.1250 | 0.7500 | 0.1250 | [+0.0018, +0.0335] | 20 |
| 1D ResNet | 0.1069 | 0.1578 | +32.2 | 3/4 | 0.2500 | 0.7500 | 0.6250 | [+0.0071, +0.0947] | 20 |
| Transformer | 0.1069 | 0.2604 | +58.9 | 4/4 | 0.1250 | 0.7500 | 0.1250 | [+0.1083, +0.1966] | 20 |
| TSMC-CNN [21] | 0.1069 | 0.2593 | +58.8 | 4/4 | 0.1250 | 0.7500 | 0.1250 | [+0.0783, +0.2135] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1069 | 0.1010 | -5.8 | 0/4 | 0.1250 | 0.8750 | 0.1250 | [-0.0114, -0.0010] | 20 |
| temporal stream only | 0.1069 | 0.1036 | -3.2 | 1/4 | 0.6250 | 0.8750 | 0.6250 | [-0.0227, +0.0146] | 20 |
| spectral stream only | 0.1069 | 0.2126 | +49.7 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0442, +0.1671] | 20 |
| without PGMC | 0.1069 | 0.1028 | -4.0 | 0/4 | 0.1250 | 0.8750 | 0.1250 | [-0.0044, -0.0040] | 20 |
| without DANN | 0.1069 | 0.1030 | -3.8 | 0/4 | 0.1250 | 0.8750 | 0.1250 | [-0.0061, -0.0023] | 20 |
| without PGMC and DANN | 0.1069 | 0.1006 | -6.3 | 0/4 | 0.1250 | 0.8750 | 0.1250 | [-0.0103, -0.0030] | 20 |
| PGMC batch ordering | 0.1069 | 0.1032 | -3.6 | 0/4 | 0.1250 | 0.8750 | 0.1250 | [-0.0054, -0.0013] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1030 | 0.1501 | +31.4 | 3/4 | 0.2500 | 0.5000 | 0.6250 | [+0.0123, +0.0777] | 5 |
| lambda by validation argmin | 0.1069 | 0.0944 | -13.2 | 2/4 | 0.5000 | 0.5000 | 1.0000 | [-0.0304, +0.0033] | 20 |

## orig4 (4 bearings), RAW MAE (smallest attainable flip p = 0.1250)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1103 | 0.1302 | +15.3 | 3/4 | 0.2500 | 0.7500 | 0.6250 | [+0.0042, +0.0365] | 20 |
| BiLSTM | 0.1103 | 0.1216 | +9.3 | 3/4 | 0.3750 | 0.7500 | 0.6250 | [-0.0009, +0.0312] | 20 |
| CNN-LSTM | 0.1103 | 0.1273 | +13.4 | 4/4 | 0.1250 | 0.7500 | 0.1250 | [+0.0021, +0.0319] | 20 |
| 1D ResNet | 0.1103 | 0.1581 | +30.2 | 3/4 | 0.2500 | 0.7500 | 0.6250 | [+0.0062, +0.0894] | 20 |
| Transformer | 0.1103 | 0.2662 | +58.6 | 4/4 | 0.1250 | 0.7500 | 0.1250 | [+0.1163, +0.1954] | 20 |
| TSMC-CNN [21] | 0.1103 | 0.2540 | +56.6 | 4/4 | 0.1250 | 0.7500 | 0.1250 | [+0.0726, +0.1892] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1103 | 0.1043 | -5.8 | 0/4 | 0.1250 | 0.8750 | 0.1250 | [-0.0117, -0.0013] | 20 |
| temporal stream only | 0.1103 | 0.1063 | -3.8 | 1/4 | 0.6250 | 0.8750 | 0.6250 | [-0.0220, +0.0136] | 20 |
| spectral stream only | 0.1103 | 0.2215 | +50.2 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0525, +0.1698] | 20 |
| without PGMC | 0.1103 | 0.1068 | -3.3 | 0/4 | 0.1250 | 0.8750 | 0.1250 | [-0.0036, -0.0034] | 20 |
| without DANN | 0.1103 | 0.1055 | -4.5 | 0/4 | 0.1250 | 0.8750 | 0.1250 | [-0.0065, -0.0036] | 20 |
| without PGMC and DANN | 0.1103 | 0.1033 | -6.8 | 0/4 | 0.1250 | 0.8750 | 0.1250 | [-0.0102, -0.0043] | 20 |
| PGMC batch ordering | 0.1103 | 0.1071 | -3.0 | 0/4 | 0.1250 | 0.8750 | 0.1250 | [-0.0052, -0.0012] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1066 | 0.1509 | +29.4 | 3/4 | 0.2500 | 0.5000 | 0.6250 | [+0.0111, +0.0738] | 5 |
| lambda by validation argmin | 0.1103 | 0.0998 | -10.5 | 2/4 | 0.5000 | 0.5000 | 1.0000 | [-0.0262, +0.0016] | 20 |

## zs3 (3 bearings), CES MAE (smallest attainable flip p = 0.2500)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1569 | 0.1507 | -4.1 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0588, +0.0686] | 20 |
| BiLSTM | 0.1569 | 0.1401 | -12.0 | 1/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0599, +0.0575] | 20 |
| CNN-LSTM | 0.1569 | 0.1512 | -3.8 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0537, +0.0678] | 20 |
| 1D ResNet | 0.1569 | 0.1792 | +12.5 | 2/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0693, +0.1100] | 20 |
| Transformer | 0.1569 | 0.2914 | +46.2 | 2/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0047, +0.2759] | 20 |
| TSMC-CNN [21] | 0.1569 | 0.1561 | -0.5 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0828, +0.1006] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1569 | 0.1586 | +1.1 | 2/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0021, +0.0064] | 20 |
| temporal stream only | 0.1569 | 0.1567 | -0.1 | 2/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0298, +0.0179] | 20 |
| spectral stream only | 0.1569 | 0.1788 | +12.3 | 2/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0374, +0.0966] | 20 |
| without PGMC | 0.1569 | 0.1628 | +3.6 | 2/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0004, +0.0118] | 20 |
| without DANN | 0.1569 | 0.1593 | +1.5 | 2/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0120, +0.0172] | 20 |
| without PGMC and DANN | 0.1569 | 0.1607 | +2.4 | 2/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0093, +0.0119] | 20 |
| PGMC batch ordering | 0.1569 | 0.1591 | +1.4 | 3/3 | 0.2500 | 1.0000 | 0.2500 | [+0.0001, +0.0049] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1636 | 0.1349 | -21.2 | 1/3 | 0.7500 | 0.7500 | 1.0000 | [-0.1040, +0.0762] | 5 |
| lambda by validation argmin | 0.1569 | 0.2051 | +23.5 | 3/3 | 0.2500 | 0.5000 | 0.2500 | [+0.0305, +0.0656] | 20 |

## zs3 (3 bearings), RAW MAE (smallest attainable flip p = 0.2500)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1630 | 0.1561 | -4.5 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0568, +0.0645] | 20 |
| BiLSTM | 0.1630 | 0.1456 | -12.0 | 1/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0574, +0.0502] | 20 |
| CNN-LSTM | 0.1630 | 0.1553 | -5.0 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0510, +0.0590] | 20 |
| 1D ResNet | 0.1630 | 0.1798 | +9.3 | 2/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0716, +0.0993] | 20 |
| Transformer | 0.1630 | 0.2970 | +45.1 | 3/3 | 0.2500 | 1.0000 | 0.2500 | [+0.0073, +0.2622] | 20 |
| TSMC-CNN [21] | 0.1630 | 0.1574 | -3.6 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0857, +0.0905] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1630 | 0.1646 | +0.9 | 2/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0021, +0.0065] | 20 |
| temporal stream only | 0.1630 | 0.1616 | -0.9 | 2/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0286, +0.0129] | 20 |
| spectral stream only | 0.1630 | 0.2033 | +19.8 | 2/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0311, +0.1299] | 20 |
| without PGMC | 0.1630 | 0.1690 | +3.5 | 3/3 | 0.2500 | 1.0000 | 0.2500 | [+0.0002, +0.0114] | 20 |
| without DANN | 0.1630 | 0.1641 | +0.6 | 2/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0128, +0.0146] | 20 |
| without PGMC and DANN | 0.1630 | 0.1657 | +1.6 | 2/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0099, +0.0097] | 20 |
| PGMC batch ordering | 0.1630 | 0.1652 | +1.3 | 3/3 | 0.2500 | 1.0000 | 0.2500 | [+0.0002, +0.0048] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1695 | 0.1400 | -21.0 | 1/3 | 0.7500 | 0.7500 | 1.0000 | [-0.1018, +0.0703] | 5 |
| lambda by validation argmin | 0.1630 | 0.2103 | +22.5 | 3/3 | 0.2500 | 0.5000 | 0.2500 | [+0.0305, +0.0633] | 20 |
