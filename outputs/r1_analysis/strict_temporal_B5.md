# Strict bearing-level inference (reference: outputs/r1_final_temporal_B5/final20)

Status: complete.

Baselines used: 1D CNN = outputs/r1_reeval/baseline_cnn; BiLSTM = outputs/r1_baselines_final5/bilstm; CNN-LSTM = outputs/r1_baselines_final5/cnnlstm; 1D ResNet = outputs/r1_reeval/baseline_resnet; Transformer = outputs/r1_baselines_final5/transformer; TSMC-CNN [21] = outputs/r1_reeval/published_tsmccnn; MSCNN [7] = outputs/r1_mscnn/final

Δ% = (comparator − reference) / comparator on bearing-averaged MAE; positive: reference better. flip p: exact two-sided sign-flip permutation test over bearings. Holm: adjusted within each family (baselines, ablations, protocol). Bootstrap interval shown for continuity only.

## all10 (10 bearings), CES MAE (smallest attainable flip p = 0.0020)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1535 | 0.1616 | +5.0 | 8/10 | 0.1758 | 0.3516 | 0.1094 | [-0.0014, +0.0190] | 20 |
| BiLSTM | 0.1535 | 0.1782 | +13.9 | 8/10 | 0.0137 | 0.0684 | 0.1094 | [+0.0084, +0.0444] | 20 |
| CNN-LSTM | 0.1535 | 0.1637 | +6.2 | 5/10 | 0.1777 | 0.3516 | 1.0000 | [-0.0029, +0.0233] | 20 |
| 1D ResNet | 0.1535 | 0.1929 | +20.4 | 8/10 | 0.0078 | 0.0469 | 0.1094 | [+0.0180, +0.0637] | 20 |
| Transformer | 0.1535 | 0.2509 | +38.8 | 10/10 | 0.0020 | 0.0137 | 0.0020 | [+0.0667, +0.1279] | 20 |
| TSMC-CNN [21] | 0.1535 | 0.2304 | +33.4 | 8/10 | 0.0293 | 0.1172 | 0.1094 | [+0.0238, +0.1342] | 20 |
| MSCNN [7] | 0.1535 | 0.2109 | +27.2 | 8/10 | 0.0742 | 0.2227 | 0.1094 | [-0.0019, +0.1034] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1535 | 0.1495 | -2.7 | 2/10 | 0.1426 | 0.8555 | 0.1094 | [-0.0092, +0.0000] | 20 |
| temporal stream only | 0.1535 | 0.1569 | +2.2 | 5/10 | 0.9004 | 1.0000 | 1.0000 | [-0.0188, +0.0329] | 20 |
| spectral stream only | 0.1535 | 0.2295 | +33.1 | 8/10 | 0.0117 | 0.0820 | 0.1094 | [+0.0325, +0.1201] | 20 |
| without PGMC | 0.1535 | 0.1550 | +1.0 | 7/10 | 0.2871 | 1.0000 | 0.3438 | [-0.0011, +0.0039] | 20 |
| without DANN | 0.1535 | 0.1537 | +0.1 | 5/10 | 0.9473 | 1.0000 | 1.0000 | [-0.0037, +0.0040] | 20 |
| without PGMC and DANN | 0.1535 | 0.1560 | +1.6 | 7/10 | 0.2559 | 1.0000 | 0.3438 | [-0.0016, +0.0062] | 20 |
| PGMC batch ordering | 0.1535 | 0.1544 | +0.6 | 7/10 | 0.6465 | 1.0000 | 0.3438 | [-0.0023, +0.0043] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1553 | 0.1857 | +16.4 | 8/10 | 0.0098 | 0.0098 | 0.1094 | [+0.0127, +0.0483] | 5 |

## all10 (10 bearings), RAW MAE (smallest attainable flip p = 0.0020)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1533 | 0.1646 | +6.8 | 9/10 | 0.0293 | 0.0879 | 0.0215 | [+0.0026, +0.0218] | 20 |
| BiLSTM | 0.1533 | 0.1774 | +13.6 | 9/10 | 0.0098 | 0.0488 | 0.0215 | [+0.0090, +0.0430] | 20 |
| CNN-LSTM | 0.1533 | 0.1658 | +7.5 | 6/10 | 0.0898 | 0.1445 | 0.7539 | [+0.0006, +0.0244] | 20 |
| 1D ResNet | 0.1533 | 0.1923 | +20.3 | 8/10 | 0.0078 | 0.0469 | 0.1094 | [+0.0180, +0.0627] | 20 |
| Transformer | 0.1533 | 0.2596 | +40.9 | 10/10 | 0.0020 | 0.0137 | 0.0020 | [+0.0768, +0.1354] | 20 |
| TSMC-CNN [21] | 0.1533 | 0.2300 | +33.3 | 8/10 | 0.0176 | 0.0703 | 0.1094 | [+0.0288, +0.1268] | 20 |
| MSCNN [7] | 0.1533 | 0.2119 | +27.6 | 8/10 | 0.0723 | 0.1445 | 0.1094 | [+0.0005, +0.1040] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1533 | 0.1491 | -2.9 | 2/10 | 0.1191 | 0.7148 | 0.1094 | [-0.0095, -0.0001] | 20 |
| temporal stream only | 0.1533 | 0.1564 | +2.0 | 5/10 | 0.9102 | 1.0000 | 1.0000 | [-0.0193, +0.0332] | 20 |
| spectral stream only | 0.1533 | 0.2304 | +33.5 | 8/10 | 0.0098 | 0.0684 | 0.1094 | [+0.0347, +0.1202] | 20 |
| without PGMC | 0.1533 | 0.1548 | +1.0 | 7/10 | 0.2812 | 1.0000 | 0.3438 | [-0.0010, +0.0039] | 20 |
| without DANN | 0.1533 | 0.1531 | -0.1 | 5/10 | 0.9199 | 1.0000 | 1.0000 | [-0.0039, +0.0035] | 20 |
| without PGMC and DANN | 0.1533 | 0.1555 | +1.4 | 7/10 | 0.3242 | 1.0000 | 0.3438 | [-0.0020, +0.0057] | 20 |
| PGMC batch ordering | 0.1533 | 0.1542 | +0.6 | 7/10 | 0.6426 | 1.0000 | 0.3438 | [-0.0022, +0.0043] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1551 | 0.1846 | +16.0 | 8/10 | 0.0117 | 0.0117 | 0.1094 | [+0.0119, +0.0471] | 5 |

## new6 (6 bearings), CES MAE (smallest attainable flip p = 0.0312)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1855 | 0.1855 | -0.0 | 4/6 | 1.0000 | 1.0000 | 0.6875 | [-0.0086, +0.0062] | 20 |
| BiLSTM | 0.1855 | 0.2029 | +8.6 | 4/6 | 0.1875 | 0.9375 | 0.6875 | [+0.0010, +0.0353] | 20 |
| CNN-LSTM | 0.1855 | 0.1958 | +5.3 | 3/6 | 0.3750 | 1.0000 | 1.0000 | [-0.0078, +0.0285] | 20 |
| 1D ResNet | 0.1855 | 0.2163 | +14.3 | 4/6 | 0.1250 | 0.7500 | 0.6875 | [+0.0096, +0.0537] | 20 |
| Transformer | 0.1855 | 0.2538 | +26.9 | 6/6 | 0.0312 | 0.2188 | 0.0312 | [+0.0401, +0.0965] | 20 |
| TSMC-CNN [21] | 0.1855 | 0.2112 | +12.2 | 4/6 | 0.3438 | 1.0000 | 0.6875 | [-0.0134, +0.0704] | 20 |
| MSCNN [7] | 0.1855 | 0.2142 | +13.4 | 4/6 | 0.5312 | 1.0000 | 0.6875 | [-0.0472, +0.1029] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1855 | 0.1852 | -0.2 | 1/6 | 0.7812 | 1.0000 | 0.2188 | [-0.0022, +0.0019] | 20 |
| temporal stream only | 0.1855 | 0.1970 | +5.9 | 3/6 | 0.8438 | 1.0000 | 1.0000 | [-0.0238, +0.0562] | 20 |
| spectral stream only | 0.1855 | 0.2405 | +22.9 | 4/6 | 0.1875 | 1.0000 | 0.6875 | [-0.0002, +0.1136] | 20 |
| without PGMC | 0.1855 | 0.1856 | +0.1 | 3/6 | 0.9062 | 1.0000 | 1.0000 | [-0.0033, +0.0038] | 20 |
| without DANN | 0.1855 | 0.1836 | -1.0 | 2/6 | 0.5312 | 1.0000 | 0.6875 | [-0.0070, +0.0041] | 20 |
| without PGMC and DANN | 0.1855 | 0.1848 | -0.3 | 3/6 | 0.8125 | 1.0000 | 1.0000 | [-0.0056, +0.0040] | 20 |
| PGMC batch ordering | 0.1855 | 0.1838 | -0.9 | 3/6 | 0.4688 | 1.0000 | 1.0000 | [-0.0049, +0.0014] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1866 | 0.2031 | +8.1 | 4/6 | 0.1562 | 0.1562 | 0.6875 | [+0.0007, +0.0365] | 5 |

## new6 (6 bearings), RAW MAE (smallest attainable flip p = 0.0312)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1844 | 0.1875 | +1.6 | 5/6 | 0.3438 | 1.0000 | 0.2188 | [-0.0037, +0.0083] | 20 |
| BiLSTM | 0.1844 | 0.2011 | +8.3 | 5/6 | 0.1562 | 0.7812 | 0.2188 | [+0.0021, +0.0320] | 20 |
| CNN-LSTM | 0.1844 | 0.1964 | +6.1 | 4/6 | 0.2500 | 1.0000 | 0.6875 | [-0.0042, +0.0285] | 20 |
| 1D ResNet | 0.1844 | 0.2151 | +14.3 | 4/6 | 0.1250 | 0.7500 | 0.6875 | [+0.0098, +0.0526] | 20 |
| Transformer | 0.1844 | 0.2636 | +30.1 | 6/6 | 0.0312 | 0.2188 | 0.0312 | [+0.0511, +0.1079] | 20 |
| TSMC-CNN [21] | 0.1844 | 0.2140 | +13.8 | 4/6 | 0.2812 | 1.0000 | 0.6875 | [-0.0050, +0.0700] | 20 |
| MSCNN [7] | 0.1844 | 0.2142 | +13.9 | 4/6 | 0.5312 | 1.0000 | 0.6875 | [-0.0442, +0.1023] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1844 | 0.1838 | -0.3 | 1/6 | 0.6562 | 1.0000 | 0.2188 | [-0.0028, +0.0022] | 20 |
| temporal stream only | 0.1844 | 0.1958 | +5.8 | 3/6 | 0.8438 | 1.0000 | 1.0000 | [-0.0245, +0.0577] | 20 |
| spectral stream only | 0.1844 | 0.2405 | +23.3 | 4/6 | 0.1562 | 1.0000 | 0.6875 | [+0.0032, +0.1119] | 20 |
| without PGMC | 0.1844 | 0.1846 | +0.1 | 3/6 | 0.9062 | 1.0000 | 1.0000 | [-0.0032, +0.0037] | 20 |
| without DANN | 0.1844 | 0.1824 | -1.1 | 2/6 | 0.5312 | 1.0000 | 0.6875 | [-0.0070, +0.0039] | 20 |
| without PGMC and DANN | 0.1844 | 0.1836 | -0.4 | 3/6 | 0.8125 | 1.0000 | 1.0000 | [-0.0057, +0.0038] | 20 |
| PGMC batch ordering | 0.1844 | 0.1828 | -0.8 | 3/6 | 0.4688 | 1.0000 | 1.0000 | [-0.0048, +0.0015] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1855 | 0.2015 | +7.9 | 4/6 | 0.1562 | 0.1562 | 0.6875 | [+0.0003, +0.0353] | 5 |

## orig4 (4 bearings), CES MAE (smallest attainable flip p = 0.1250)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1055 | 0.1257 | +16.0 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0067, +0.0386] | 20 |
| BiLSTM | 0.1055 | 0.1411 | +25.2 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0122, +0.0762] | 20 |
| CNN-LSTM | 0.1055 | 0.1155 | +8.7 | 2/4 | 0.5000 | 0.8750 | 1.0000 | [-0.0075, +0.0275] | 20 |
| 1D ResNet | 0.1055 | 0.1578 | +33.1 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0116, +0.0929] | 20 |
| Transformer | 0.1055 | 0.2467 | +57.2 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.1099, +0.1642] | 20 |
| TSMC-CNN [21] | 0.1055 | 0.2593 | +59.3 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0750, +0.2208] | 20 |
| MSCNN [7] | 0.1055 | 0.2059 | +48.7 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0780, +0.1150] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1055 | 0.0960 | -10.0 | 1/4 | 0.2500 | 0.8750 | 0.6250 | [-0.0181, -0.0014] | 20 |
| temporal stream only | 0.1055 | 0.0967 | -9.1 | 2/4 | 0.5000 | 0.8750 | 1.0000 | [-0.0289, +0.0112] | 20 |
| spectral stream only | 0.1055 | 0.2130 | +50.5 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0532, +0.1617] | 20 |
| without PGMC | 0.1055 | 0.1091 | +3.3 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0015, +0.0055] | 20 |
| without DANN | 0.1055 | 0.1088 | +3.0 | 3/4 | 0.2500 | 0.8750 | 0.6250 | [+0.0003, +0.0057] | 20 |
| without PGMC and DANN | 0.1055 | 0.1128 | +6.5 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0048, +0.0098] | 20 |
| PGMC batch ordering | 0.1055 | 0.1102 | +4.2 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0014, +0.0100] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1083 | 0.1597 | +32.2 | 4/4 | 0.1250 | 0.1250 | 0.1250 | [+0.0236, +0.0668] | 5 |

## orig4 (4 bearings), RAW MAE (smallest attainable flip p = 0.1250)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1068 | 0.1302 | +18.0 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0107, +0.0410] | 20 |
| BiLSTM | 0.1068 | 0.1419 | +24.8 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0123, +0.0750] | 20 |
| CNN-LSTM | 0.1068 | 0.1199 | +10.9 | 2/4 | 0.5000 | 0.8750 | 1.0000 | [-0.0030, +0.0292] | 20 |
| 1D ResNet | 0.1068 | 0.1581 | +32.5 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0111, +0.0916] | 20 |
| Transformer | 0.1068 | 0.2535 | +57.9 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.1145, +0.1673] | 20 |
| TSMC-CNN [21] | 0.1068 | 0.2540 | +58.0 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0753, +0.1967] | 20 |
| MSCNN [7] | 0.1068 | 0.2085 | +48.8 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0785, +0.1179] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1068 | 0.0970 | -10.1 | 1/4 | 0.2500 | 0.8750 | 0.6250 | [-0.0185, -0.0017] | 20 |
| temporal stream only | 0.1068 | 0.0974 | -9.7 | 2/4 | 0.5000 | 0.8750 | 1.0000 | [-0.0284, +0.0096] | 20 |
| spectral stream only | 0.1068 | 0.2153 | +50.4 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0547, +0.1623] | 20 |
| without PGMC | 0.1068 | 0.1102 | +3.1 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0014, +0.0054] | 20 |
| without DANN | 0.1068 | 0.1093 | +2.3 | 3/4 | 0.3750 | 0.8750 | 0.6250 | [-0.0010, +0.0051] | 20 |
| without PGMC and DANN | 0.1068 | 0.1133 | +5.8 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0034, +0.0094] | 20 |
| PGMC batch ordering | 0.1068 | 0.1113 | +4.0 | 4/4 | 0.1250 | 0.8750 | 0.1250 | [+0.0013, +0.0100] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1095 | 0.1593 | +31.3 | 4/4 | 0.1250 | 0.1250 | 0.1250 | [+0.0220, +0.0654] | 5 |

## zs3 (3 bearings), CES MAE (smallest attainable flip p = 0.2500)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1827 | 0.1507 | -21.2 | 1/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0839, +0.0607] | 20 |
| BiLSTM | 0.1827 | 0.1473 | -24.0 | 1/3 | 0.7500 | 1.0000 | 1.0000 | [-0.1080, +0.0979] | 20 |
| CNN-LSTM | 0.1827 | 0.1515 | -20.5 | 1/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0780, +0.0569] | 20 |
| 1D ResNet | 0.1827 | 0.1792 | -1.9 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0944, +0.1021] | 20 |
| Transformer | 0.1827 | 0.3144 | +41.9 | 2/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0289, +0.2889] | 20 |
| TSMC-CNN [21] | 0.1827 | 0.1561 | -17.0 | 1/3 | 0.7500 | 1.0000 | 1.0000 | [-0.1079, +0.0927] | 20 |
| MSCNN [7] | 0.1827 | 0.2287 | +20.1 | 2/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0092, +0.1027] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1827 | 0.1861 | +1.8 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0046, +0.0158] | 20 |
| temporal stream only | 0.1827 | 0.1880 | +2.9 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0299, +0.0700] | 20 |
| spectral stream only | 0.1827 | 0.1848 | +1.1 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0374, +0.0745] | 20 |
| without PGMC | 0.1827 | 0.1746 | -4.6 | 0/3 | 0.2500 | 1.0000 | 0.2500 | [-0.0124, -0.0006] | 20 |
| without DANN | 0.1827 | 0.1682 | -8.6 | 1/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0288, +0.0020] | 20 |
| without PGMC and DANN | 0.1827 | 0.1637 | -11.6 | 1/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0333, +0.0015] | 20 |
| PGMC batch ordering | 0.1827 | 0.1745 | -4.7 | 0/3 | 0.2500 | 1.0000 | 0.2500 | [-0.0154, -0.0024] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1863 | 0.1084 | -71.9 | 1/3 | 0.5000 | 0.5000 | 1.0000 | [-0.1198, +0.0034] | 5 |

## zs3 (3 bearings), RAW MAE (smallest attainable flip p = 0.2500)

### baselines

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| 1D CNN | 0.1874 | 0.1561 | -20.1 | 1/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0805, +0.0583] | 20 |
| BiLSTM | 0.1874 | 0.1518 | -23.4 | 1/3 | 0.5000 | 1.0000 | 1.0000 | [-0.1004, +0.0881] | 20 |
| CNN-LSTM | 0.1874 | 0.1562 | -20.0 | 1/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0755, +0.0507] | 20 |
| 1D ResNet | 0.1874 | 0.1798 | -4.2 | 1/3 | 0.7500 | 1.0000 | 1.0000 | [-0.0954, +0.0930] | 20 |
| Transformer | 0.1874 | 0.3190 | +41.2 | 2/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0141, +0.2759] | 20 |
| TSMC-CNN [21] | 0.1874 | 0.1574 | -19.1 | 1/3 | 0.7500 | 1.0000 | 1.0000 | [-0.1095, +0.0843] | 20 |
| MSCNN [7] | 0.1874 | 0.2301 | +18.5 | 2/3 | 0.5000 | 1.0000 | 1.0000 | [-0.0084, +0.1008] | 20 |

### ablations

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| static fusion | 0.1874 | 0.1908 | +1.8 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0047, +0.0157] | 20 |
| temporal stream only | 0.1874 | 0.1916 | +2.2 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0298, +0.0702] | 20 |
| spectral stream only | 0.1874 | 0.1910 | +1.9 | 1/3 | 1.0000 | 1.0000 | 1.0000 | [-0.0410, +0.0799] | 20 |
| without PGMC | 0.1874 | 0.1794 | -4.5 | 0/3 | 0.2500 | 1.0000 | 0.2500 | [-0.0125, -0.0005] | 20 |
| without DANN | 0.1874 | 0.1718 | -9.1 | 0/3 | 0.2500 | 1.0000 | 0.2500 | [-0.0294, -0.0009] | 20 |
| without PGMC and DANN | 0.1874 | 0.1675 | -11.9 | 0/3 | 0.2500 | 1.0000 | 0.2500 | [-0.0340, -0.0008] | 20 |
| PGMC batch ordering | 0.1874 | 0.1793 | -4.5 | 0/3 | 0.2500 | 1.0000 | 0.2500 | [-0.0152, -0.0023] | 20 |

### protocol

| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | sign p | bootstrap 95% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| in-bearing early stopping | 0.1909 | 0.1119 | -70.6 | 0/3 | 0.2500 | 0.2500 | 0.2500 | [-0.1189, -0.0013] | 5 |
