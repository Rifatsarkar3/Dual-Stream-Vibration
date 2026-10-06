# Rebuilt components (revision R1, analysis plan Amendment 10)

Reproduction check: submitted network recomputed, max |difference| to the stored values 0.0e+00 (c* 3.9473, CES MAE 0.1516, cost 592.8 / 722.0).

Selected setting of R: B = 5, lambda = 0.08.

### Primary family, PRONOSTIA (ten test bearings)

| Test | R | Comparator | R lower by | R better on | p | Holm p | Significant, R better |
|---|---|---|---|---|---|---|---|
| A10-E E, envelope stream (R against raw-vibration stream only) | 0.1599 | 0.1616 | +1.1 % | 6/10 | 0.3887 | 0.7773 | no |
| A10-G G, gate under noise (R against static fusion, five noise levels) | 0.1917 | 0.1912 | -0.2 % | 4/10 | 0.6328 | 0.7773 | no |
| A10-M M, pair PGMC (R against no PGMC) | 0.1599 | 0.1584 | -0.9 % | 2/10 | 0.0410 | 0.1230 | no |

### Replication family, XJTU-SY (six test and five Condition-3 bearings)

| Test | R | Comparator | R lower by | R better on | p | Holm p | Significant, R better |
|---|---|---|---|---|---|---|---|
| A10-E E, envelope stream (R against raw-vibration stream only) | 0.2487 | 0.2580 | +3.6 % | 7/11 | 0.5293 | 1.0000 | no |
| A10-G G, gate under noise (R against static fusion, five noise levels) | 0.2415 | 0.2408 | -0.3 % | 7/11 | 0.8223 | 1.0000 | no |
| A10-M M, pair PGMC (R against no PGMC) | 0.2487 | 0.2457 | -1.2 % | 6/11 | 0.3477 | 1.0000 | no |

### Claims

| Component | Measurable effect | Replicated | May be listed as a contribution |
|---|---|---|---|
| A10-E | no | no | no |
| A10-G | no | no | no |
| A10-M | no | no | no |

### Secondary (i): each change alone

| Test | R | Comparator | R lower by | R better on | p | Holm p | Significant, R better |
|---|---|---|---|---|---|---|---|
| old_env submitted envelope extractor | 0.1599 | 0.1674 | +4.5 % | 6/10 | 0.1309 | 0.5234 | no |
| no_aug no input noise | 0.1599 | 0.1514 | -5.6 % | 3/10 | 0.4180 | 1.0000 | no |
| adjacent adjacent-window PGMC | 0.1599 | 0.1596 | -0.2 % | 3/10 | 0.7949 | 1.0000 | no |
| no_dann without DANN | 0.1599 | 0.1622 | +1.4 % | 6/10 | 0.3516 | 1.0000 | no |
| spectral envelope stream only | 0.1599 | 0.2554 | +37.4 % | 8/10 | 0.0117 | 0.0586 | no |

### Secondary (ii): R against the submitted network and the 1D CNN

| Test | R | Comparator | R lower by | R better on | p | Holm p | Significant, R better |
|---|---|---|---|---|---|---|---|
| submitted network  | 0.1599 | 0.1516 | -5.4 % | 2/10 | 0.3809 | 0.7617 | no |
| 1D CNN  | 0.1599 | 0.1616 | +1.0 % | 3/10 | 0.8926 | 0.8926 | no |

### Descriptive: calibration and decisions (own validation runs; unadjusted p against R)

| Model | c* | Coverage | Late life | Interval score | Tail ratio | Cost r = 5 | Cost r = 10 | CES MAE |
|---|---|---|---|---|---|---|---|---|
| full | 4.35 | 93.2 % | 73.4 % | 1.061 | 1.12 | 689.0 | 875.2 | 0.1599 |
| rawonly | 4.47 | 95.0 % | 81.5 % | 1.055 (p 0.725) | 1.21 | 750.1 (p 0.453) | 923.0 (p 0.258) | 0.1616 |
| static | 4.26 | 92.2 % | 68.8 % | 1.068 (p 0.381) | 1.09 | 659.1 (p 0.430) | 872.5 (p 0.805) | 0.1607 |
| no_pgmc | 4.42 | 94.2 % | 77.3 % | 1.057 (p 0.781) | 1.16 | 639.4 (p 0.047) | 805.9 (p 0.031) | 0.1584 |

### Mechanism

Gate weight of R by noise level: clean 0.664, 20 0.662, 10 0.652, 5 0.642, 0 0.637, -5 0.643.
Share of increasing raw-prediction pairs 5 to 10 % of the life apart: R 0.279, without PGMC 0.280 (p 0.6992).
Envelope stream alone, CES MAE: rebuilt 0.2554, submitted 0.2318 (5/10 bearings, p 0.4180).

