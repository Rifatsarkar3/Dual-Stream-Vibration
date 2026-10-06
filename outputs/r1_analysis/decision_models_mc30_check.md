# Amendment 4 check: results that use no MC-dropout predictions

| Part | largest relative difference to Amendment 2 | status |
|---|---|---|
| calibration (c*, residual quantiles) | 0 | identical |
| P1 (cost-matched trigger, every set) | 0 | identical |
| P3 (framework vs standard practice) | 0 | identical |
| CES MAE ranking and Kendall tau | 0 | identical |

Overall: PASS.

Paths of the baselines' final predictions in A4:

- 1D CNN: `outputs/r1_baseline_mc30/cnn`
- BiLSTM: `outputs/r1_baseline_mc30/bilstm`
- CNN-LSTM: `outputs/r1_baseline_mc30/cnnlstm`
- 1D ResNet: `outputs/r1_baseline_mc30/resnet`
- Transformer: `outputs/r1_baseline_mc30/transformer`
- TSMC-CNN [21]: `outputs/r1_reeval/published_tsmccnn`
- MSCNN [7]: `outputs/r1_baseline_mc30/mscnn`
