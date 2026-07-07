# Leakage-Free, Monotonic, and Uncertainty-Calibrated Bearing RUL Prediction

Code, trained model checkpoints, and result artifacts accompanying the manuscript *"Leakage-Free,
Monotonic, and Uncertainty-Calibrated Bearing RUL Prediction for Reliability-Centered Maintenance,"*
by Sarkar Mohammad Raziul Hasan Rifat, Tao Zhang, Akter Labani, Boudjelkha Mohammed Djamel Eddine,
and Saleh Mahamat Aboubakar Ousmane (Faculty of Mechanical and Material Engineering, Huai'an
University), submitted to *Reliability Engineering & System Safety* (Elsevier). See
[`paper/manuscript.pdf`](paper/manuscript.pdf) for the full manuscript and
[`highlights.docx`](highlights.docx) for the paper's highlights.

## What this is

The paper's central contribution is a **reliability-oriented evaluation protocol** for bearing
remaining-useful-life (RUL) prediction: model selection that never touches held-out test bearings,
degradation trajectories that are evaluated for (and corrected toward) monotonicity, predictive
uncertainty that is calibrated without test-set information, and a maintenance-decision analysis
that goes beyond point-error metrics to alarm timing and asymmetric decision loss. See
[`paper/figures/Fig1_workflow.png`](paper/figures/Fig1_workflow.png) for the workflow diagram.

The protocol is instantiated with **DSAFLite+PGMC**, a compact (0.19M-parameter) dual-stream network
used as the vehicle for demonstrating the protocol, not as an independent architectural claim:

- **Temporal stream**: 1D CNN over raw vibration windows.
- **Spectral stream**: fixed (non-learnable) Hilbert-envelope-spectrum extractor + 1D CNN.
- **Adaptive scalar fusion gate**: learns the per-window contribution of each stream.
- **PGMC**: a soft monotonic training regularizer (not a hard constraint) — an auxiliary prior, not
  a standalone contribution.
- **DANN**: a lightweight source-condition adversarial regularizer (gradient-reversal), discarded at
  inference — also an auxiliary prior.
- **Post-processing**: offline PAVA (isotonic regression, retrospective-only) and causal online
  filters (RTRM, CES), which supply the actual monotonicity guarantee — PGMC alone does not.

Training-budget selection uses a two-fold bearing-level sweep over **training** bearings only, then
is frozen and evaluated once on held-out bearings — this is shown to matter a great deal on small
run-to-failure benchmarks (PRONOSTIA, XJTU-SY): checkpoints selected via in-bearing early stopping
degrade held-out-bearing MAE by 41% relative to the leakage-free protocol.

See the manuscript for full results; headline numbers on PRONOSTIA (20 seeds): raw MAE 0.0897 ± 0.0050
(R² 0.8169 ± 0.0176), PAVA-smoothed MAE 0.0789 ± 0.0068 (R² 0.8646 ± 0.0224), outperforming five
baselines trained under the identical protocol. Leak-free calibration raises one-sided lower-bound
coverage (LBC) from 59.9% to 89.1% at nominal 90%. Reframing the interval-triggered maintenance rule
around alarm timing shows that the best-point-accuracy variant (PAVA) is not the best-decision variant
under a 5:1 late-vs-early cost asymmetry — see `Experiments/exp29_maintenance_timing.py`.

## Repository structure

```
src/                  Core model, losses, post-processing, and utility code
  dsaf_v2_lite.py        DSAFLite architecture (the model actually used in the paper)
  mini_dann.py           Gradient-reversal DANN discriminator
  monotone_rul_head.py   PGMC soft monotonic regularizer
  online_filters.py      Causal post-processing (RTRM, CES)
  isotonic_eval.py        Offline post-processing (PAVA) + monotonicity metrics
  vanilla_baselines.py   Vanilla 1D CNN and CNN-BiLSTM baselines
  sota_baselines.py      CNN-LSTM, Transformer, and Deep 1D ResNet baselines
  data_loader.py          Dataloader construction
  preprocess_xjtu.py     XJTU-SY raw-file -> RUL-labeled-window preprocessing
  utils/                  Metrics, hardware profiling, saving, plotting helpers

Experiments/           Entry-point scripts that produce every number/table in the paper
  select_epoch_budget.py    Leave-one-bearing-out budget sweep (Section 4.3)
  exp16_ress.py              PRONOSTIA main results, ablations, PGMC/DANN training loop
  exp17_xjtu.py              XJTU-SY in-domain and zero-shot results
  exp18_noise_gate.py        Additive-noise robustness and gate-response study
  exp19_baselines.py         Baseline architectures under the identical protocol
  exp20_cwru_zeroshot.py     CWRU negative-transfer control (Section 6.4)
  exp21_efficiency.py        Computational profiling (Table 7)
  exp22_uq_calibration.py    MC-dropout interval calibration, LBC (Section 5.7)
  exp23_risk_decision.py     Interval-triggered maintenance decision rule (Table 8/9)
  exp29_maintenance_timing.py  Alarm-timing and asymmetric decision-loss analysis (Table 10, Fig. 10)
  exp_task1_online_filters.py  RTRM / CES post-processing comparison
  compute_stats.py            Paired t-tests and Cohen's d for every reported comparison

files/                 Figure-generation scripts (paper/figures/*.png)
  generate_fig1.py           Fig. 2, DSAFLite+PGMC architecture diagram
  generate_fig_workflow.py   Fig. 1, reliability-oriented workflow diagram

paper/                 Compiled manuscript PDF and final figures (paper/figures/, numbered to match
                        the manuscript's actual figure order)

outputs/exp16_ress/final20/  Trained checkpoints (model_seed*.pth), raw per-seed predictions
                        (preds_seed*.npz), and aggregate results.json for the 20-seed headline
                        PRONOSTIA run reported in Table 1 and throughout the paper
outputs/stats_tests_n20.json  Paired significance tests (p-values, Cohen's d) backing every
                        reported comparison in the ablation and baseline tables
```

## What's included vs. deliberately excluded

**Included**: the trained model checkpoints, per-seed predictions, and aggregate results for the
headline 20-seed PRONOSTIA run (`outputs/exp16_ress/final20/`), so the paper's central numbers can
be reproduced or inspected without retraining.

**Deliberately not included**:

- **Raw and preprocessed datasets** (PRONOSTIA / FEMTO, XJTU-SY, CWRU) — these are public
  third-party datasets and are not redistributed here; download them from their original
  sources (see the manuscript's dataset citations) and point `Experiments/*.py` at the local
  paths they expect.
- **Checkpoints and per-seed result dumps for other experiments** (ablations, baselines, noise
  sweep, XJTU-SY, CWRU, budget sweep) — not included due to size; every script above regenerates
  them (each full multi-seed run completes in minutes on a single consumer GPU per the paper's
  Section 4.4/4.5).
- **An unrelated exploratory thread** (`DVA` / `RITA` alignment modules and experiments,
  and their unit tests) that is not referenced anywhere in the manuscript — omitted
  to keep this repository scoped to what the paper actually reports. Ask if you want it added back.
- A separate `train.py` / `src/dsaf_core.py` pair present in the working repo was excluded:
  it implements an earlier, pre-DSAFLite architecture (with a torchvision/ImageNet backbone)
  that isn't used anywhere in the confirmed training pipeline above.

## Reproducing the results

1. `pip install -r requirements.txt`
2. Download PRONOSTIA and XJTU-SY (and, optionally, CWRU for the negative control) from their
   original sources and preprocess with `src/preprocess_xjtu.py` (XJTU-SY) as needed.
3. Run `Experiments/select_epoch_budget.py` to reproduce the leakage-free budget sweep, then
   the individual `exp*.py` scripts, each of which is a self-contained entry point for the
   corresponding paper section/table. `exp16_ress.py` reproduces the checkpoints and predictions
   already provided in `outputs/exp16_ress/final20/`.
4. `Experiments/compute_stats.py` reproduces every paired significance test reported.

## Citation

If you use this code, please cite the paper (citation details to be added on acceptance/publication).

## License

See [LICENSE](LICENSE).
