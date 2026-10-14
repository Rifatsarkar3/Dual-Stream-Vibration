# Leakage-Free and Uncertainty-Calibrated Remaining Useful Life Prediction of Bearings for Reliability-Centered Maintenance

Code, analysis plan, experiment registry, result files, test predictions and trained models accompanying the
manuscript JRESS-D-26-05118, *Reliability Engineering & System Safety* (revision 1, 2026), by Sarkar Mohammad
Raziul Hasan Rifat, Tao Zhang, Labani Akter, Boudjelkha Mohammed Djamel Eddine and Saleh Mahamat Aboubakar
Ousmane (Huai'an University).

## Correction notice

The first version of this repository (July 2026) accompanied the originally submitted manuscript. Its PRONOSTIA
test results were computed on processed files of test bearings Bearing1_4 and Bearing1_5 that contained
temperature records read as vibration windows (237 and 410 records). The error affected only the test evaluation:
no selection decision or trained model changed. For the revision every PRONOSTIA test result was recomputed on
corrected files and on an extended test set, and the headline numbers of the first README (for example a
smoothed MAE of 0.0789 and a coverage of 89.1 %) are superseded. The artifacts of the first version remain in
`outputs/exp16_ress/final20/` for reference; the test metrics stored there use the uncorrected files. The
preprocessing now reads vibration records only and checks every bearing's window count against its file
inventory (`src/preprocess_pronostia.py`, `tests/test_preprocess_pronostia.py`).

## What the revised paper shows

A maintenance alarm built on a remaining useful life (RUL) prediction depends on the tail of the prediction error
on bearings the model has never seen: the alarm rule that minimizes the expected cost of late and early
maintenance compares the r/(1 + r) quantile of that error distribution with the maintenance threshold, so an
understated tail can only delay maintenance. The paper builds a reliability-oriented framework around this
decision quantile (model selection that never uses test bearings, trajectory corrections declared offline or
causal, intervals calibrated before decision use, claims kept within validated conditions) and evaluates eight
networks under identical rules on PRONOSTIA and four on XJTU-SY:

- With few training units, the source of the calibration errors mattered more than the network. Validation
  windows taken from the training bearings understated the 0.9-quantile of the test error 1.8 to 9 times in all
  twelve network-dataset combinations. Calibrated on them, 90 % intervals covered 21 to 50 % of PRONOSTIA test
  outcomes, against 64 to 84 % when calibrated on held-out bearings; a replication on XJTU-SY specified in advance
  confirmed the difference for the four networks pooled (11 of 11 bearings, p = 0.001), though not for our
  network alone (p = 0.19).
- The understatement grew with the epoch that in-bearing early stopping selected (9 of 12 combinations, exact
  p = 0.0025): the error on new bearings widened with further training, and the validation windows did not
  register it (Amendment 9).
- Tests over random seeds declared 29 of 44 comparisons significant, tests over bearings 7.
- The most accurate offline trajectory correction (PAVA) alarmed late on 81.5 to 95.5 % of trajectories, and the
  decision-optimal rule changes with the cost ratio at break-even values given in closed form. With ten or eleven
  bearings, only maintenance-cost differences of about 30 % or more are detectable, so the paper makes no cost
  claim.
- The compact dual-stream network DSAFLite+PGMC has the lowest mean error of the eight networks on PRONOSTIA
  (bearing-averaged CES MAE 0.1516 over ten test bearings), significantly lower than four of the seven baselines
  after Holm adjustment and not distinguishable from the other three; of its components only the raw-vibration
  stream has a measurable effect. Its calibrated intervals cover 88.0 % (PRONOSTIA) and 86.6 % (XJTU-SY) of test
  outcomes at a 90 % target.
- Rebuilt versions of three components (band envelope stream, gate trained with input noise, pair penalty),
  specified in Amendment 10 before any of their runs, had no measurable effect either; the rebuilt network did
  not meet the plan's threshold for replacing the submitted one.
- On the simulated C-MAPSS turbofan engines (Amendment 11: FD001, with FD003 as replication; training fleets of
  4, 10, 25 and 80 engines, 20 random fleets each), intervals calibrated on windows of the training engines
  covered 20.8 and 31.0 % of test outcomes with four engines, against 87.6 and 82.5 % with held-out engines.
  The gap shrank as the fleet grew and was not detected with 80 engines: the understatement is a property of
  evaluating with few training units, the situation of both bearing benchmarks.
- Calibration on contiguous time blocks of the training engines, separated from the training windows by buffers
  (Amendment 12), did not understate the tail: against the expectation stated in the plan, it overstated it, with
  intervals 1.2 to 1.9 times as wide as with held-out engines. The understatement disappears when the validation
  windows are separated in time from the training windows; a random split of windows interleaves them.

## Repository structure

```
src/                     models, post-processing, preprocessing
  dsaf_v2_lite.py           DSAFLite network; mini_dann.py, monotone_rul_head.py: training-only terms
  online_filters.py         causal filters (CES, running minimum); isotonic_eval.py: PAVA (offline)
  vanilla_baselines.py, sota_baselines.py   1D CNN, CNN-BiLSTM, CNN-LSTM, 1D ResNet, Transformer
  published_baselines.py    TSMC-CNN (Jiang et al. 2020), reproduced
  mscnn_li2019.py           multi-scale CNN (Li et al. 2019), reproduced
  temporal_context.py       temporal-context variant (secondary analysis)
  components_r2.py          rebuilt components of the network (Amendment 10): band envelope, noise-trained gate,
                            pair penalty
  preprocess_pronostia.py, preprocess_xjtu.py   raw files -> labelled windows
Experiments/
  exp16_ress.py, exp17_xjtu.py, exp18_noise_gate.py, exp19_baselines.py, exp20_cwru_zeroshot.py
                            training and evaluation harnesses (original study, extended for the revision)
  r1_*.py, r1_*.sh          the revision: selection, final runs, re-evaluation, statistics, calibration,
                            decisions, in-bearing comparison, power, random-split check (see below)
  r1_provenance.py          records code hashes, plan hash, git state and command in every result file
  r1_components*.py, r1_band_select.py   rebuilt components: selection, final runs, noise, statistics (Amendment 10)
  r1_cmapss.py, r1_cmapss_stats.py       C-MAPSS turbofan check of the calibration source (Amendment 11)
  r1_cmapss_blocked.py, r1_cmapss_blocked_stats.py   C-MAPSS: calibration on time blocks of the training
                            engines against held-out engines (Amendment 12)
tests/                     preprocessing, MSCNN, temporal-context, rebuilt-component and C-MAPSS tests (both schemes)
paper/revision_R1/
  R1_analysis_plan.md       dated analysis plan with twelve amendments (Amendment 11 with two corrections and
                            Amendment 12 with one, each made before its analysis) and one declared deviation
  R1_experiment_registry.md every run, including smoke tests and failed attempts
  manuscript_R1/make_supplement_tables.py   supplementary tables (numbered as in an earlier draft; same values)
paper/revision_R2/manuscript_R2/make_figures_R2.py   figures and graphical abstract of the revised manuscript,
                            from the saved result files only
outputs/r1_analysis/       result tables (Markdown) and JSON, each with provenance
outputs/<run>/             test predictions (raw, target, MC-dropout mean and SD) of every reported model,
                           20 seeds; trained models of our final runs in outputs/r1_final_B10/final20
                           (PRONOSTIA) and outputs/exp17_xjtu/final20 (XJTU-SY)
outputs/r1_tsmc_randsplit/ TSMC-CNN under random splits of windows (Amendment 8)
outputs/r1_components*/    rebuilt network and its ablations: selection files, test predictions, noise results
                           (Amendment 10)
outputs/r1_cmapss/         C-MAPSS runs: per fleet and repeat, calibration errors and test predictions of both
                           schemes, with run records (Amendment 11)
outputs/r1_cmapss_blocked/ C-MAPSS runs of the blocked scheme: block errors, budgets and test predictions, with
                           run records (Amendment 12)
outputs/exp16_ress/final20/ artifacts of the first version (see the correction notice)
```

The analysis plan and the registry call the plan "pre-registered". Because the plan was written during the
revision, after results of the original study on part of the data were known, the paper describes it as a dated
analysis plan with amendments (Supplementary Material S1).

## Data

PRONOSTIA (IEEE PHM 2012), XJTU-SY and C-MAPSS are public third-party datasets and are not redistributed here.
For C-MAPSS, extract `CMAPSSData.zip` from the data repository of NASA's Prognostics Center of Excellence into
`data_cmapss/CMAPSSData/`; `Experiments/r1_cmapss.py` checks the SHA-256 of the FD001 and FD003 files recorded in
Amendment 11. For the bearing datasets, after downloading them:

```bash
pip install -r requirements.txt
python src/preprocess_pronostia.py --raw-root <PHM 2012 folder> --out-dir Processed_PRONOSTIA_v2
python src/preprocess_pronostia.py --raw-root <PHM 2012 folder> --out-dir Processed_PRONOSTIA_v2hv --with-vertical
python src/preprocess_xjtu.py --raw-dir <XJTU-SY folder> --out-dir Processed_XJTU
python -m pytest tests/
```

## Reproducing the revision

Each row of `paper/revision_R1/R1_experiment_registry.md` names the script and settings of one run, in the order
in which the runs were made; the plan states what each analysis tests. The main steps:

```bash
# selection on the four training bearings only (deterministic kernels)
bash Experiments/r1_lambda_lobo4.sh && python Experiments/r1_select_lambda.py
python Experiments/r1_baseline_select5.py
python Experiments/r1_mscnn.py --stage smoke && python Experiments/r1_mscnn.py --stage select
# final runs (20 seeds) and evaluation
bash Experiments/r1_final_B10.sh
python Experiments/r1_baseline_final.py --model bilstm     # also cnnlstm, transformer
python Experiments/r1_mscnn.py --stage final
python Experiments/r1_strict_stats.py && python Experiments/r1_tables.py
# calibration and decisions
python Experiments/r1_calibration.py
python Experiments/r1_decision.py --preds-dir outputs/r1_final_B10/final20 --set all10 \
    --calibration-json outputs/r1_analysis/calibration_v2.json
python Experiments/r1_decision_models_mc30.py
# XJTU-SY
python Experiments/r1_xjtu_baselines.py --stage select --model all
python Experiments/r1_xjtu_baselines.py --stage final --model all
bash Experiments/r1_xjtu_calib_runs.sh full && python Experiments/r1_xjtu_calibration.py
# evaluation choices, calibration source, power, random-split check
python Experiments/r1_eval_choices.py
python Experiments/r1_decision_inbearing.py && python Experiments/r1_decision_xjtu_inbearing.py
python Experiments/r1_decision_bestckpt.py && python Experiments/r1_power.py
python Experiments/r1_mechanism.py           # Amendment 9: understatement against training progress
python Experiments/r1_tsmc_randsplit.py --design a && python Experiments/r1_tsmc_randsplit.py --design b \
    && python Experiments/r1_tsmc_randsplit.py --summary
# Amendment 10: rebuilt components (stages in the order of the plan)
bash Experiments/r1_components_runs.sh band     # then smoke, budget, lambda, valpreds, final, xjtu-*
python Experiments/r1_components_noise.py --dataset pronostia && python Experiments/r1_components_stats.py
# Amendment 11: C-MAPSS turbofan check
python tests/test_cmapss.py && python Experiments/r1_cmapss.py --smoke
python Experiments/r1_cmapss.py --dataset FD001 && python Experiments/r1_cmapss.py --dataset FD003
python Experiments/r1_cmapss_stats.py --dry-run && python Experiments/r1_cmapss_stats.py
# Amendment 12: blocked in-unit calibration on C-MAPSS (needs the Amendment 11 outputs)
python tests/test_cmapss_blocked.py && python Experiments/r1_cmapss_blocked.py --smoke
python Experiments/r1_cmapss_blocked.py --dataset FD001 && python Experiments/r1_cmapss_blocked.py --dataset FD003
python Experiments/r1_cmapss_blocked_stats.py --dry-run && python Experiments/r1_cmapss_blocked_stats.py
# figures of the revised manuscript and supplementary tables
python paper/revision_R2/manuscript_R2/make_figures_R2.py
python paper/revision_R1/manuscript_R1/make_supplement_tables.py
```

Everything above runs on one consumer GPU; most multi-seed runs take minutes.

## Not included

- The datasets (see Data).
- Trained baseline models, trained models of the rebuilt network (Amendment 10) and intermediate files of the
  selection runs (several hundred MB); they are available from the authors on request and are regenerated by the
  scripts above. The C-MAPSS networks are not stored; each run is deterministic and is regenerated in minutes.
- An unrelated exploratory thread of the working repository that the paper does not use.

## Citation

If you use this code, please cite the paper (citation details will be added on publication).

## License

See [LICENSE](LICENSE).
