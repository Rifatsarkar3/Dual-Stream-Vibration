# Dual-Stream-Vibration

Code, result files, test predictions and trained models for remaining-useful-life prediction of rolling-element
bearings with the dual-stream network DSAFLite+PGMC and seven baselines (PRONOSTIA, XJTU-SY; check on C-MAPSS).

## Contents

```
src/                     models, post-processing, preprocessing
  dsaf_v2_lite.py           DSAFLite network; mini_dann.py, monotone_rul_head.py: training-only terms
  online_filters.py         causal filters (CES, running minimum); isotonic_eval.py: PAVA (offline)
  vanilla_baselines.py, sota_baselines.py   1D CNN, CNN-BiLSTM, CNN-LSTM, 1D ResNet, Transformer
  published_baselines.py    TSMC-CNN (Jiang et al. 2020), reproduced
  mscnn_li2019.py           multi-scale CNN (Li et al. 2019), reproduced
  temporal_context.py       temporal-context variant
  components_r2.py          rebuilt components of the network: band envelope, noise-trained gate, pair penalty
  preprocess_pronostia.py, preprocess_xjtu.py   raw files -> labelled windows
Experiments/
  exp16_ress.py, exp17_xjtu.py, exp18_noise_gate.py, exp19_baselines.py, exp20_cwru_zeroshot.py
                            training and evaluation harnesses
  r1_*.py, r1_*.sh          selection, final runs, re-evaluation, statistics, calibration, decisions,
                            in-bearing comparison, power, random-split check (see below)
  r1_provenance.py          records code hashes, plan hash, git state and command in every result file
  r1_components*.py, r1_band_select.py   rebuilt components: selection, final runs, noise, statistics
  r1_cmapss*.py             C-MAPSS check of the calibration source, incl. calibration on time blocks
tests/                     preprocessing, MSCNN, temporal-context, rebuilt-component and C-MAPSS tests
paper/revision_R1/
  R1_analysis_plan.md       analysis plan with its dated amendments
  R1_experiment_registry.md every run, including smoke tests and failed attempts
  manuscript_R1/            table script (reads the saved result files only), release script
paper/revision_R2/manuscript_R2/make_figures_R2.py   figure script (reads the saved result files only; PNG at
                           1000 dpi and a vector PDF of every figure)
outputs/r1_analysis/       result tables (Markdown) and JSON, each with provenance
outputs/<run>/             test predictions (raw, target, MC-dropout mean and SD) of every reported model,
                           20 seeds; trained models of the final runs in outputs/r1_final_B10/final20
                           (PRONOSTIA) and outputs/exp17_xjtu/final20 (XJTU-SY)
outputs/r1_tsmc_randsplit/ TSMC-CNN under random splits of windows
outputs/r1_components*/    rebuilt network and its ablations: selection files, test predictions, noise results
outputs/r1_cmapss*/        C-MAPSS runs: calibration errors, budgets and test predictions, with run records
outputs/exp16_ress/final20/ models and predictions of the first version (see below)
```

## Result files of the first version

`outputs/exp16_ress/final20/` holds the models and test predictions of the first version (tag
`submission-2026-07-12`). Their PRONOSTIA test metrics were computed on processed files of Bearing1_4 and
Bearing1_5 that included 237 and 410 temperature records; all later results use corrected files, which
`src/preprocess_pronostia.py` builds from the vibration records only and `tests/test_preprocess_pronostia.py`
checks against each bearing's file inventory. The re-evaluated first-version models are in
`outputs/r1_analysis/tables.md` (section "orig4").

## Data

PRONOSTIA (IEEE PHM 2012), XJTU-SY and C-MAPSS are public third-party datasets and are not redistributed here.
For C-MAPSS, extract `CMAPSSData.zip` from the data repository of NASA's Prognostics Center of Excellence into
`data_cmapss/CMAPSSData/`; `Experiments/r1_cmapss.py` checks the SHA-256 of the FD001 and FD003 files. For the
bearing datasets, after downloading them:

```bash
pip install -r requirements.txt
python src/preprocess_pronostia.py --raw-root <PHM 2012 folder> --out-dir Processed_PRONOSTIA_v2
python src/preprocess_pronostia.py --raw-root <PHM 2012 folder> --out-dir Processed_PRONOSTIA_v2hv --with-vertical
python src/preprocess_xjtu.py --raw-dir <XJTU-SY folder> --out-dir Processed_XJTU
python -m pytest tests/
```

## Running the analyses

Each row of `paper/revision_R1/R1_experiment_registry.md` names the script and settings of one run, in the order
in which the runs were made. The main steps:

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
python Experiments/r1_mechanism.py           # calibration-source ratio against training progress
python Experiments/r1_tsmc_randsplit.py --design a && python Experiments/r1_tsmc_randsplit.py --design b \
    && python Experiments/r1_tsmc_randsplit.py --summary
# rebuilt components (stages in the order of the plan)
bash Experiments/r1_components_runs.sh band     # then smoke, budget, lambda, valpreds, final, xjtu-*
python Experiments/r1_components_noise.py --dataset pronostia && python Experiments/r1_components_stats.py
# C-MAPSS check
python tests/test_cmapss.py && python Experiments/r1_cmapss.py --smoke
python Experiments/r1_cmapss.py --dataset FD001 && python Experiments/r1_cmapss.py --dataset FD003
python Experiments/r1_cmapss_stats.py --dry-run && python Experiments/r1_cmapss_stats.py
# C-MAPSS, calibration on time blocks of the training engines (needs the outputs of the previous step)
python tests/test_cmapss_blocked.py && python Experiments/r1_cmapss_blocked.py --smoke
python Experiments/r1_cmapss_blocked.py --dataset FD001 && python Experiments/r1_cmapss_blocked.py --dataset FD003
python Experiments/r1_cmapss_blocked_stats.py --dry-run && python Experiments/r1_cmapss_blocked_stats.py
# figures and tables from the saved result files
python paper/revision_R2/manuscript_R2/make_figures_R2.py
python paper/revision_R1/manuscript_R1/make_supplement_tables.py
```

Everything above runs on one consumer GPU; most multi-seed runs take minutes.

## Not included

- The datasets (see Data).
- Trained baseline models, trained models of the rebuilt network and intermediate files of the selection runs
  (several hundred MB); they are available from the authors on request and are regenerated by the scripts above.
  The C-MAPSS networks are not stored; each run is deterministic and is regenerated in minutes.

## License

See [LICENSE](LICENSE).
