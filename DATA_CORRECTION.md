# Data correction: why the PRONOSTIA test numbers changed after the submission

This note explains how the PRONOSTIA test results of the originally submitted manuscript (July 2026) became
those of revision 1 (October 2026). Two things changed them: a data-reading error that we found and corrected
during the revision, and the wider evaluation that the revision uses. Each number below names its source: a file
in this repository or, for values quoted from the submission, the submitted manuscript.

## 1. The error

In the PHM 2012 data, the test folders `Full_Test_Set/Bearing1_4` and `Full_Test_Set/Bearing1_5` contain
vibration snapshots (`acc_*.csv`: 1,428 and 2,463 files) and temperature records (`temp_*.csv`: 237 and 410
files). The original preprocessing read every `*.csv` file of a folder. The processed test sequences of these two
bearings therefore ended with 237 and 410 temperature records that were treated as vibration windows, and their
remaining-useful-life labels were normalized by the inflated file counts (1,665 and 2,873) instead of the number
of vibration snapshots (1,428 and 2,463).

- The four test bearings of the submission (Bearing1_3, Bearing1_4, Bearing1_5, Bearing2_3) were evaluated on
  8,868 windows instead of 8,221 (647 temperature records too many).
- The training bearings (`Learning_set`) contain no temperature files, so the training data, every selection
  decision and every trained model are unaffected. The XJTU-SY data are unaffected.
- Several other test folders (Bearing1_6, Bearing1_7, Bearing2_4 to Bearing2_7, Bearing3_3) also contain
  temperature files. They were not part of the submitted evaluation; the corrected preprocessing reads vibration
  files only, for every bearing, so the extended test set of the revision is clean.

## 2. How it was found and corrected

- Found while re-running the complete pipeline for the revision, in an audit of the data preparation.
- `src/preprocess_pronostia.py` now reads `acc_*.csv` only (output folder `Processed_PRONOSTIA_v2`).
- `tests/test_preprocess_pronostia.py` checks that every bearing's window count equals its number of vibration
  files. Applied to the old processed files, the check flags exactly the 647 temperature records.
- All saved models were re-evaluated on the corrected files (`Experiments/r1_reeval.py`). On the unaffected
  windows, the new predictions match the original ones to within 1.2 × 10⁻⁷. Only the test evaluation changed.

## 3. Effect of the correction alone

Same trained models, same four test bearings, window-pooled metrics, 20 seeds:

| Quantity | Submitted | After the correction |
|---|---|---|
| Test windows | 8,868 | 8,221 |
| Raw MAE (causal) | 0.0897 | 0.1029 |
| CES MAE (causal) | 0.0841 | 0.0977 |
| PAVA MAE / R² (offline) | 0.0789 / 0.865 | 0.0931 / 0.797 |
| Static fusion instead of the adaptive gate (PAVA MAE) | 8.5 % higher error | 0.2 % lower error |
| Temporal stream only instead of both streams (PAVA MAE) | 6.9 % higher error | 5.7 % lower error |
| Five baselines (PAVA MAE, relative to each baseline) | 16.6 % to 67.0 % higher error | 15.2 % to 59.2 % higher error |
| Within-bearing early stopping (PAVA MAE) | 41 % higher error | 52.6 % higher error |
| Two-sided interval coverage at c\* = 2.41 (90 % target) | 89.1 % | 84.4 % |

Sources. Submitted: the submitted manuscript, and `outputs/exp16_ress/final20/results.json` (aggregated raw, CES
and PAVA MAE and PAVA R²; 5 training epochs). After the correction: `outputs/r1_analysis/tables.md`, section
"orig4 (4 bearings)", rows `final20` (the submitted model), `abl20_static`, `abl20_temporal`, `baseline_*` and
`full_inbearing_es`; its coverage column (PICP) uses the submission's factor c\* = 2.414
(`Experiments/r1_analysis.py`). Ablation and baseline differences are relative to the variant,
(variant − full) / variant, as in the submission; early stopping is relative to the protocol model,
(0.1421 − 0.0931) / 0.0931.

The correction raised every error. On the offline PAVA error that the submission reported, the gain of the
adaptive gate vanished and the gain of the second stream reversed; on the raw and causal errors, both reversed
(same file, paired tests over seeds). The revision therefore withdraws these component claims.

## 4. From the corrected numbers to those of revision 1

The revision also changed how the model is selected and evaluated. The step-by-step path for our network on
PRONOSTIA (window-pooled metrics, 20 seeds, unless stated):

| Step | What changes | Test bearings | Raw MAE | CES MAE | PAVA MAE / R² | Source |
|---|---|---|---|---|---|---|
| 0 | submitted | 4 | 0.0897 | 0.0841 | 0.0789 / 0.865 | `outputs/exp16_ress/final20/results.json` |
| 1 | data correction | 4 | 0.1029 | 0.0977 | 0.0931 / 0.797 | `outputs/r1_analysis/tables.md` (orig4, `final20`) |
| 2 | training budget chosen by four-fold leave-one-bearing-out validation on the training bearings (10 epochs instead of 5), retrained | 4 | 0.1089 | 0.1052 | 0.1010 / 0.761 | `outputs/r1_analysis/tables_B10.md` (orig4, `B10:final20`) |
| 3 | all ten Condition 1 and 2 test bearings | 10 | 0.1262 | 0.1221 | 0.1146 / 0.710 | `tables_B10.md` (all10, `B10:final20`); manuscript Table 4 |
| 4 | primary endpoint: CES MAE averaged over bearings | 10 | | 0.1516 | | `outputs/r1_analysis/strict_B10.md`; manuscript Table 4 |

The revised model is selected without any test bearing, so on the four submitted bearings it is slightly less
accurate than the submitted one (step 2). The six added bearings have larger errors (step 3). The primary endpoint
averages over bearings, so each bearing counts once whatever its length (step 4).

Interval coverage changed for two further reasons:

- **Label.** The coverage that the submission reported as one-sided lower-bound coverage had been computed as
  two-sided interval coverage. The revision reports two-sided coverage and defines it.
- **Calibration data.** The submitted factor c\* = 2.41 was fitted on checkpoints that the calibration bearing
  itself had selected, while the deployed model keeps its final epoch. The factor is now fitted on the
  final-epoch predictions of the selection runs on their held-out bearings: c\* = 3.95. On the ten corrected test
  bearings, coverage is 73.1 % with the submitted factor and 88.0 % with the new one
  (`outputs/r1_analysis/calibration_v2.json`: `global/c_star`; `test/all10/superseded_two_sided`, pooled;
  `test/all10/two_sided_c_star`, pooled).

Statistics now treat the bearing, not the seed, as the unit: exact sign-flip tests over bearings with Holm
adjustment (`outputs/r1_analysis/strict_B10.md`). On the same predictions and after Holm adjustment, tests over
seeds declare 29 of 44 comparisons significant, tests over bearings 7 (`outputs/r1_analysis/eval_choices.md`).

## 5. The submitted version stays available

- Tag `submission-2026-07-12`: the repository as it was when the manuscript was submitted, with the first README
  and the first-version results.
- `outputs/exp16_ress/final20/`: models, test predictions and results of the first version; their test metrics
  use the uncorrected files.
- Tag `revision-1-2026-10-14`: this version.

## 6. Re-running the correction

With the PHM 2012 data downloaded (see the README, section Data):

```bash
python src/preprocess_pronostia.py --raw-root <PHM 2012 folder> --out-dir Processed_PRONOSTIA_v2
python -m pytest tests/test_preprocess_pronostia.py       # window count = vibration files, every bearing
python Experiments/r1_reeval.py --which final20           # the submitted models on the corrected files
python Experiments/r1_analysis.py                         # results.json and the tables of tables.md
```

The trained models of the submission (`outputs/exp16_ress/final20/`) are in this repository; the baseline and
ablation models are available from the authors on request, so with the files here the last two commands
re-evaluate the submitted network.
