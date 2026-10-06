# Shape of the raw predictions with and without the monotonicity regularization (all10, 10 bearings; exploratory)

Per bearing: seed mean (20 seeds) of the share of non-increasing consecutive raw predictions and of the mean upward step. Δ: variant minus full model, averaged over bearings. p: exact two-sided sign-flip test over bearings, unadjusted.

| Model | Variant | non-increasing steps (%) | Δ (pp) | full model higher on | p | mean upward step | Δ (%) | full model smaller on | p |
|---|---|---|---|---|---|---|---|---|---|
| DSAFLite+PGMC with temporal context (adopted) | full model | 51.65 | – | – | – | 0.00641 | – | – | – |
| DSAFLite+PGMC with temporal context (adopted) | without PGMC | 51.54 | -0.11 | 6/10 | 0.283 | 0.00651 | +1.5 | 9/10 | 0.004 |
| DSAFLite+PGMC with temporal context (adopted) | without PGMC and DANN | 51.66 | +0.01 | 5/10 | 0.842 | 0.00567 | -13.1 | 0/10 | 0.002 |
| DSAFLite+PGMC with temporal context (adopted) | PGMC batch ordering | 51.59 | -0.06 | 9/10 | 0.082 | 0.00631 | -1.6 | 3/10 | 0.025 |
| DSAFLite+PGMC, submitted architecture | full model | 50.89 | – | – | – | 0.02232 | – | – | – |
| DSAFLite+PGMC, submitted architecture | without PGMC | 50.90 | +0.01 | 4/10 | 0.898 | 0.02282 | +2.2 | 9/10 | 0.006 |
| DSAFLite+PGMC, submitted architecture | without PGMC and DANN | 50.90 | +0.01 | 7/10 | 0.955 | 0.02029 | -10.0 | 1/10 | 0.004 |
| DSAFLite+PGMC, submitted architecture | PGMC batch ordering | 50.87 | -0.02 | 5/10 | 0.771 | 0.02207 | -1.1 | 2/10 | 0.061 |

Δ (%) for the mean upward step is relative to the variant: (variant − full) / variant.
