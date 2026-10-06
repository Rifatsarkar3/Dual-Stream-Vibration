# Published maintenance cost ratios on the break-even analysis (revision R1, Amendment 1, E4)

| Study | Reported costs | Ratios | Checked |
|---|---|---|---|
| Mitici, de Pater, Barros, Zeng (RESS 234, 2023, 109199), Section 5.3 | preventive replacement c_r = 10, failure c_f = 50, generic maintenance slot c_g = 10 | failure / preventive = 5; failure / (preventive + generic slot) = 2.5 | read in the published version (TU Delft repository) |
| de Pater, Reijns, Mitici (RESS 221, 2022, 108341), Section 4.2 | additional maintenance task c_p = 10,000, engine failure c_f = 50,000, rescheduling c_r = 5,000, generic slot c_g = 10^6; scheduling penalties p_late = 1000 and p_early = 1 per day | failure / preventive = 5; late / early, per day = 1000 | read in the published version (TU Delft repository) |
| Nguyen, Medjaher (RESS 188, 2019, 251-262), Sections 4.2 and 4.3 | preventive replacement C_p = 100, corrective replacement C_c = 500 (includes the downtime cost), inventory C_i = 0.1 per unit time, out-of-stock C_os = 10 (Table 3); case B of Table 4 C_c = 200; Fig. 8a-c C_c = 10 C_p; sensitivity analyses over C_c / C_p from 1 to 30 (Fig. 8d) and 1 to 100 (Fig. 9) | corrective / preventive, Table 3 = 5; Table 4, case B = 2; Fig. 8a-c = 10 | read in the published version (authors' copy of the article) |

Decision-optimal rule at each ratio (lower envelope of r A + B, ten PRONOSTIA test bearings; break-even ratios in the last column):

| Version | τ | r = 2 | r = 2.5 | r = 5 | r = 10 | r = 1000 | break-even ratios |
|---|---|---|---|---|---|---|---|
| adopted variant | 0.30 | CES | CES | calibrated lower bound, CES mean | calibrated lower bound, CES mean | calibrated lower bound, raw mean | PAVA (offline) 0.50–0.86, CES 0.86–2.86, raw 2.86–3.23, calibrated lower bound, CES mean 3.23–12.29, calibrated lower bound, raw mean 12.29–100.00 |
| adopted variant | 0.20 | raw | raw | calibrated lower bound, raw mean | calibrated lower bound, raw mean | calibrated lower bound, raw mean | PAVA (offline) 0.50–0.94, CES 0.94–1.82, raw 1.82–4.69, calibrated lower bound, raw mean 4.69–100.00 |
| adopted variant | 0.10 | PAVA (offline) | CES | raw | calibrated lower bound, CES mean | calibrated lower bound, raw mean | PAVA (offline) 0.50–2.02, CES 2.02–2.76, raw 2.76–5.79, calibrated lower bound, CES mean 5.79–17.66, calibrated lower bound, raw mean 17.66–100.00 |
| submitted architecture | 0.30 | raw | raw | raw | calibrated lower bound, raw mean | calibrated lower bound, raw mean | PAVA (offline) 0.50–0.79, CES 0.79–1.90, raw 1.90–5.85, calibrated lower bound, CES mean 5.85–9.10, calibrated lower bound, raw mean 9.10–100.00 |
| submitted architecture | 0.20 | CES | CES | calibrated lower bound, CES mean | calibrated lower bound, raw mean | calibrated lower bound, raw mean | PAVA (offline) 0.50–1.07, CES 1.07–2.52, raw 2.52–4.10, calibrated lower bound, CES mean 4.10–9.26, calibrated lower bound, raw mean 9.26–100.00 |
| submitted architecture | 0.10 | PAVA (offline) | PAVA (offline) | raw | calibrated lower bound, CES mean | calibrated lower bound, raw mean | PAVA (offline) 0.50–2.52, CES 2.52–4.69, raw 4.69–5.50, calibrated lower bound, CES mean 5.50–16.90, calibrated lower bound, raw mean 16.90–100.00 |
