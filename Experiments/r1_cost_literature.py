"""
r1_cost_literature.py — Revision R1, analysis plan Amendment 1, E4: published maintenance cost ratios
placed on the existing break-even analysis of plan Section 9 (decision_*.json from r1_decision.py).
No decision rule, threshold or break-even value is changed; this script only evaluates the existing
lower envelope L_v(r) = r A_v + B_v at the published ratios.

Our r is a ratio of costs per unit time of late and of early maintenance. The studies below report
costs per maintenance event, except the per-day scheduling penalties of de Pater et al. Reading a
per-event ratio as r assumes that each unit of lateness carries the failure cost and each unit of
earliness the preventive cost; the per-day penalties need no such assumption but are penalties of
a scheduling model, not cost estimates.

USAGE:
  python Experiments/r1_cost_literature.py [--out outputs/r1_analysis/cost_ratios_literature.md]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Experiments'))
from r1_provenance import provenance  # noqa: E402

SOURCES = [
    {'study': 'Mitici, de Pater, Barros, Zeng (RESS 234, 2023, 109199), Section 5.3',
     'values': 'preventive replacement c_r = 10, failure c_f = 50, generic maintenance slot c_g = 10',
     'ratios': {'failure / preventive': 5.0, 'failure / (preventive + generic slot)': 2.5},
     'checked': 'read in the published version (TU Delft repository)'},
    {'study': 'de Pater, Reijns, Mitici (RESS 221, 2022, 108341), Section 4.2',
     'values': ('additional maintenance task c_p = 10,000, engine failure c_f = 50,000, rescheduling c_r = 5,000, '
                'generic slot c_g = 10^6; scheduling penalties p_late = 1000 and p_early = 1 per day'),
     'ratios': {'failure / preventive': 5.0, 'late / early, per day': 1000.0},
     'checked': 'read in the published version (TU Delft repository)'},
    {'study': 'Nguyen, Medjaher (RESS 188, 2019, 251-262), Sections 4.2 and 4.3',
     'values': ('preventive replacement C_p = 100, corrective replacement C_c = 500 (includes the downtime cost), '
                'inventory C_i = 0.1 per unit time, out-of-stock C_os = 10 (Table 3); case B of Table 4 C_c = 200; '
                'Fig. 8a-c C_c = 10 C_p; sensitivity analyses over C_c / C_p from 1 to 30 (Fig. 8d) and 1 to 100 (Fig. 9)'),
     'ratios': {'corrective / preventive, Table 3': 5.0, 'Table 4, case B': 2.0, 'Fig. 8a-c': 10.0},
     'checked': 'read in the published version (authors\' copy of the article)'},
]
FILES = {'adopted variant': 'outputs/r1_analysis/decision_temporal_B5_all10.json',
         'submitted architecture': 'outputs/r1_analysis/decision_final20_all10.json'}
NAMES = {'pava': 'PAVA (offline)', 'ces': 'CES', 'raw': 'raw', 'rtrm': 'RTRM',
         'lb_raw': 'calibrated lower bound, raw mean', 'lb_ces': 'calibrated lower bound, CES mean'}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'cost_ratios_literature.md'))
    args = ap.parse_args()
    ratios = sorted({r for s in SOURCES for r in s['ratios'].values()})
    report = {'sources': SOURCES, 'optimal': {}}
    lines = ['# Published maintenance cost ratios on the break-even analysis (revision R1, Amendment 1, E4)', '',
             '| Study | Reported costs | Ratios | Checked |', '|---|---|---|---|']
    for s in SOURCES:
        lines.append(f"| {s['study']} | {s['values']} | "
                     + '; '.join(f'{k} = {v:g}' for k, v in s['ratios'].items()) + f" | {s['checked']} |")
    lines += ['', 'Decision-optimal rule at each ratio (lower envelope of r A + B, ten PRONOSTIA test bearings; '
                  'break-even ratios in the last column):', '',
              '| Version | τ | ' + ' | '.join(f'r = {r:g}' for r in ratios) + ' | break-even ratios |',
              '|---|---|' + '---|' * len(ratios) + '---|']
    for version, f in FILES.items():
        d = json.loads((ROOT / f).read_text(encoding='utf-8'))
        report['optimal'][version] = {'c_star': d['c_star'], 'file': f, 'tau': {}}
        for tau in ('0.30', '0.20', '0.10'):
            st = d['timing'][tau]['stats']
            best = {r: min(st, key=lambda v: r * st[v]['A'] + st[v]['B']) for r in ratios}
            env = d['timing'][tau]['optimal_by_ratio']
            report['optimal'][version]['tau'][tau] = {'best': {str(r): v for r, v in best.items()}, 'envelope': env}
            lines.append(f'| {version} | {tau} | ' + ' | '.join(NAMES.get(best[r], best[r]) for r in ratios) + ' | '
                         + ', '.join(f"{NAMES.get(e['variant'], e['variant'])} {e['r_from']:.2f}–{e['r_to']:.2f}"
                                     for e in env) + ' |')
    report['provenance'] = provenance(('Experiments/r1_cost_literature.py',))
    text = '\n'.join(lines) + '\n'
    print(text)
    Path(args.out).write_text(text, encoding='utf-8')
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'written {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
