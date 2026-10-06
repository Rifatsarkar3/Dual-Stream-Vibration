"""
prepare_release.py: copy the revision's code, plan, registry, result files, test predictions and our final
checkpoints into the local public-release repository (github_release/). Nothing is committed or pushed.

Copied: the code the revision's scripts import (listed in release_files.txt next to this script), the analysis plan
and experiment registry, outputs/r1_analysis, the test predictions of every reported model on both bearing datasets
(including the rebuilt network and its ablations of Amendment 10), the Amendment 8 outputs, the C-MAPSS runs of
Amendment 11 (predictions, calibration errors and run records), and the trained models of our final runs
(PRONOSTIA and XJTU-SY). Not copied: datasets, baseline checkpoints and the checkpoints of the rebuilt network
(several hundred MB; available from the authors on request) and intermediate files.

USAGE:
  python paper/revision_R1/manuscript_R1/prepare_release.py [release_files.txt]
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REL = ROOT / 'github_release'
LIST = Path(__file__).resolve().parent / 'release_files.txt'
A10_VARIANTS = ['full', 'rawonly', 'static', 'no_pgmc', 'old_env', 'no_aug', 'adjacent', 'no_dann', 'spectral']
A10_XJTU_VARIANTS = ['full', 'rawonly', 'static', 'no_pgmc']
PRED_DIRS = ['outputs/r1_final_B10/final20', 'outputs/r1_final_temporal_B5/final20', 'outputs/r1_reeval/baseline_cnn',
             'outputs/r1_baselines_final5/cnnlstm', 'outputs/r1_baselines_final5/bilstm',
             'outputs/r1_reeval/baseline_resnet', 'outputs/r1_mscnn/final', 'outputs/r1_reeval/published_tsmccnn',
             'outputs/r1_baselines_final5/transformer', 'outputs/exp17_xjtu/final20',
             'outputs/r1_final_temporal_xjtu/final20', 'outputs/r1_xjtu_baselines/final/cnn',
             'outputs/r1_xjtu_baselines/final/bilstm', 'outputs/r1_xjtu_baselines/final/cnnlstm']
PRED_DIRS += [f'outputs/r1_components_final/{v}' for v in A10_VARIANTS]
PRED_DIRS += [f'outputs/r1_components_xjtu_final/{v}' for v in A10_XJTU_VARIANTS]
CKPT_DIRS = ['outputs/r1_final_B10/final20', 'outputs/exp17_xjtu/final20']
WHOLE_DIRS = ['outputs/r1_analysis', 'outputs/r1_tsmc_randsplit', 'outputs/r1_cmapss', 'outputs/r1_cmapss_smoke']
SUMMARY = {'outputs/r1_components': ['*.json', '*.txt', 'launchers/*.sh'],
           'outputs/r1_components_final': ['noise.json'], 'outputs/r1_components_xjtu_final': ['noise.json']}
EXTRA = ['paper/revision_R1/R1_analysis_plan.md', 'paper/revision_R1/R1_experiment_registry.md',
         'Experiments/__init__.py', 'src/__init__.py']


def copy(rel):
    src, dst = ROOT / rel, REL / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def gitignore_rules():
    rules = [f'!{d}' for d in WHOLE_DIRS]
    nested = PRED_DIRS + CKPT_DIRS + list(SUMMARY)
    for d in sorted({'/'.join(p.split('/')[:k]) for p in nested for k in range(2, len(p.split('/')) + 1)}):
        rules += [f'!{d}', f'{d}/*']
    for d in PRED_DIRS:
        rules += [f'!{d}/*preds_seed*.npz', f'!{d}/*.json']
    for d in CKPT_DIRS:
        rules.append(f'!{d}/model_seed*.pth')
    for d, pats in SUMMARY.items():
        for pat in pats:
            if '/' in pat:
                sub = pat.split('/')[0]
                rules += [f'!{d}/{sub}', f'{d}/{sub}/*']
            rules.append(f'!{d}/{pat}')
    out, seen = [], set()
    for r in rules:
        if r not in seen:
            out.append(r)
            seen.add(r)
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    listing = Path(sys.argv[1]) if len(sys.argv) > 1 else LIST
    files = [l.strip() for l in listing.read_text(encoding='utf-8').splitlines() if l.strip()]
    files += EXTRA
    for d in WHOLE_DIRS:
        files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / d).rglob('*')) if p.is_file()]
    for d in PRED_DIRS:
        files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / d).glob('*preds_seed*.npz'))]
        files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / d).glob('*.json'))]
    for d in CKPT_DIRS:
        files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / d).glob('model_seed*.pth'))]
    for d, pats in SUMMARY.items():
        for pat in pats:
            files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / d).glob(pat)) if p.is_file()]
    files = sorted(set(f for f in files if '_tmp' not in Path(f).name))
    missing = [f for f in files if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f'missing files: {missing[:5]}')
    total = sum(copy(f) for f in files)
    gi = REL / '.gitignore'
    text = gi.read_text(encoding='utf-8')
    present = set(text.splitlines())
    new = [r for r in gitignore_rules() if r not in present]
    if new:
        header = '\n# Revision 1 (2026), Amendments 10 and 11: rebuilt components, C-MAPSS check\n'
        gi.write_text(text.rstrip('\n') + '\n' + header + '\n'.join(new) + '\n', encoding='utf-8')
    print(f'copied {len(files)} files, {total / 2**20:.1f} MB, into {REL}; {len(new)} new .gitignore rules')


if __name__ == '__main__':
    main()
