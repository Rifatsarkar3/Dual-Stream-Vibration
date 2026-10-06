"""
preprocess_pronostia.py — PRONOSTIA (IEEE PHM 2012 / FEMTO-ST) raw CSV -> RUL-labelled windows.

Reads ONLY the accelerometer snapshots (acc_*.csv). Several Full_Test_Set folders
(e.g. Bearing1_4, Bearing1_5) also hold temperature logs (temp_*.csv); a generic
*.csv glob appends those records after the vibration snapshots and inflates the
trajectory length T used for the RUL labels. This script never reads them.

Raw format: one file per 0.1 s snapshot (2,560 samples at 25.6 kHz) recorded every
10 s. Columns: hour, minute, second, microsecond, horizontal acc, vertical acc.
Delimiter is ',' except Full_Test_Set/Bearing1_4, which uses ';'.

Output (one .npz per bearing, same keys the experiment loaders read):
  vibration : (T, 2560) float32, horizontal channel, recorded units, chronological
  rul       : (T,) float32, y_i = (T - i) / T for i = 0..T-1
  vibration_v : (T, 2560) float32, vertical channel (only with --with-vertical)
plus manifest.json with per-bearing file counts.

USAGE:
  python src/preprocess_pronostia.py                         # all 17 bearings
  python src/preprocess_pronostia.py --verify-against Processed_PRONOSTIA
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WINDOW = 2560
HORIZONTAL_COL = 4
VERTICAL_COL = 5

# processed-file name -> (raw subset folder, bearing folder)
BEARINGS = {
    # training bearings (Conditions 1-2), the paper's training set
    'train_01_Bearing1_1': ('Learning_set', 'Bearing1_1'),
    'train_02_Bearing1_2': ('Learning_set', 'Bearing1_2'),
    'train_03_Bearing2_1': ('Learning_set', 'Bearing2_1'),
    'train_04_Bearing2_2': ('Learning_set', 'Bearing2_2'),
    # original four test bearings
    'test_01_Bearing1_3': ('Full_Test_Set', 'Bearing1_3'),
    'test_02_Bearing1_4': ('Full_Test_Set', 'Bearing1_4'),
    'test_03_Bearing1_5': ('Full_Test_Set', 'Bearing1_5'),
    'test_04_Bearing2_3': ('Full_Test_Set', 'Bearing2_3'),
    # remaining Condition 1-2 bearings of the official test set (never used for selection)
    'test_05_Bearing1_6': ('Full_Test_Set', 'Bearing1_6'),
    'test_06_Bearing1_7': ('Full_Test_Set', 'Bearing1_7'),
    'test_07_Bearing2_4': ('Full_Test_Set', 'Bearing2_4'),
    'test_08_Bearing2_5': ('Full_Test_Set', 'Bearing2_5'),
    'test_09_Bearing2_6': ('Full_Test_Set', 'Bearing2_6'),
    'test_10_Bearing2_7': ('Full_Test_Set', 'Bearing2_7'),
    # Condition 3 (unseen operating condition; zero-shot only)
    'zs_01_Bearing3_1': ('Learning_set', 'Bearing3_1'),
    'zs_02_Bearing3_2': ('Learning_set', 'Bearing3_2'),
    'zs_03_Bearing3_3': ('Full_Test_Set', 'Bearing3_3'),
}

ACC_RE = re.compile(r'^acc_(\d+)\.csv$')


def acc_files(folder: Path) -> list:
    """Accelerometer snapshots only, in recording order (numeric index)."""
    found = [(int(m.group(1)), p) for p in folder.iterdir()
             if (m := ACC_RE.match(p.name))]
    found.sort()
    idx = [i for i, _ in found]
    if idx != list(range(1, len(idx) + 1)):
        raise ValueError(f'{folder}: acc file indices are not contiguous from 1')
    return [p for _, p in found]


def read_channels(path: Path, sep: str, cols: list) -> np.ndarray:
    df = pd.read_csv(path, header=None, sep=sep, usecols=cols, dtype=np.float64, engine='c')
    if df.shape[0] != WINDOW:
        raise ValueError(f'{path}: {df.shape[0]} rows, expected {WINDOW}')
    return df[cols].to_numpy().T.astype(np.float32)          # (len(cols), 2560)


def process_bearing(folder: Path, with_vertical: bool = False) -> tuple:
    files = acc_files(folder)
    first_line = files[0].read_text().splitlines()[0]
    sep = ';' if ';' in first_line else ','
    cols = [HORIZONTAL_COL, VERTICAL_COL] if with_vertical else [HORIZONTAL_COL]
    data = np.stack([read_channels(f, sep, cols) for f in files])   # (T, C, 2560)
    T = len(files)
    rul = ((T - np.arange(T)) / T).astype(np.float32)
    n_temp = sum(1 for p in folder.iterdir() if p.name.startswith('temp_'))
    arrays = {'vibration': data[:, 0], 'rul': rul}
    if with_vertical:
        arrays['vibration_v'] = data[:, 1]
    return arrays, {'acc_files': T, 'temp_files_ignored': n_temp, 'delimiter': sep,
                    'windows': T, 'duration_h': round(T * 10 / 3600, 3)}


def verify(new_dir: Path, old_dir: Path) -> bool:
    """Compare against a previously processed directory, window by window."""
    ok = True
    for name in BEARINGS:
        old_f, new_f = old_dir / f'{name}.npz', new_dir / f'{name}.npz'
        if not old_f.exists():
            continue
        old, new = np.load(old_f), np.load(new_f)
        n_old, n_new = len(old['rul']), len(new['rul'])
        n = min(n_old, n_new)
        same_prefix = np.array_equal(old['vibration'][:n], new['vibration'][:n])
        same_labels = n_old == n_new and np.allclose(old['rul'], new['rul'], atol=1e-6)
        status = 'IDENTICAL' if (same_prefix and same_labels) else (
            f'PREFIX-IDENTICAL, {n_old - n_new:+d} windows in old file, labels differ'
            if same_prefix else 'VIBRATION DIFFERS')
        if not same_prefix:
            ok = False
        print(f'  {name:22s} old={n_old:5d} new={n_new:5d}  {status}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', default=str(ROOT / 'phm-ieee-2012-data-challenge-dataset-master'))
    ap.add_argument('--out-dir', default=str(ROOT / 'Processed_PRONOSTIA_v2'))
    ap.add_argument('--only', type=str, default=None,
                    help='comma-separated processed names to (re)build')
    ap.add_argument('--verify-against', type=str, default=None,
                    help='old processed dir to compare with (read-only)')
    ap.add_argument('--with-vertical', action='store_true',
                    help="also store the vertical channel as 'vibration_v'")
    args = ap.parse_args()

    raw_root, out_dir = Path(args.raw_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ([s.strip() for s in args.only.split(',')] if args.only else list(BEARINGS))

    manifest_path = out_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in names:
        subset, bearing = BEARINGS[name]
        t0 = time.time()
        arrays, info = process_bearing(raw_root / subset / bearing, args.with_vertical)
        np.savez_compressed(out_dir / f'{name}.npz', **arrays)
        manifest[name] = {'raw': f'{subset}/{bearing}', **info}
        print(f'{name:22s} windows={info["windows"]:5d}  temp_ignored={info["temp_files_ignored"]:4d}  '
              f'sep="{info["delimiter"]}"  ({time.time() - t0:.1f}s)', flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if args.verify_against:
        print(f'\nverification against {args.verify_against}:')
        return 0 if verify(out_dir, Path(args.verify_against)) else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
