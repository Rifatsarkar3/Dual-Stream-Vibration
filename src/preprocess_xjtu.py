"""
preprocess_xjtu.py — XJTU-SY Bearing Dataset Preprocessing
===========================================================

Converts the raw XJTU-SY run-to-failure CSV recordings into the same npz
schema used by the PRONOSTIA pipeline ('vibration' (N, 2560) float32,
'rul' (N,) float32 normalized [1 -> 0]), so the exp16 training code can be
reused unchanged.

DATASET FACTS (Wang et al., IEEE Trans. Reliability 2020):
  - 15 bearings (LDK UER204), 3 operating conditions:
      Condition 1: 35 Hz / 12 kN  -> Bearing1_1 .. Bearing1_5
      Condition 2: 37.5 Hz / 11 kN -> Bearing2_1 .. Bearing2_5
      Condition 3: 40 Hz / 10 kN  -> Bearing3_1 .. Bearing3_5
  - Sampling: 25.6 kHz, 1.28 s (32768 samples) recorded every minute.
  - One CSV per minute: columns 'Horizontal_vibration_signals',
    'Vertical_vibration_signals'. Horizontal channel is used (same axis
    convention as our PRONOSTIA preprocessing).

LABELS:
  Total life of a bearing = number of minute-files N_f. All windows cut from
  file i (1-indexed) get RUL = (N_f - i) / N_f, matching the PRONOSTIA
  chronological normalization y_t = (T - t)/T.

WINDOWS:
  windows_per_file non-overlapping 2560-sample (100 ms) windows are cut from
  the start of each 32768-sample recording (max 12). Default 12 gives a
  training volume comparable to PRONOSTIA (~11k windows for 4 bearings).

SPLIT (mirrors the PRONOSTIA protocol):
  train: Bearing1_1, Bearing1_2, Bearing2_1, Bearing2_2
  test (in-domain conditions): Bearing1_3..1_5, Bearing2_3..2_5
  zero-shot (unseen Condition 3): Bearing3_1..3_5  (saved with 'zeroshot_' prefix)

USAGE:
  .venv\\Scripts\\python.exe src/preprocess_xjtu.py ^
      --raw-dir  data/XJTU_SY_extracted ^
      --out-dir  Processed_XJTU
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

WINDOW = 2560

TRAIN = ['Bearing1_1', 'Bearing1_2', 'Bearing2_1', 'Bearing2_2']
TEST = ['Bearing1_3', 'Bearing1_4', 'Bearing1_5',
        'Bearing2_3', 'Bearing2_4', 'Bearing2_5']
ZEROSHOT = ['Bearing3_1', 'Bearing3_2', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5']


def find_bearing_dirs(raw_dir: Path) -> dict:
    """Locate Bearing*_* directories anywhere under raw_dir (condition folders
    have names like '35Hz12kN')."""
    found = {}
    for d in raw_dir.rglob('Bearing*_*'):
        if d.is_dir():
            found[d.name] = d
    return found


def numeric_key(p: Path) -> int:
    try:
        return int(p.stem)
    except ValueError:
        return 10 ** 9


def load_file_horizontal(csv_path: Path) -> np.ndarray:
    """Fast CSV read of the horizontal channel (column 0)."""
    # np.loadtxt is slow for 9k files; manual parse of col 0 is ~3x faster.
    out = np.empty(32768, dtype=np.float32)
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        h_col = 0
        for i, name in enumerate(header):
            if 'horizontal' in name.strip().lower():
                h_col = i
                break
        n = 0
        for row in reader:
            if n >= 32768:
                break
            out[n] = float(row[h_col])
            n += 1
    return out[:n]


def process_bearing(bdir: Path, windows_per_file: int) -> tuple:
    files = sorted(bdir.glob('*.csv'), key=numeric_key)
    n_files = len(files)
    if n_files == 0:
        raise FileNotFoundError(f'no csv files in {bdir}')
    vibs, ruls = [], []
    for i, f in enumerate(files, start=1):
        sig = load_file_horizontal(f)
        k = min(windows_per_file, len(sig) // WINDOW)
        rul = (n_files - i) / n_files
        for w in range(k):
            vibs.append(sig[w * WINDOW:(w + 1) * WINDOW])
            ruls.append(rul)
    return np.stack(vibs).astype(np.float32), np.array(ruls, dtype=np.float32), n_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--out-dir', default='Processed_XJTU')
    ap.add_argument('--windows-per-file', type=int, default=12)
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = find_bearing_dirs(raw_dir)
    print(f'found bearing dirs: {sorted(dirs)}')
    missing = [b for b in TRAIN + TEST + ZEROSHOT if b not in dirs]
    if missing:
        print(f'ERROR: missing bearings {missing}', file=sys.stderr)
        return 1

    manifest = {}
    for role, names in (('train', TRAIN), ('test', TEST), ('zeroshot', ZEROSHOT)):
        for idx, name in enumerate(names, start=1):
            t0 = time.time()
            vib, rul, n_files = process_bearing(dirs[name], args.windows_per_file)
            out = out_dir / f'{role}_{idx:02d}_{name}.npz'
            np.savez_compressed(out, vibration=vib, rul=rul)
            manifest[name] = {'role': role, 'files': n_files, 'windows': len(rul)}
            print(f'{out.name}: {n_files} files -> {len(rul)} windows '
                  f'({time.time() - t0:.1f}s)', flush=True)

    import json
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    total = sum(m['windows'] for m in manifest.values())
    print(f'\nDone. {total} windows total. Manifest saved.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
