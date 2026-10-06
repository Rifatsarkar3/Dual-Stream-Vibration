"""Integrity checks for the processed PRONOSTIA windows (acc_*.csv only, no temp logs)."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / 'Processed_PRONOSTIA_v2'
RAW = ROOT / 'phm-ieee-2012-data-challenge-dataset-master'

pytestmark = pytest.mark.skipif(not (PROCESSED / 'manifest.json').exists(),
                                reason='run src/preprocess_pronostia.py first')


def _manifest():
    return json.loads((PROCESSED / 'manifest.json').read_text())


@pytest.mark.parametrize('name', sorted(json.loads((PROCESSED / 'manifest.json').read_text()))
                         if (PROCESSED / 'manifest.json').exists() else [])
def test_window_count_equals_acc_file_count(name):
    raw_folder = RAW / _manifest()[name]['raw']
    n_acc = sum(1 for p in raw_folder.iterdir() if p.name.startswith('acc_'))
    d = np.load(PROCESSED / f'{name}.npz')
    assert d['vibration'].shape == (n_acc, 2560)
    assert d['rul'].shape == (n_acc,)


@pytest.mark.parametrize('name', sorted(json.loads((PROCESSED / 'manifest.json').read_text()))
                         if (PROCESSED / 'manifest.json').exists() else [])
def test_labels_are_linear_ramp_to_zero(name):
    rul = np.load(PROCESSED / f'{name}.npz')['rul']
    T = len(rul)
    assert np.allclose(rul, (T - np.arange(T)) / T, atol=1e-6)


@pytest.mark.parametrize('name', sorted(json.loads((PROCESSED / 'manifest.json').read_text()))
                         if (PROCESSED / 'manifest.json').exists() else [])
def test_no_zero_padded_windows(name):
    # Temperature logs hold 600 samples; read as vibration they leave a zero-padded tail.
    vib = np.load(PROCESSED / f'{name}.npz')['vibration']
    tail_zeros = (vib[:, -1000:] == 0).all(axis=1)
    assert not tail_zeros.any()
