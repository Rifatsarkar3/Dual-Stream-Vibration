"""
r1_components.py — Revision R1, analysis plan Amendment 10: the dual-stream network with the
rebuilt components of src/components_r2.py, run through Experiments/exp16_ress.py.

  E  envelope stream on one resonance band, log amplitude, no per-window normalization
  G  input noise during training (random SNR per window), so that the gate has stream
     reliabilities to adapt to
  M  PGMC on window pairs 5 to 10 % of the bearing life apart

Inside this process only, three exp16 functions are replaced (exp16 itself is not modified,
so its code hash in earlier result files stays valid):
  make_model              -> exp16 model, envelope extractor replaced (E), wrapped with
                             InputNoise (G)
  build_loaders           -> the auxiliary PGMC loader draws window pairs (M)
  sequential_pgmc_penalty -> squared hinge on those pairs (M)
Every exp16 option works as before, including the ablations (--stream temporal|spectral,
--static-gate, --no-pgmc, --no-dann). Three options of this script undo one fix each:
  --envelope submitted    the submitted extractor (whole band, divided by the window maximum)
  --augment none          no input noise
  --pgmc-pairs adjacent   the submitted adjacent-window PGMC
With all three, model and training are exactly exp16's (tests/test_components_r2.py).

The constants below are fixed by Amendment 10 and are not tuned. The band comes from
Experiments/r1_band_select.py (training bearings only).

USAGE (component options first or anywhere; all other options are exp16's):
  python Experiments/r1_components.py --all-seeds --pgmc-mode aux-seq --val-mode bearing \
      --val-bearings train_02_Bearing1_2 --epochs 10 --proportional-warmup --mc-passes 5 \
      --patience 999 --skip-test --save-val-preds --deterministic \
      --data-dir Processed_PRONOSTIA_v2 --output-root outputs/r1_components \
      --variant-name lam0.08_B10_foldA
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import Experiments.exp16_ress as e16  # noqa: E402
from Experiments.r1_provenance import provenance as _provenance  # noqa: E402
from src.components_r2 import (  # noqa: E402
    BandEnvelopeSpectrumExtractor, InputNoise, PairBatchSampler, WithInputNoise, pair_penalty)

# Fixed by Amendment 10.
FS = 25600.0                    # both datasets
SNR_DB = (0.0, 30.0)            # input-noise SNR range during training
P_NOISE = 0.5                   # share of training windows that receive noise
GAP_FRAC = (0.05, 0.10)         # pair gap as a fraction of the bearing's training windows
N_PAIRS = 16                    # pairs per auxiliary batch (32 windows, as the 32-window chunks)
PAIR_SEED_OFFSET = 7919         # pair sampler generator seed = run seed + offset
BAND_JSON = {'pronostia': ROOT / 'outputs' / 'r1_components' / 'band_pronostia.json',
             'xjtu': ROOT / 'outputs' / 'r1_components' / 'band_xjtu.json'}
CODE = ('Experiments/r1_components.py', 'src/components_r2.py')

PATCHED = ('make_model', 'build_loaders', 'sequential_pgmc_penalty')


def _genuine(name):
    """exp16's own function, or None if another wrapper (e.g. Experiments/r1_temporal.py, which
    patches exp16 when imported) replaced it in this process before this module was imported."""
    f = getattr(e16, name)
    return f if getattr(f, '__module__', None) == e16.__name__ else None


_ORIGINAL = {name: _genuine(name) for name in PATCHED}
SETTINGS = {}


def make_model(config, device, static_gate=False, stream='dual'):
    model = _ORIGINAL['make_model'](config, device, static_gate=static_gate, stream=stream)
    if SETTINGS['envelope'] == 'band':
        model.stream_b.extractor = BandEnvelopeSpectrumExtractor(SETTINGS['band_hz'], fs=FS)
    if SETTINGS['augment'] == 'awgn':
        model = WithInputNoise(model, InputNoise(SNR_DB, P_NOISE))
    return model.to(device)


def build_loaders(config, seed):
    train_loader, val_loader, seq_loader, test_bearings = _ORIGINAL['build_loaders'](config, seed)
    if seq_loader is not None:  # aux-seq PGMC
        ds = train_loader.dataset
        gen = torch.Generator()
        gen.manual_seed(seed + PAIR_SEED_OFFSET)
        sampler = PairBatchSampler(ds.entries, N_PAIRS, GAP_FRAC,
                                   max(1, len(ds) // (2 * N_PAIRS)), gen)
        seq_loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)
    return train_loader, val_loader, seq_loader, test_bearings


def pair_pgmc_penalty(model, seq_iter, seq_loader, device):
    """exp16's sequential_pgmc_penalty with window pairs instead of adjacent windows."""
    try:
        batch = next(seq_iter)
    except StopIteration:
        seq_iter = iter(seq_loader)
        batch = next(seq_iter)
    x = batch[0].to(device)
    with e16._FrozenBN(model):
        pred, *_ = model(x)
    return pair_penalty(pred, pred.shape[0] // 2), seq_iter


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe() -> dict:
    """Component settings recorded in every result file."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in SETTINGS.items()}


def configure(envelope='band', augment='awgn', pgmc_pairs='gap', dataset='pronostia',
              band_json=None, band=None):
    """Select the fixes and patch exp16 accordingly; with every fix off, exp16 is restored."""
    if envelope not in ('band', 'submitted') or augment not in ('awgn', 'none') \
            or pgmc_pairs not in ('gap', 'adjacent'):
        raise ValueError((envelope, augment, pgmc_pairs))
    ours = {'make_model': make_model, 'build_loaders': build_loaders,
            'sequential_pgmc_penalty': pair_pgmc_penalty}
    foreign = [n for n in PATCHED
               if _ORIGINAL[n] is None or getattr(e16, n) not in (_ORIGINAL[n], ours[n])]
    if foreign:
        raise RuntimeError(f'exp16 {foreign} replaced by another wrapper in this process; '
                           'run r1_components in a process of its own')
    SETTINGS.clear()
    SETTINGS.update(envelope=envelope, augment=augment, pgmc_pairs=pgmc_pairs, dataset=dataset)
    if envelope == 'band':
        if band is None:
            path = Path(band_json) if band_json else BAND_JSON[dataset]
            if not path.exists():
                raise FileNotFoundError(f'{path}: run Experiments/r1_band_select.py --dataset '
                                        f'{dataset} first')
            sel = json.loads(path.read_text(encoding='utf-8'))
            if sel['dataset'] != dataset:
                raise ValueError(f'{path} is for {sel["dataset"]}, not {dataset}')
            band = sel['band_hz']
            try:
                shown = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                shown = str(path)
            SETTINGS.update(band_json=shown, band_json_sha256=_sha256(path))
        SETTINGS['band_hz'] = (float(band[0]), float(band[1]))
    if augment == 'awgn':
        SETTINGS.update(snr_db=SNR_DB, p_noise=P_NOISE)
    if pgmc_pairs == 'gap':
        SETTINGS.update(gap_frac=GAP_FRAC, n_pairs=N_PAIRS, pair_seed_offset=PAIR_SEED_OFFSET)

    changed = envelope == 'band' or augment == 'awgn'
    e16.make_model = make_model if changed else _ORIGINAL['make_model']
    gap = pgmc_pairs == 'gap'
    e16.build_loaders = build_loaders if gap else _ORIGINAL['build_loaders']
    e16.sequential_pgmc_penalty = pair_pgmc_penalty if gap else _ORIGINAL['sequential_pgmc_penalty']
    e16.provenance = lambda: {**_provenance(extra_code=CODE), 'components_r2': describe()}


def parse_component_args(argv, ablation=False):
    """Split this script's options from the harness options."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--envelope', choices=['band', 'submitted'], default='band')
    ap.add_argument('--augment', choices=['awgn', 'none'], default='awgn')
    ap.add_argument('--pgmc-pairs', choices=['gap', 'adjacent'], default='gap')
    ap.add_argument('--band-json', default=None)
    if ablation:  # exp17 has no ablation flags of its own
        ap.add_argument('--ablation', default='none',
                        choices=['none', 'temporal', 'spectral', 'static', 'no_pgmc', 'no_dann'])
    return ap.parse_known_args(argv)


def main():
    comp, rest = parse_component_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    configure(comp.envelope, comp.augment, comp.pgmc_pairs, 'pronostia', comp.band_json)
    print(f'[r1_components] {describe()}', flush=True)
    return e16.main()


if __name__ == '__main__':
    sys.exit(main())
