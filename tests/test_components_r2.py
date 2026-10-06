"""Unit tests for the rebuilt components of analysis plan Amendment 10 (src/components_r2.py,
Experiments/r1_components.py, Experiments/r1_band_select.py). Synthetic signals only; no data
files are read.

E: the band envelope stream finds a defect modulation that an out-of-band tone hides from the
   submitted extractor, keeps amplitude, and rejects out-of-band content.
G: input noise reaches its SNR, is applied to the stated share of windows, and is off in eval
   mode and under MC dropout.
M: window pairs come from one bearing, in chronological order, with the stated gaps, and the
   auxiliary pass leaves BatchNorm statistics unchanged.
Wrapper: with every fix off it restores exp16 exactly; with every fix on the network has the
   submitted initial weights plus two BatchNorm parameters.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.components_r2 import (  # noqa: E402
    BandEnvelopeSpectrumExtractor, InputNoise, PairBatchSampler, WithInputNoise, pair_penalty)
from src.dsaf_v2_lite import EnvelopeSpectrumExtractor  # noqa: E402

FS, L = 25600.0, 2560
T = torch.arange(L, dtype=torch.float64) / FS
BAND = (3200.0, 4800.0)


def am_signal(carrier=4000.0, mod=160.0, depth=0.8, tone=1000.0, tone_amp=3.0):
    """A 4 kHz resonance amplitude-modulated at a 160 Hz defect frequency, plus a strong
    1 kHz tone outside the band (for example a gear mesh or shaft harmonic)."""
    x = (1 + depth * torch.cos(2 * np.pi * mod * T)) * torch.cos(2 * np.pi * carrier * T)
    return (x + tone_amp * torch.cos(2 * np.pi * tone * T)).view(1, 1, L)


# ---- E: band envelope stream ------------------------------------------------------------

def test_band_extractor_finds_the_defect_frequency():
    s = BandEnvelopeSpectrumExtractor(BAND).spectrum(am_signal().float())[0]
    assert int(torch.argmax(s[1:101])) + 1 == 16          # 160 Hz at 10 Hz per bin


def test_submitted_extractor_is_dominated_by_the_out_of_band_tone():
    s = EnvelopeSpectrumExtractor()(am_signal().float())[0, 0]
    assert s[300] > 5 * s[16]                              # 3 kHz beat of tone and carrier


def test_band_extractor_keeps_amplitude_and_submitted_one_does_not():
    ext = BandEnvelopeSpectrumExtractor(BAND)
    x = am_signal().float()
    s1, s10 = ext.spectrum(x)[0], ext.spectrum(10 * x)[0]
    live = s1 > np.log10(ext.eps) + 3
    assert int(live.sum()) >= 2
    assert torch.allclose(s10[live] - s1[live], torch.ones(int(live.sum())), atol=1e-3)
    old = EnvelopeSpectrumExtractor()
    assert torch.allclose(old(x), old(10 * x), atol=1e-6)


def test_band_extractor_rejects_out_of_band_content():
    ext = BandEnvelopeSpectrumExtractor(BAND)
    x = (3.0 * torch.cos(2 * np.pi * 1000.0 * T)).view(1, 1, L)   # float64
    assert float(ext.spectrum(x).max()) < np.log10(ext.eps) + 0.01


def test_band_extractor_shapes_and_parameters():
    ext = BandEnvelopeSpectrumExtractor(BAND)
    assert ext(torch.randn(4, 1, L)).shape == (4, 1, 512)
    assert sum(p.numel() for p in ext.parameters()) == 2


def test_band_outside_the_spectrum_is_rejected():
    with pytest.raises(ValueError):
        BandEnvelopeSpectrumExtractor((0.0, 4000.0))
    with pytest.raises(ValueError):
        BandEnvelopeSpectrumExtractor((5000.0, 13000.0))


# ---- G: input noise ---------------------------------------------------------------------

def test_input_noise_is_identity_in_eval_mode_and_at_p_zero():
    x = torch.randn(8, 1, L)
    assert torch.equal(InputNoise(p=1.0).eval()(x), x)
    assert torch.equal(InputNoise(p=0.0).train()(x), x)


def test_input_noise_reaches_the_snr_whatever_the_window_amplitude():
    torch.manual_seed(0)
    x = torch.randn(256, 1, L) * torch.logspace(-2, 1, 256).view(-1, 1, 1)
    y = InputNoise((10.0, 10.0), p=1.0).train()(x)
    snr = 10 * torch.log10(x.pow(2).mean(dim=(1, 2)) / (y - x).pow(2).mean(dim=(1, 2)))
    assert abs(float(snr.mean()) - 10.0) < 0.1
    assert float(snr.std()) < 0.2


def test_input_noise_reaches_the_stated_share_of_windows():
    torch.manual_seed(1)
    x = torch.randn(4000, 1, 64)
    y = InputNoise((0.0, 30.0), p=0.5).train()(x)
    untouched = float((y == x).all(dim=2).all(dim=1).float().mean())
    assert abs(untouched - 0.5) < 0.04


def test_mc_dropout_does_not_switch_the_noise_on():
    import Experiments.exp16_ress as e16
    base = torch.nn.Sequential(torch.nn.Dropout(0.2))
    model = WithInputNoise(base, InputNoise(p=1.0))
    model.eval()
    e16._enable_mc_dropout(model)
    assert base[0].training and not model.noise.training


# ---- M: pair PGMC -----------------------------------------------------------------------

def make_entries(sizes):
    entries = [(b, w) for b, n in enumerate(sizes) for w in range(n)]
    order = np.random.default_rng(0).permutation(len(entries))
    return [entries[i] for i in order]          # sampler must not rely on the listing order


def test_pair_sampler_pairs_are_within_one_bearing_in_order_with_the_stated_gaps():
    sizes = (100, 300, 50)
    entries = make_entries(sizes)
    sampler = PairBatchSampler(entries, 16, (0.05, 0.10), 200, torch.Generator().manual_seed(3))
    counts = np.zeros(len(sizes))
    for batch in sampler:
        assert len(batch) == 32
        for e_pos, l_pos in zip(batch[:16], batch[16:]):
            (be, we), (bl, wl) = entries[e_pos], entries[l_pos]
            n = sizes[be]
            assert be == bl and wl > we
            assert max(1, round(0.05 * n)) <= wl - we <= round(0.10 * n)
            counts[be] += 1
    assert np.allclose(counts / counts.sum(), np.array(sizes) / sum(sizes), atol=0.03)


def test_pair_sampler_is_reproducible():
    entries = make_entries((100, 300, 50))
    runs = [list(PairBatchSampler(entries, 16, (0.05, 0.10), 5, torch.Generator().manual_seed(7)))
            for _ in range(2)]
    assert runs[0] == runs[1] and len(runs[0]) == 5


def test_pair_penalty():
    assert float(pair_penalty(torch.tensor([0.9, 0.8, 0.5, 0.4]), 2)) == 0.0
    up = pair_penalty(torch.tensor([0.2, 0.5, 0.5, 0.4]), 2)       # late - early: +0.3, -0.1
    assert abs(float(up) - 0.045) < 1e-7


def test_pair_penalty_pass_leaves_batchnorm_statistics_unchanged():
    import Experiments.exp16_ress as e16
    import Experiments.r1_components as rc
    rng = np.random.default_rng(0)
    bearings = [e16.BearingArrays(f'b{i}', rng.standard_normal((n, L)).astype(np.float32),
                                  np.linspace(1, 0, n), 0) for i, n in enumerate((40, 60))]
    ds = e16.WindowDataset(bearings, [np.arange(b.n) for b in bearings])
    loader = torch.utils.data.DataLoader(ds, batch_sampler=PairBatchSampler(
        ds.entries, 4, (0.05, 0.10), 3, torch.Generator().manual_seed(0)))
    from src.dsaf_v2_lite import DSAFLite
    from src.monotone_rul_head import upgrade_dsaflite_regressor
    torch.manual_seed(0)
    model = DSAFLite(feat_dim=128, dropout=0.2)
    upgrade_dsaflite_regressor(model, feat_dim=128, dropout=0.2)
    model.train()
    bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d))
    before = bn.running_mean.clone()
    pen, _ = rc.pair_pgmc_penalty(model, iter(loader), loader, torch.device('cpu'))
    assert pen.ndim == 0 and pen.requires_grad and float(pen.detach()) >= 0.0
    assert torch.equal(bn.running_mean, before)


# ---- wrapper (fresh interpreters: Experiments/r1_temporal.py, imported by another test
# module, patches exp16 on import) --------------------------------------------------------

def run_isolated(code: str):
    res = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=ROOT,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr


def test_wrapper_with_every_fix_off_restores_exp16():
    run_isolated("""
        import Experiments.exp16_ress as e16
        import Experiments.r1_components as rc
        genuine = (e16.make_model, e16.build_loaders, e16.sequential_pgmc_penalty)
        rc.configure('band', 'awgn', 'gap', band=(3200.0, 4800.0))
        assert e16.make_model is rc.make_model and e16.build_loaders is rc.build_loaders
        rc.configure('submitted', 'none', 'adjacent')
        assert (e16.make_model, e16.build_loaders, e16.sequential_pgmc_penalty) == genuine
    """)


def test_wrapper_with_every_fix_on_keeps_the_submitted_initial_weights():
    run_isolated("""
        import torch
        import Experiments.exp16_ress as e16
        import Experiments.r1_components as rc
        from src.components_r2 import BandEnvelopeSpectrumExtractor, WithInputNoise
        cfg, dev = dict(e16.CONFIG), torch.device('cpu')
        genuine = e16.make_model
        rc.configure('band', 'awgn', 'gap', band=(3200.0, 4800.0))
        torch.manual_seed(0); new = e16.make_model(cfg, dev)
        torch.manual_seed(0); old = genuine(cfg, dev)
        assert isinstance(new, WithInputNoise)
        assert isinstance(new.base.stream_b.extractor, BandEnvelopeSpectrumExtractor)
        n_old = sum(p.numel() for p in old.parameters())
        assert sum(p.numel() for p in new.parameters()) == n_old + 2
        new_sd = {k[len('base.'):]: v for k, v in new.state_dict().items()}
        assert all(torch.equal(new_sd[k], v) for k, v in old.state_dict().items())
        pred, alpha, fused, fa, fb = new.eval()(torch.randn(3, 1, 2560))
        assert pred.shape == alpha.shape == (3,) and fused.shape == (3, 128)
        static = e16.make_model(cfg, dev, static_gate=True)
        assert isinstance(static.base.fusion, e16.FixedGate)
    """)


def test_wrapper_refuses_to_stack_on_another_wrapper():
    run_isolated("""
        import Experiments.r1_temporal
        import Experiments.r1_components as rc
        try:
            rc.configure('band', 'awgn', 'gap', band=(3200.0, 4800.0))
        except RuntimeError:
            pass
        else:
            raise AssertionError('configure must refuse a patched exp16')
    """)


# ---- band selection ---------------------------------------------------------------------

def test_candidate_bands():
    from Experiments.r1_band_select import candidate_bands
    bands = candidate_bands()
    assert len(bands) == 52
    assert all(0 < lo < hi <= FS / 2 for lo, hi in bands)


def test_band_selection_finds_an_impact_excited_resonance():
    from Experiments.r1_band_select import select_band
    rng = np.random.default_rng(0)
    t = np.arange(L) / FS
    ring = np.exp(-t / 1e-3) * np.sin(2 * np.pi * 5200.0 * t)   # 5.2 kHz resonance, 1 ms decay
    late = {}
    for b in range(2):
        w = 0.5 * rng.standard_normal((30, L))
        for i in range(30):
            impacts = np.zeros(L)
            impacts[(np.arange(0, L, int(FS / 160)) + rng.integers(0, 20)) % L] = 1.0
            w[i] += 5.0 * np.convolve(impacts, ring)[:L]
        late[f'b{b}'] = w
    band, rows = select_band(late)
    assert band[0] <= 5200.0 <= band[1]
    assert rows[0]['score'] > 1.0 and rows[-1]['score'] < 0.5
