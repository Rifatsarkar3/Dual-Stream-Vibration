"""
exp21_efficiency.py — Computational profile of the final model.

Measures (for the manuscript's efficiency table, replacing unverified claims):
  - inference parameter count (exact)
  - FP32 GPU latency per 100 ms window (batch=1, median of 300 runs after warmup)
  - throughput at batch=256
  - peak VRAM at batch=1
  - PAVA post-processing time per bearing (CPU)
  - approximate MACs/GFLOPs via torch profiler flops counting (fallback: manual)

RUN: .venv\\Scripts\\python.exe Experiments/exp21_efficiency.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from Experiments.exp16_ress import CONFIG, make_model  # noqa: E402
from src.isotonic_eval import IsotonicRegressor  # noqa: E402


def main():
    device = torch.device('cuda')
    model = make_model(CONFIG, device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'inference parameters: {n_params:,}')

    x1 = torch.randn(1, 1, 2560, device=device)

    # Latency batch=1
    for _ in range(50):
        with torch.no_grad():
            model(x1)
    torch.cuda.synchronize()
    times = []
    for _ in range(300):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x1)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    lat_med = float(np.median(times))
    lat_p95 = float(np.percentile(times, 95))
    print(f'FP32 latency batch=1: median {lat_med:.3f} ms, p95 {lat_p95:.3f} ms')

    # Peak VRAM batch=1
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x1)
    torch.cuda.synchronize()
    vram_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f'peak VRAM batch=1: {vram_mb:.1f} MB')

    # Throughput batch=256
    xb = torch.randn(256, 1, 2560, device=device)
    for _ in range(10):
        with torch.no_grad():
            model(xb)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            model(xb)
    torch.cuda.synchronize()
    thr = 256 * 50 / (time.perf_counter() - t0)
    print(f'throughput batch=256: {thr:,.0f} windows/s')

    # FLOPs via torch profiler
    gflops = None
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True) as prof:
            with torch.no_grad():
                model(x1)
        total_flops = sum(e.flops for e in prof.key_averages() if e.flops)
        gflops = total_flops / 1e9
        print(f'GFLOPs per window (profiler, conv/linear only): {gflops:.4f}')
    except Exception as e:
        print(f'profiler flops failed: {e}')

    # PAVA timing per bearing (1500-point sequence, CPU)
    rng = np.random.default_rng(0)
    seq = np.clip(np.linspace(1, 0, 1500) + rng.normal(0, 0.08, 1500), 0, 1)
    t0 = time.perf_counter()
    reps = 100
    for _ in range(reps):
        IsotonicRegressor(increasing=False).fit_transform(seq.copy())
    pava_us = (time.perf_counter() - t0) / reps * 1e6
    print(f'PAVA per 1500-window bearing: {pava_us:.0f} us (CPU)')

    out = Path('outputs/exp21_efficiency/results.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'n_params': int(n_params),
                   'latency_ms_median': lat_med, 'latency_ms_p95': lat_p95,
                   'peak_vram_mb_b1': float(vram_mb),
                   'throughput_b256_per_s': float(thr),
                   'gflops_per_window': gflops,
                   'pava_us_per_bearing_1500w': float(pava_us),
                   'gpu': torch.cuda.get_device_name(0),
                   'real_time_factor': 100.0 / lat_med}, f, indent=2)
    print(f'saved {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
