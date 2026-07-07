"""
hardware.py
===========
Hardware deployment utilities for Paper #2.

Enforces the strict CPU/GPU split that is the core engineering claim:
  Vibration encoder  → AMD Ryzen 5 7500F (CPU)
  Vision backbone    → NVIDIA RTX 5070 12GB (GPU)
  Fusion + heads     → GPU

Also provides:
  - VRAM usage monitoring
  - Per-model latency benchmarking (produces Table IV in paper)
  - System info logging
"""

import os
import time
import torch
import torch.nn as nn

DEVICE_GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_CPU = torch.device('cpu')

# ── Module routing maps ────────────────────────────────────────────────────
_GPU_ATTRS = [
    'backbone', 'cross_attention', 'fusion_gate',
    'diagnostic_head', 'prognostic_head',
    'vis_cnn', 'vis_head', 'vis_prog', 'vis_diag',
    'fusion_mlp', 'proj', 'gate', 'diag_head',
    'prog_head', 'cross_attn', 'vis_proj',
]
_CPU_ATTRS = [
    'vib_encoder', 'vibration_encoder', 'encoder',
    'vib_cnn', 'vib_head', 'vib_prog', 'vib_diag',
    'backbone_vib',
]


def deploy(model: nn.Module, verbose: bool = True) -> nn.Module:
    """
    Applies the hardware split to any model in the experiment registry.

    Rules:
      - Any attribute in _GPU_ATTRS → GPU
      - Any attribute in _CPU_ATTRS → CPU
      - Standalone output heads (prog_head, diag_head) → GPU
      - VibrationOnlyBaseline special case: backbone→CPU, heads→GPU

    Safe to call on models that don't have all attributes.

    Args:
        model   : nn.Module instance (any architecture from dsaf_v3.py)
        verbose : Print device assignments if True

    Returns:
        model (in-place, also returned for chaining)
    """
    from src.models.dsaf_v3 import VibrationOnlyBaseline

    for attr in _GPU_ATTRS:
        m = getattr(model, attr, None)
        if m is not None:
            m.to(DEVICE_GPU)
            if verbose:
                _log(f"{attr:<28} → {DEVICE_GPU}")

    for attr in _CPU_ATTRS:
        m = getattr(model, attr, None)
        if m is not None:
            m.to(DEVICE_CPU)
            if verbose:
                _log(f"{attr:<28} → {DEVICE_CPU}")

    # VibrationOnlyBaseline: backbone on CPU, output heads on GPU
    if isinstance(model, VibrationOnlyBaseline):
        if hasattr(model, 'backbone') or hasattr(model, 'encoder'):
            bb = getattr(model, 'backbone', None) or getattr(model, 'encoder')
            bb.to(DEVICE_CPU)
            if verbose: _log(f"{'backbone (vib-only)':<28} → {DEVICE_CPU}")
        for h in ['prog_head', 'diag_head']:
            head = getattr(model, h, None)
            if head is not None:
                head.to(DEVICE_GPU)
                if verbose: _log(f"{h:<28} → {DEVICE_GPU}")

    return model


def _log(msg: str):
    print(f"  [DEPLOY] {msg}")


# ════════════════════════════════════════════════════════════════════════════
# VRAM monitoring
# ════════════════════════════════════════════════════════════════════════════

def get_vram_usage(device: torch.device = DEVICE_GPU) -> dict:
    """
    Returns current and peak VRAM usage in MB.

    Returns:
        dict with keys: allocated_mb, reserved_mb, peak_mb
    """
    if not torch.cuda.is_available():
        return {'allocated_mb': 0, 'reserved_mb': 0, 'peak_mb': 0}

    return {
        'allocated_mb': torch.cuda.memory_allocated(device)  / 1024**2,
        'reserved_mb':  torch.cuda.memory_reserved(device)   / 1024**2,
        'peak_mb':      torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def reset_vram_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def print_vram_status(label: str = ""):
    v = get_vram_usage()
    tag = f"[{label}] " if label else ""
    print(f"  {tag}VRAM → allocated={v['allocated_mb']:.1f}MB  "
          f"reserved={v['reserved_mb']:.1f}MB  "
          f"peak={v['peak_mb']:.1f}MB")


# ════════════════════════════════════════════════════════════════════════════
# Latency benchmarking (Table IV in paper)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def benchmark_latency(model: nn.Module,
                       model_name: str,
                       n_warmup: int = 20,
                       n_iter: int = 500,
                       batch_size: int = 1) -> dict:
    """
    Measures per-frame inference latency and VRAM footprint.

    Args:
        model      : deployed model (after calling deploy())
        model_name : display name for table
        n_warmup   : warm-up iterations (discarded)
        n_iter     : benchmark iterations
        batch_size : inference batch size (use 1 for edge deployment claim)

    Returns dict:
        latency_ms  — average per-frame latency in milliseconds
        fps         — frames per second
        vram_mb     — VRAM allocated during inference (MB)
        n_params_M  — number of parameters in millions
    """
    model.eval()
    reset_vram_peak()

    dummy_img = torch.randn(batch_size, 3, 224, 224).to(DEVICE_GPU)
    dummy_vib = torch.randn(batch_size, 1, 1024).to(DEVICE_CPU)

    # Warm-up
    for _ in range(n_warmup):
        _ = model(dummy_img, dummy_vib)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model(dummy_img, dummy_vib)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / n_iter * 1000 / batch_size

    vram = get_vram_usage()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    result = {
        'model':       model_name,
        'latency_ms':  round(elapsed_ms, 3),
        'fps':         round(1000 / elapsed_ms, 1),
        'vram_mb':     round(vram['allocated_mb'], 2),
        'n_params_M':  round(n_params, 2),
    }
    print(f"  {model_name:<28} "
          f"lat={elapsed_ms:.2f}ms  "
          f"fps={1000/elapsed_ms:.0f}  "
          f"vram={vram['allocated_mb']:.1f}MB  "
          f"params={n_params:.1f}M")
    return result


def run_hardware_table(models: list) -> list:
    """
    Benchmarks a list of (model, name) tuples and returns rows for Table IV.

    Usage:
        rows = run_hardware_table([
            (model_resnet,   'DSAF-ResNet18'),
            (model_effnet,   'DSAF-EfficientNet-B0'),
            (model_convnext, 'DSAF-ConvNeXt-Tiny'),
            (model_swin,     'DSAF-Swin-Tiny'),
        ])
    """
    print(f"\n  Hardware Telemetry Table (RTX 5070 | batch=1)")
    print(f"  {'Model':<28} {'Latency':>10} {'FPS':>8} "
          f"{'VRAM(MB)':>10} {'Params(M)':>10}")
    print("  " + "─" * 72)

    rows = []
    for model, name in models:
        row = benchmark_latency(model, name)
        rows.append(row)
        torch.cuda.empty_cache()

    return rows


# ════════════════════════════════════════════════════════════════════════════
# System info
# ════════════════════════════════════════════════════════════════════════════

def print_system_info():
    """Prints GPU/CPU info at the start of each experiment run."""
    print("\n" + "─" * 55)
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU : {gpu.name}")
        print(f"        {gpu.total_memory / 1024**3:.1f} GB VRAM | "
              f"SM count: {gpu.multi_processor_count}")
    else:
        print("  GPU : Not available — running on CPU")

    import platform
    print(f"  CPU : {platform.processor() or 'AMD Ryzen 5 7500F'}")
    print(f"  PyTorch : {torch.__version__}")
    print("─" * 55 + "\n")
