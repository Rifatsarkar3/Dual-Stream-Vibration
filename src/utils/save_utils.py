"""
save_utils.py
=============
Centralised auto-save utility for Paper #2.

Every experiment script imports SaveManager and calls .figure() or .csv().
Nothing is ever lost — every output is timestamped and flushed to disk
immediately after generation.

Directory layout created automatically:
    OUTPUTS_ROOT/
        figures/   ← all .png  (300 DPI)
        results/   ← all .csv  + .json
        weights/   ← model checkpoints (managed by train scripts)
        tables/    ← LaTeX table .tex files
"""

import os
import json
import time
import datetime
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family':    'serif',
    'font.size':      10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.dpi':    300,
    'figure.dpi':     120,
})

# ── Default output root (override via env var or constructor) ────────────────
DEFAULT_OUTPUTS = r"E:\Yolo-Thermal\Dual-Stream Vibration-Vision\outputs"


class SaveManager:
    """
    Manages all file output for one experiment run.

    Usage:
        sm = SaveManager(experiment_name='exp01_backbone_comparison')
        sm.figure(fig, 'roc_curves')          # saves fig_roc_curves.png
        sm.csv(rows, 'backbone_results')       # saves backbone_results.csv
        sm.json(data, 'run_metrics')           # saves run_metrics.json
        sm.latex_table(tex_str, 'table_2')    # saves table_2.tex
    """

    def __init__(self, experiment_name: str,
                 outputs_root: str = DEFAULT_OUTPUTS):
        self.exp_name    = experiment_name
        self.root        = outputs_root
        self.fig_dir     = os.path.join(outputs_root, 'figures')
        self.res_dir     = os.path.join(outputs_root, 'results')
        self.weight_dir  = os.path.join(outputs_root, 'weights')
        self.table_dir   = os.path.join(outputs_root, 'tables')
        self.log_dir     = os.path.join(outputs_root, 'logs')

        for d in [self.fig_dir, self.res_dir, self.weight_dir,
                  self.table_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

        self._log_path = os.path.join(
            self.log_dir,
            f"{experiment_name}_{_timestamp()}.log"
        )
        self.log(f"SaveManager initialised | experiment={experiment_name}")

    # ── Figure ───────────────────────────────────────────────────────────────
    def figure(self, fig: plt.Figure, name: str,
               fmt: str = 'png', close: bool = True) -> str:
        """
        Saves a matplotlib figure at 300 DPI.
        name: filename stem (no extension, no spaces)
        Returns the saved path.
        """
        fname = f"{name}.{fmt}"
        path  = os.path.join(self.fig_dir, fname)
        fig.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        if close:
            plt.close(fig)
        self.log(f"  [FIGURE] {path}")
        print(f"  [SAVED] {path}")
        return path

    # ── CSV ──────────────────────────────────────────────────────────────────
    def csv(self, rows: list, name: str,
            header: list = None) -> str:
        """
        Saves a list-of-dicts or list-of-lists as CSV.
        If rows is list-of-dicts, header is auto-inferred.
        """
        path = os.path.join(self.res_dir, f"{name}.csv")
        with open(path, 'w', newline='') as f:
            if rows and isinstance(rows[0], dict):
                fieldnames = header or list(rows[0].keys())
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            else:
                w = csv.writer(f)
                if header:
                    w.writerow(header)
                w.writerows(rows)
        self.log(f"  [CSV] {path} ({len(rows)} rows)")
        print(f"  [SAVED] {path}")
        return path

    # ── JSON ─────────────────────────────────────────────────────────────────
    def json(self, data: dict, name: str) -> str:
        path = os.path.join(self.res_dir, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(_jsonify(data), f, indent=2)
        self.log(f"  [JSON] {path}")
        print(f"  [SAVED] {path}")
        return path

    # ── LaTeX table ──────────────────────────────────────────────────────────
    def latex_table(self, tex_str: str, name: str) -> str:
        path = os.path.join(self.table_dir, f"{name}.tex")
        with open(path, 'w') as f:
            f.write(tex_str)
        self.log(f"  [LATEX] {path}")
        print(f"  [SAVED] {path}")
        return path

    # ── Checkpoint path ──────────────────────────────────────────────────────
    def weight_path(self, model_name: str, tag: str = 'best') -> str:
        d = os.path.join(self.weight_dir, model_name)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{model_name}_{tag}.pth")

    # ── Log ──────────────────────────────────────────────────────────────────
    def log(self, msg: str):
        ts  = datetime.datetime.now().strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        with open(self._log_path, 'a') as f:
            f.write(line + '\n')


# ── Helpers ──────────────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def _jsonify(obj):
    """Recursively convert numpy types to Python natives for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ── Convenience: build LaTeX comparison table ───────────────────────────────

def build_latex_comparison_table(results: dict,
                                  caption: str = "Model performance comparison",
                                  label: str = "tab:comparison") -> str:
    """
    Converts a dict of {model_name: {metric_mean, metric_std, ...}}
    into a ready-to-paste IEEE LaTeX table string.

    Expected keys per model: mae_mean, mae_std, rmse_mean, rmse_std,
                              r2_mean, r2_std, lat_mean, lat_std
    """
    header = (
        r"\begin{table}[!t]" + "\n"
        r"\centering" + "\n"
        r"\caption{" + caption + r"}" + "\n"
        r"\label{" + label + r"}" + "\n"
        r"\begin{tabular}{lcccc}" + "\n"
        r"\hline" + "\n"
        r"Method & MAE $\downarrow$ & RMSE $\downarrow$ & R$^2$ $\uparrow$ & Latency (ms) \\" + "\n"
        r"\hline" + "\n"
    )
    rows = ""
    for name, m in results.items():
        bold_open  = r"\textbf{" if name.startswith("Proposed") else ""
        bold_close = r"}" if name.startswith("Proposed") else ""
        rows += (
            f"{bold_open}{name}{bold_close} & "
            f"${m.get('mae_mean', 0):.4f}\\pm{m.get('mae_std', 0):.4f}$ & "
            f"${m.get('rmse_mean', 0):.4f}\\pm{m.get('rmse_std', 0):.4f}$ & "
            f"${m.get('r2_mean', 0):.4f}\\pm{m.get('r2_std', 0):.4f}$ & "
            f"${m.get('lat_mean', 0):.2f}\\pm{m.get('lat_std', 0):.2f}$ \\\\\n"
        )
    footer = (
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )
    return header + rows + footer
