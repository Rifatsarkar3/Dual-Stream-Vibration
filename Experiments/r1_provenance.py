"""
r1_provenance.py — Revision R1: provenance block written into every R1 result file
(analysis plan Section 10). Records what produced a number: the pre-registered plan,
the code, the data fingerprints, the git state, the command line and the software.
"""

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / 'paper' / 'revision_R1' / 'R1_analysis_plan.md'
DATA_LISTS = [ROOT / 'outputs' / 'r1_provenance' / 'data_sha256_v2.txt',
              ROOT / 'outputs' / 'r1_provenance' / 'data_sha256_v2hv.txt']
CODE = ['Experiments/exp16_ress.py', 'Experiments/exp19_baselines.py',
        'Experiments/r1_provenance.py', 'src/dsaf_v2_lite.py', 'src/ablation_models.py',
        'src/published_baselines.py', 'src/sota_baselines.py', 'src/vanilla_baselines.py',
        'src/mini_dann.py', 'src/online_filters.py', 'src/isotonic_eval.py']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def provenance(extra_code=()) -> dict:
    import numpy
    import scipy
    import torch
    code = {}
    for rel in list(CODE) + list(extra_code):
        p = ROOT / rel
        code[rel] = sha256(p) if p.exists() else 'missing'
    return {
        'plan_sha256': sha256(PLAN) if PLAN.exists() else 'missing',
        'code_sha256': code,
        'data_list_sha256': {p.name: (sha256(p) if p.exists() else 'missing') for p in DATA_LISTS},
        'git_head': _git('rev-parse', 'HEAD'),
        'git_tree_differs': bool(_git('status', '--porcelain', '--', 'Experiments', 'src')),
        'command': ' '.join(sys.argv),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'numpy': numpy.__version__,
        'scipy': scipy.__version__,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    }
