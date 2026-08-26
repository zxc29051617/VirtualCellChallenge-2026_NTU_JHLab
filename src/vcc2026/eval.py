from __future__ import annotations

import subprocess
from pathlib import Path


def run_cell_eval(
    *,
    predicted_h5ad: str | Path,
    real_h5ad: str | Path,
    outdir: str | Path = "artifacts/cell_eval",
    profile: str = "vcc",
    control_pert: str = "non-targeting",
    pert_col: str = "target_gene",
    allow_discrete: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        "uv",
        "tool",
        "run",
        "cell-eval",
        "run",
        "-ap",
        str(predicted_h5ad),
        "-ar",
        str(real_h5ad),
        "--profile",
        profile,
        "--control-pert",
        control_pert,
        "--pert-col",
        pert_col,
        "-o",
        str(outdir),
    ]
    if allow_discrete:
        command.append("--allow-discrete")
    return subprocess.run(command, check=True, text=True, capture_output=True)

