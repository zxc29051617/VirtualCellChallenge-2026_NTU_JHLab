from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ChallengeManifest:
    season: str
    partition: str
    panel_id: str
    contexts: list[str]
    pert_col: str
    context_col: str
    control_label: str
    n_genes: int
    n_constructs: int
    cells_per_pert: int
    per_context: dict[str, dict[str, int]]


def load_manifest(path: str | Path) -> ChallengeManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ChallengeManifest(
        season=payload["season"],
        partition=payload["partition"],
        panel_id=payload["panel_id"],
        contexts=list(payload["contexts"]),
        pert_col=payload["pert_col"],
        context_col=payload["context_col"],
        control_label=payload["control_label"],
        n_genes=int(payload["n_genes"]),
        n_constructs=int(payload["n_constructs"]),
        cells_per_pert=int(payload["cells_per_pert"]),
        per_context=dict(payload["per_context"]),
    )


def format_manifest(manifest: ChallengeManifest) -> str:
    lines = [
        f"season: {manifest.season}",
        f"partition: {manifest.partition}",
        f"panel: {manifest.panel_id}",
        f"contexts: {', '.join(manifest.contexts)}",
        f"pert_col: {manifest.pert_col}",
        f"context_col: {manifest.context_col}",
        f"control_label: {manifest.control_label}",
        f"n_genes: {manifest.n_genes}",
        f"n_constructs: {manifest.n_constructs}",
        f"cells_per_pert: {manifest.cells_per_pert}",
    ]
    for context in manifest.contexts:
        stats = manifest.per_context[context]
        lines.append(
            f"{context}: perturbations={stats['n_perturbations']}, "
            f"control_cells={stats['control_cells']}, "
            f"ground_truth_cells={stats['ground_truth_cells']}, "
            f"ntc_ids={stats['n_ntc_ids']}"
        )
    return "\n".join(lines)


def expected_prediction_cells(manifest: ChallengeManifest) -> int:
    return len(manifest.contexts) * manifest.n_constructs * manifest.cells_per_pert


def validate_contexts(contexts: Iterable[str], expected: Iterable[str]) -> list[str]:
    actual = list(contexts)
    exp = list(expected)
    missing = [ctx for ctx in exp if ctx not in actual]
    extra = [ctx for ctx in actual if ctx not in exp]
    return missing + extra

