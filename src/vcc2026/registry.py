from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    run_id: str
    timestamp_utc: str
    git_commit: str
    dataset_version: str
    seed: int
    model_name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    training_time_seconds: float | None = None
    compute: str | None = None


def default_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_record(path: str | Path, record: ExperimentRecord) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

