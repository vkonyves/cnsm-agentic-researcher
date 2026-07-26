from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RunStore:
    def __init__(self, root: Path, run_id: str) -> None:
        self.path = root / run_id
        self.path.mkdir(parents=True, exist_ok=False)

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        path = self.path / "events.jsonl"
        record = {"timestamp": utc_now(), "type": event_type, **payload}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
