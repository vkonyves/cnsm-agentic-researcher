from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JsonPayload = dict[str, Any]


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


class RunStore:
    """Store artefacts and provenance events for one experiment run."""

    def __init__(
        self,
        root: Path,
        run_id: str,
    ) -> None:
        self.root = root
        self.run_id = run_id
        self.path = root / run_id
        self.events_path = self.path / "events.jsonl"

        self.path.mkdir(
            parents=True,
            exist_ok=False,
        )

    def resolve_path(
        self,
        relative_path: str | Path,
    ) -> Path:
        """Resolve a path inside the run directory."""
        path = self.path / relative_path
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        return path

    def write_text(
        self,
        relative_path: str | Path,
        text: str,
    ) -> Path:
        """Write UTF-8 text into the run directory."""
        path = self.resolve_path(relative_path)

        path.write_text(
            text,
            encoding="utf-8",
        )

        return path

    def write_json(
        self,
        relative_path: str | Path,
        payload: Any,
    ) -> Path:
        """Write a JSON artefact into the run directory."""
        path = self.resolve_path(relative_path)

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        return path

    def append_jsonl(
        self,
        relative_path: str | Path,
        payload: Any,
    ) -> Path:
        """Append one JSON object to a JSONL file."""
        path = self.resolve_path(relative_path)

        with path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
            handle.flush()

        return path

    def append_event(
        self,
        event_type: str,
        payload: JsonPayload | None = None,
        **fields: Any,
    ) -> Path:
        """
        Append a timestamped event to events.jsonl.

        Both forms are supported:

            store.append_event(
                "stage_started",
                {"stage": "discovery"},
            )

        and:

            store.append_event(
                "stage_started",
                stage="discovery",
            )
        """
        event_payload: JsonPayload = {}

        if payload is not None:
            event_payload.update(payload)

        event_payload.update(fields)

        record = {
            "timestamp": utc_now(),
            "type": event_type,
            **event_payload,
        }

        return self.append_jsonl(
            "events.jsonl",
            record,
        )