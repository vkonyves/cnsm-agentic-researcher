from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cnsm_agentic.config import load_config
from cnsm_agentic.planner import generate_pilot_plan
from cnsm_agentic.provenance import RunStore, sha256_text
from cnsm_agentic.settings import load_settings


def make_run_id(run_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in run_name)
    return f"{timestamp}-{safe_name}"


async def run_plan(config_path: Path) -> int:
    settings = load_settings()
    if not settings.api_key_present:
        print("OPENAI_API_KEY is missing. Add it to .env.", file=sys.stderr)
        return 2

    config_text = config_path.read_text(encoding="utf-8")
    config = load_config(config_path)
    model = settings.worker_model

    run_id = make_run_id(config.run_name)
    store = RunStore(settings.artifact_root, run_id)
    store.append_event(
        "run_started",
        {
            "config_path": str(config_path),
            "config_sha256": sha256_text(config_text),
            "model": model,
            "experiment_family": config.experiment_family,
        },
    )
    store.write_json("config.snapshot.json", config.model_dump())

    try:
        plan = await generate_pilot_plan(
            config=config,
            model=model,
            max_turns=settings.max_agent_turns,
        )
    except Exception as exc:
        store.append_event(
            "run_failed",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise

    output_path = store.write_json("pilot_plan.json", plan.model_dump())
    store.append_event("run_completed", {"output_path": str(output_path)})

    print(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))
    print(f"\nSaved run: {store.path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CNSM agentic researcher starter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Generate a structured pilot plan")
    plan_parser.add_argument("--config", type=Path, required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan":
        raise SystemExit(asyncio.run(run_plan(args.config)))


if __name__ == "__main__":
    main()
