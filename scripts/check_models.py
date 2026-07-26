from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing from .env")

    client = OpenAI()
    available = {model.id for model in client.models.list().data}
    requested = [
        os.getenv("OPENAI_WORKER_MODEL", "gpt-5-nano"),
        os.getenv("OPENAI_RESEARCH_MODEL", "gpt-5-mini"),
        os.getenv("OPENAI_LEAD_MODEL", "gpt-5"),
        os.getenv("OPENAI_REVIEW_MODEL", "gpt-5"),
    ]

    print(f"API returned {len(available)} accessible models.")
    for model in dict.fromkeys(requested):
        status = "available" if model in available else "not listed"
        print(f"{model}: {status}")


if __name__ == "__main__":
    main()
