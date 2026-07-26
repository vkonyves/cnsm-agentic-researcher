from __future__ import annotations

import os

import httpx
from agents import set_default_openai_client
from openai import AsyncOpenAI


def configure_openai_client() -> None:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    timeout = httpx.Timeout(
        timeout=300.0,
        connect=20.0,
        read=300.0,
        write=60.0,
        pool=20.0,
    )

    client = AsyncOpenAI(
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )

    set_default_openai_client(client)
