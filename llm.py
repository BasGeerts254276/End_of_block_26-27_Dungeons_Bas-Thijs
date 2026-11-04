"""Thin wrapper around the Anthropic Messages API.

The contract of this file: a call never propagates an exception to the rest
of the program. If anything fails it returns None and the caller falls back
to its scripted policy. That is what keeps the demo standing.
"""
from __future__ import annotations

import json
import os
import time

import requests

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# Haiku is fast and cheap and fine for iterating. Switch to a larger model
# for the final measurements and report both.
DEFAULT_MODEL = os.environ.get("WARBAND_MODEL", "claude-haiku-4-5-20251001")

TIMEOUT = 20
MAX_RETRIES = 2


class LLMUnavailable(RuntimeError):
    pass


def have_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def call_tools(
    system: str,
    user: str,
    tools: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 400,
    force_tool: bool = True,
) -> tuple[str, dict] | None:
    """Ask the model for exactly one tool call.

    Returns (tool_name, args), or None if anything goes wrong.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "tools": tools,
    }
    if force_tool:
        payload["tool_choice"] = {"type": "any"}

    headers = {
        "x-api-key": key,
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                API_URL, headers=headers, data=json.dumps(payload),
                timeout=TIMEOUT,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code != 200:
                return None
            data = resp.json()
        except (requests.RequestException, ValueError):
            time.sleep(1.0 * (attempt + 1))
            continue

        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                return block.get("name", ""), block.get("input", {}) or {}
        return None
    return None
