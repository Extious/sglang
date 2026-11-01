"""HTTP calls to one SGLang Worker: streaming generation and AegisKV control."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

import aiohttp


class WorkerDied(Exception):
    """The Worker's connection broke mid-request (fail-stop)."""


class ControlError(Exception):
    pass


async def control(
    session: aiohttp.ClientSession, url: str, op: str, args: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        async with session.post(
            f"{url}/aegiskv/control", json={"op": op, "args": args}
        ) as resp:
            body = await resp.read()
            if resp.status != 200:
                raise ControlError(f"{op} -> {resp.status}: {body[:500]!r}")
            return json.loads(body)
    except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError) as e:
        raise WorkerDied(str(e)) from e


async def generate_stream(
    session: aiohttp.ClientSession,
    url: str,
    *,
    rid: str,
    input_ids: List[int],
    max_new_tokens: int,
    priority: int = 0,
) -> AsyncIterator[Dict[str, Any]]:
    """Yield stream chunks; each carries the cumulative `output_ids`."""
    payload = {
        "rid": rid,
        "input_ids": input_ids,
        "stream": True,
        "priority": priority,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    try:
        async with session.post(f"{url}/generate", json=payload) as resp:
            if resp.status != 200:
                raise WorkerDied(f"generate -> {resp.status}: {await resp.text()}")
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                yield json.loads(data)
        # A SIGKILLed server can close the stream cleanly, without [DONE].
        raise WorkerDied("stream ended without [DONE]")
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientPayloadError,
    ) as e:
        raise WorkerDied(str(e)) from e
