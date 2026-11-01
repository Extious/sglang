"""Batched AegisKV control channel to one Worker.

Every control message waits for one scheduler iteration, and the server handles
one message at a time, so issuing ops one by one makes the backup pipeline lag
by many iterations under load. A channel keeps one batch in flight per Worker
and coalesces every op issued meanwhile into the next batch.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import aiohttp

from harness.client import ControlError, WorkerDied, control


class ControlChannel:
    def __init__(self, *, session: aiohttp.ClientSession, url: str):
        self._session = session
        self._url = url
        self._pending: List[Tuple[str, Dict[str, Any], asyncio.Future]] = []
        self._wakeup = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def call(self, op: str, args: Dict[str, Any]) -> Dict[str, Any]:
        future = asyncio.get_running_loop().create_future()
        self._pending.append((op, args, future))
        self._wakeup.set()
        return await future

    def close(self) -> None:
        self._task.cancel()
        for _, _, future in self._pending:
            if not future.done():
                future.set_exception(WorkerDied("channel closed"))
        self._pending = []

    async def _run(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            batch, self._pending = self._pending, []
            if not batch:
                continue
            ops = [{"op": op, "args": args} for op, args, _ in batch]
            try:
                res = await control(self._session, self._url, "batch", {"ops": ops})
            except (WorkerDied, ControlError) as e:
                for _, _, future in batch:
                    if not future.done():
                        future.set_exception(type(e)(str(e)))
                continue
            for (_, _, future), item in zip(batch, res["results"]):
                if future.done():
                    continue
                if item["ok"]:
                    future.set_result(item["result"])
                else:
                    future.set_exception(ControlError(item["error"]))
