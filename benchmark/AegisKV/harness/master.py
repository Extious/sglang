"""Router, checkpoint coordinator and takeover for a set of TP=1 Workers.

The Master runs every Agent call: it picks the execution Worker and a distinct
retry Worker (the holder of the call's backups), streams tokens, asks the policy
which backups to keep, and on a fail-stop failure imports the latest committed
backup on the holder and resubmits the call there with its accepted tokens.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp
import msgspec
from harness import workload
from harness.channel import ControlChannel
from harness.client import ControlError, WorkerDied, generate_stream
from harness.cluster import Cluster, Worker
from harness.config import ExperimentConfig

logger = logging.getLogger(__name__)

NUM_LAYERS = 32


class SlotPool:
    """First-fit allocator over arena slots `[base, base + size)`."""

    def __init__(self, *, base: int, size: int):
        self.size = size
        self.used = 0
        self._free: List[List[int]] = [[base, size]]

    def alloc(self, n: int) -> Optional[int]:
        for i, (start, length) in enumerate(self._free):
            if length >= n:
                if length == n:
                    self._free.pop(i)
                else:
                    self._free[i] = [start + n, length - n]
                self.used += n
                return start
        return None

    def free(self, start: int, n: int) -> None:
        self.used -= n
        self._free.append([start, n])
        self._free.sort()
        merged: List[List[int]] = []
        for s, l in self._free:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1][1] += l
            else:
                merged.append([s, l])
        self._free = merged

    @property
    def available(self) -> int:
        return self.size - self.used


class Protection(msgspec.Struct, kw_only=True):
    holder: int
    arena: str
    tag: str
    # Peer-DRAM checkpoint: all layers, tokens [0, max_tokens).
    dram_offset: int = -1
    dram_capacity: int = 0
    max_tokens: int = 0
    dram_committed: int = 0
    # Peer-HBM region of the first `region_layers` layers, fed from a source extent.
    region_id: str = ""
    region_layers: int = 0
    region_slots: int = 0
    region_committed: int = 0
    region_requested: int = 0
    staging_offset: int = -1
    staging_capacity: int = 0
    staging_committed: int = 0

    @property
    def has_dram(self) -> bool:
        return self.dram_offset >= 0

    @property
    def has_region(self) -> bool:
        return bool(self.region_id)


class Call(msgspec.Struct, kw_only=True):
    key: str
    workflow: int
    role: str
    prompt: List[int]
    max_new: int
    output: List[int] = []
    exec_w: int = -1
    holder: int = -1
    attempt: int = 0
    t_submit: float = 0.0
    t_first: float = 0.0
    t_done: float = 0.0
    submit_stats: Dict[str, int] = {}
    prot: Optional[Protection] = None
    failures: List[Dict[str, Any]] = []

    @property
    def total_len(self) -> int:
        return len(self.prompt) + self.max_new

    @property
    def history_len(self) -> int:
        return len(self.prompt) + len(self.output)


class Master:
    def __init__(self, *, cfg: ExperimentConfig, cluster: Cluster, policy: Any):
        self.cfg = cfg
        self.cluster = cluster
        self.policy = policy
        self.session: Optional[aiohttp.ClientSession] = None
        self.t0 = 0.0
        self.calls: Dict[str, Call] = {}
        self.running: Dict[str, Call] = {}
        self.workflow_times: Dict[int, List[float]] = {}
        self.stats: Dict[int, Dict[str, int]] = {}
        self.dram_pools: Dict[int, SlotPool] = {}
        self.staging_pools: Dict[int, SlotPool] = {}
        self.region_slots_used: Dict[int, int] = {}
        self._target_owner: Dict[str, tuple] = {}
        self._stage_owner: Dict[str, tuple] = {}
        self._import_waiters: Dict[str, asyncio.Future] = {}
        self._stopped = False
        self._bg_tasks: set = set()
        self._channels: Dict[str, ControlChannel] = {}
        self._seq = 0
        self.fail_time: Optional[float] = None
        os.makedirs(cfg.out_dir, exist_ok=True)
        self._events = open(os.path.join(cfg.out_dir, "events.jsonl"), "w")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def now(self) -> float:
        return time.monotonic() - self.t0

    def log_event(self, kind: str, **fields: Any) -> None:
        self._events.write(json.dumps({"t": self.now(), "kind": kind, **fields}) + "\n")

    def alive(self, w: int) -> bool:
        return w >= 0 and self.cluster.workers[w].alive

    def _reset_pools(self, w: int) -> None:
        cfg = self.cfg
        self.dram_pools[w] = SlotPool(base=0, size=cfg.dram_backup_tokens)
        self.staging_pools[w] = SlotPool(
            base=cfg.dram_backup_tokens, size=cfg.staging_tokens
        )
        self.region_slots_used[w] = 0

    async def run(self) -> None:
        # No keep-alive: a restarted Worker reuses its port, and pooled
        # connections to the dead process would fail on reuse.
        connector = aiohttp.TCPConnector(limit=0, force_close=True)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as s:
            self.session = s
            for w in range(len(self.cluster.workers)):
                self._reset_pools(w)
                url = self.cluster.workers[w].url
                await self._rpc(url, "reset", {})
                # Runs reuse the same prompts; a warm radix cache would leak hits.
                async with s.post(f"{url}/flush_cache") as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"flush_cache on {url}: {await resp.text()}")
            await self._map_arenas()
            self.t0 = time.monotonic()
            poller = asyncio.create_task(self._poll_loop())
            injector = asyncio.create_task(self._failure_injector())
            try:
                await self._drive_workflows()
            finally:
                self._stopped = True
                injector.cancel()
                await asyncio.gather(poller, injector, return_exceptions=True)
                await asyncio.gather(*self._bg_tasks, return_exceptions=True)
                for url in list(self._channels):
                    self._reset_channel(url)
        self._write_results()

    # ------------------------------------------------------------------ #
    # Workload
    # ------------------------------------------------------------------ #

    async def _drive_workflows(self) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        for wf in range(self.cfg.num_workflows):
            queue.put_nowait(wf)

        async def lane(lane_id: int) -> None:
            await asyncio.sleep(lane_id * self.cfg.stagger_s)
            while not queue.empty():
                await self._run_workflow(queue.get_nowait())

        await asyncio.gather(*(lane(i) for i in range(self.cfg.concurrency)))

    async def _run_workflow(self, wf: int) -> None:
        start = self.now()
        outputs: Dict[str, List[int]] = {}
        for stage in workload.STAGES:
            calls = [self._new_call(wf, role, outputs) for role in stage]
            await asyncio.gather(*(self._run_call(c) for c in calls))
            for c in calls:
                outputs[c.role] = c.output
        self.workflow_times[wf] = [start, self.now()]
        self.log_event("workflow_done", workflow=wf, start=start, end=self.now())

    def _new_call(self, wf: int, role: str, outputs: Dict[str, List[int]]) -> Call:
        own = workload.own_history(
            seed=self.cfg.seed, workflow=wf, role=role, length=self.cfg.input_len
        )
        call = Call(
            key=f"wf{wf}-{role}",
            workflow=wf,
            role=role,
            prompt=workload.build_prompt(own=own, role=role, outputs=outputs),
            max_new=self.cfg.output_lens[role],
        )
        self.calls[call.key] = call
        return call

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def _load(self, w: int) -> int:
        return sum(1 for c in self.running.values() if c.exec_w == w)

    def pick_exec(self) -> int:
        alive = [w.index for w in self.cluster.workers if w.alive]
        return min(alive, key=lambda w: (self._load(w), w))

    def pick_holder(self, exec_w: int) -> int:
        """A live Worker other than `exec_w`, or -1 when none is left."""
        alive = [w.index for w in self.cluster.workers if w.alive and w.index != exec_w]
        if not alive:
            return -1
        return min(
            alive,
            key=lambda w: (
                self.dram_pools[w].used + self.region_slots_used[w],
                self._load(w),
                w,
            ),
        )

    # ------------------------------------------------------------------ #
    # Call execution and takeover
    # ------------------------------------------------------------------ #

    async def _run_call(self, call: Call) -> None:
        call.exec_w = self.pick_exec()
        call.holder = self.pick_holder(call.exec_w)
        call.t_submit = self.now()
        call.submit_stats = dict(self.stats.get(call.exec_w, {}))
        self.running[call.key] = call
        self.log_event(
            "call_submit", call=call.key, exec=call.exec_w, holder=call.holder
        )
        transient = 0
        while True:
            try:
                await self._stream(call)
                break
            except WorkerDied as e:
                if not self.alive(call.exec_w):
                    await self._recover(call)
                    continue
                # A live Worker's connection broke; resubmit the accepted tokens.
                transient += 1
                if transient > 3:
                    raise RuntimeError(f"{call.key}: live worker broke: {e}") from e
                self.log_event("transient_error", call=call.key, error=str(e))
                await self.release_protection(call)
                call.attempt += 1
        call.t_done = self.now()
        self.running.pop(call.key, None)
        await self.release_protection(call)
        self.log_event("call_done", call=call.key)

    async def _stream(self, call: Call) -> None:
        remaining = call.max_new - len(call.output)
        if remaining <= 0:
            return
        base = list(call.output)
        worker = self.cluster.workers[call.exec_w]
        async for chunk in generate_stream(
            self.session,
            worker.url,
            rid=f"{call.key}#a{call.attempt}",
            input_ids=call.prompt + base,
            max_new_tokens=remaining,
            # Takeover resumptions jump the retry Worker's queue (no preemption).
            priority=self.policy.resume_priority(call) if call.attempt > 0 else 0,
        ):
            new = chunk.get("output_ids") or []
            if not new:
                continue
            call.output = base + list(new)
            if not call.t_first:
                call.t_first = self.now()
            if call.failures and "t_resume_first" not in call.failures[-1]:
                record = call.failures[-1]
                record["t_resume_first"] = self.now()
                record["cached_tokens"] = chunk["meta_info"].get("cached_tokens", 0)
                if record.get("import_id"):
                    await self._release_import(call.exec_w, record["import_id"])

    async def _recover(self, call: Call) -> None:
        t_detect = self.now()
        record: Dict[str, Any] = {
            "t_fail": self.fail_time,
            "t_detect": t_detect,
            "failed_worker": call.exec_w,
            "accepted_tokens": call.history_len,
            "plan": self.policy.describe(call),
        }
        call.failures.append(record)
        target = call.holder if self.alive(call.holder) else self.pick_exec()
        restored = 0
        if call.prot is not None and call.holder == target:
            restored = await self._import_backup(call, record)
        await self.release_protection(call)
        call.attempt += 1
        call.exec_w = target
        call.holder = self.pick_holder(target)
        record.update(restored_tokens=restored, takeover=target)
        self.log_event("takeover", call=call.key, **record)

    async def _import_backup(self, call: Call, record: Dict[str, Any]) -> int:
        prot = call.prot
        seq = call.prompt + call.output
        args, usable = self._import_sources(prot)
        usable = min(usable, len(seq))
        if usable <= 0:
            return 0
        self._seq += 1
        import_id = f"{call.key}#imp{self._seq}"
        args.update(import_id=import_id, token_ids=seq[:usable])
        holder_url = self.cluster.workers[prot.holder].url
        future = asyncio.get_running_loop().create_future()
        self._import_waiters[import_id] = future
        record["source"] = "+".join(k for k in ("region_id", "arena") if k in args)
        try:
            ack = await self._rpc(holder_url, "import", args)
        except (ControlError, WorkerDied) as e:
            record["import_error"] = repr(e)
            return 0
        if not ack.get("accepted"):
            record["import_error"] = "rejected"
            return 0
        if prot.has_region and "region_id" in args:
            # The holder adopts or frees the region as part of the import.
            self.region_slots_used[prot.holder] -= prot.region_slots
            prot.region_id = ""
        state = ack.get("state")
        if state is None or state["status"] == "copying":
            state = await future
        self._import_waiters.pop(import_id, None)
        record.update(
            import_id=import_id,
            import_status=state["status"],
            prefix_hit=state.get("prefix_hit", 0),
            import_copy_s=state.get("copy_seconds", 0.0),
            t_import_done=self.now(),
        )
        return usable if state["status"] == "done" else 0

    def _import_sources(self, prot: Protection) -> tuple:
        arena = lambda lo, off: {  # noqa: E731
            "path": prot.arena,
            "slot_offset": off,
            "layer_lo": lo,
            "layer_hi": NUM_LAYERS,
        }
        if prot.has_region and prot.region_committed > 0:
            if prot.region_layers == NUM_LAYERS:
                return {"region_id": prot.region_id}, prot.region_committed
            if prot.has_dram and prot.dram_committed > 0:
                usable = min(prot.region_committed, prot.dram_committed)
                return {
                    "region_id": prot.region_id,
                    "arena": arena(prot.region_layers, prot.dram_offset),
                }, usable
        if prot.has_dram and prot.dram_committed > 0:
            return {"arena": arena(0, prot.dram_offset)}, prot.dram_committed
        return {}, 0

    async def _release_import(self, w: int, import_id: str) -> None:
        try:
            await self._rpc(
                self.cluster.workers[w].url,
                "release_import",
                {"import_id": import_id},
            )
        except (ControlError, WorkerDied):
            pass

    # ------------------------------------------------------------------ #
    # Protection primitives used by policies
    # ------------------------------------------------------------------ #

    def _new_protection(self, call: Call) -> Protection:
        holder = self.cluster.workers[call.holder]
        return Protection(
            holder=call.holder,
            arena=holder.arena_path,
            tag=f"{call.key}#a{call.attempt}",
        )

    async def _send_protect(
        self,
        call: Call,
        target_id: str,
        offset: int,
        capacity: int,
        layers: int,
        max_tokens: int,
    ) -> bool:
        spec = {
            "target_id": target_id,
            "arena": call.prot.arena,
            "slot_offset": offset,
            "capacity": capacity,
            "layer_lo": 0,
            "layer_hi": layers,
            "max_tokens": max_tokens,
        }
        try:
            await self._rpc(
                self.cluster.workers[call.exec_w].url,
                "protect",
                {"rid": f"{call.key}#a{call.attempt}", "targets": [spec]},
            )
        except (ControlError, WorkerDied):
            return False
        return True

    async def protect_dram(self, call: Call, max_tokens: int) -> bool:
        """Checkpoint tokens [0, max_tokens) of all layers into the holder's DRAM."""
        if not (self.alive(call.holder) and self.alive(call.exec_w)):
            return False
        max_tokens = min(max_tokens, call.total_len)
        call.prot = call.prot or self._new_protection(call)
        prot = call.prot
        if prot.has_dram and prot.dram_capacity >= max_tokens:
            if max_tokens != prot.max_tokens:
                prot.max_tokens = max_tokens
                await self._send_protect(
                    call,
                    f"{prot.tag}#d",
                    prot.dram_offset,
                    prot.dram_capacity,
                    NUM_LAYERS,
                    max_tokens,
                )
            return True
        if prot.has_dram:
            await self.unprotect_dram(call)
        offset = self.dram_pools[call.holder].alloc(max_tokens)
        if offset is None:
            return False
        prot.dram_offset, prot.dram_capacity = offset, max_tokens
        prot.max_tokens, prot.dram_committed = max_tokens, 0
        self._target_owner[f"{prot.tag}#d"] = (call.key, "dram")
        ok = await self._send_protect(
            call, f"{prot.tag}#d", offset, max_tokens, NUM_LAYERS, max_tokens
        )
        if not ok:
            await self.unprotect_dram(call)
        return ok

    async def unprotect_dram(self, call: Call) -> None:
        prot = call.prot
        if prot is None or not prot.has_dram:
            return
        await self._unprotect_targets(call, [f"{prot.tag}#d"])
        if self.cluster.workers[prot.holder].arena_path == prot.arena:
            self.dram_pools[prot.holder].free(prot.dram_offset, prot.dram_capacity)
        prot.dram_offset, prot.dram_capacity = -1, 0
        prot.max_tokens = prot.dram_committed = 0

    async def add_region(self, call: Call, layers: int) -> bool:
        """Replicate the first `layers` layers of the whole call into the holder's HBM."""
        if not (self.alive(call.holder) and self.alive(call.exec_w)):
            return False
        call.prot = call.prot or self._new_protection(call)
        prot = call.prot
        if prot.has_region:
            return True
        holder_url = self.cluster.workers[call.holder].url
        self._seq += 1
        region_id = f"{prot.tag}#r{self._seq}"
        # Count the slots before the RPC so concurrent admissions see them.
        estimate = -(-layers * call.total_len // NUM_LAYERS)
        holder = call.holder
        self.region_slots_used[holder] += estimate
        prot.region_id = region_id
        try:
            res = await self._rpc(
                holder_url,
                "reserve_region",
                {
                    "region_id": region_id,
                    "token_capacity": call.total_len,
                    "resident_layers": layers,
                },
            )
        except (ControlError, WorkerDied):
            res = {}
        if (
            not res.get("reserved")
            or call.prot is not prot
            or prot.region_id != region_id
        ):
            self.region_slots_used[holder] -= estimate
            prot.region_id = ""
            if res.get("reserved"):
                self._spawn(
                    self._rpc(holder_url, "free_region", {"region_id": region_id})
                )
            return False
        prot.region_layers, prot.region_slots = layers, res["num_slots"]
        prot.region_committed = prot.region_requested = 0
        if not (prot.has_dram and prot.max_tokens >= call.total_len):
            offset = self.staging_pools[call.holder].alloc(call.total_len)
            if offset is None:
                await self.drop_region(call)
                return False
            prot.staging_offset, prot.staging_capacity = offset, call.total_len
            prot.staging_committed = 0
            self._target_owner[f"{prot.tag}#s"] = (call.key, "staging")
            await self._send_protect(
                call, f"{prot.tag}#s", offset, call.total_len, layers, call.total_len
            )
        return True

    async def drop_region(self, call: Call) -> None:
        prot = call.prot
        if prot is None or not prot.has_region:
            return
        if self.alive(prot.holder):
            try:
                await self._rpc(
                    self.cluster.workers[prot.holder].url,
                    "free_region",
                    {"region_id": prot.region_id},
                )
            except (ControlError, WorkerDied):
                pass
            self.region_slots_used[prot.holder] -= prot.region_slots
        prot.region_id, prot.region_layers, prot.region_slots = "", 0, 0
        prot.region_committed = prot.region_requested = 0
        if prot.staging_offset >= 0:
            await self._unprotect_targets(call, [f"{prot.tag}#s"])
            if self.cluster.workers[prot.holder].arena_path == prot.arena:
                self.staging_pools[prot.holder].free(
                    prot.staging_offset, prot.staging_capacity
                )
            prot.staging_offset, prot.staging_capacity = -1, 0

    async def release_protection(self, call: Call) -> None:
        """Detach the call's backups now; Worker-side frees run in the background."""
        prot, call.prot = call.prot, None
        if prot is not None:
            self._spawn(self._release_backup(prot, call.exec_w))

    async def _release_backup(self, prot: Protection, exec_w: int) -> None:
        target_ids = [f"{prot.tag}#d", f"{prot.tag}#s"]
        for tid in target_ids:
            self._target_owner.pop(tid, None)
        holder_url = self.cluster.workers[prot.holder].url
        try:
            if self.alive(exec_w):
                await self._rpc(
                    self.cluster.workers[exec_w].url,
                    "unprotect",
                    {"target_ids": target_ids},
                )
            if prot.has_region and self.alive(prot.holder):
                await self._rpc(
                    holder_url, "free_region", {"region_id": prot.region_id}
                )
        except (ControlError, WorkerDied):
            pass
        if prot.has_region and self.alive(prot.holder):
            self.region_slots_used[prot.holder] -= prot.region_slots
        # Arena slots return to the pool only after the source stopped writing them.
        if self.cluster.workers[prot.holder].arena_path == prot.arena:
            if prot.has_dram:
                self.dram_pools[prot.holder].free(prot.dram_offset, prot.dram_capacity)
            if prot.staging_offset >= 0:
                self.staging_pools[prot.holder].free(
                    prot.staging_offset, prot.staging_capacity
                )

    async def _rpc(self, url: str, op: str, args: Dict[str, Any]) -> Dict[str, Any]:
        channel = self._channels.get(url)
        if channel is None:
            channel = ControlChannel(session=self.session, url=url)
            self._channels[url] = channel
        return await channel.call(op, args)

    def _reset_channel(self, url: str) -> None:
        channel = self._channels.pop(url, None)
        if channel is not None:
            channel.close()

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _unprotect_targets(self, call: Call, target_ids: List[str]) -> None:
        for tid in target_ids:
            self._target_owner.pop(tid, None)
        if self.alive(call.exec_w):
            try:
                await self._rpc(
                    self.cluster.workers[call.exec_w].url,
                    "unprotect",
                    {"target_ids": target_ids},
                )
            except (ControlError, WorkerDied):
                pass

    # ------------------------------------------------------------------ #
    # Polling: commit frontiers, region staging, imports, policy tick
    # ------------------------------------------------------------------ #

    async def _poll_one(self, worker: Worker) -> None:
        try:
            res = await self._rpc(worker.url, "poll", {})
        except (ControlError, WorkerDied):
            return
        self.stats[worker.index] = res["stats"]
        for tid, (committed, _seen, _ended) in res["targets"].items():
            owner = self._target_owner.get(tid)
            call = self.running.get(owner[0]) if owner else None
            if call is None or call.prot is None:
                continue
            if owner[1] == "dram":
                call.prot.dram_committed = committed
            else:
                call.prot.staging_committed = committed
        for task in res["stages"]:
            owner = self._stage_owner.pop(task, None)
            call = self.running.get(owner[0]) if owner else None
            if (
                call is not None
                and call.prot is not None
                and call.prot.region_id == owner[1]
            ):
                call.prot.region_committed = max(call.prot.region_committed, owner[2])
        for import_id, state in res["imports"].items():
            future = self._import_waiters.get(import_id)
            if (
                future is not None
                and state["status"] != "copying"
                and not future.done()
            ):
                future.set_result(state)

    async def _advance_regions(self) -> None:
        await asyncio.gather(
            *(self._advance_region(c) for c in list(self.running.values()))
        )

    async def _advance_region(self, call: Call) -> None:
        prot = call.prot
        if prot is None or not prot.region_layers or not self.alive(prot.holder):
            return
        if prot.staging_offset >= 0:
            src_offset, src_committed = prot.staging_offset, prot.staging_committed
        else:
            src_offset, src_committed = prot.dram_offset, prot.dram_committed
        if src_committed <= prot.region_requested:
            return
        token_lo, prot.region_requested = prot.region_requested, src_committed
        self._seq += 1
        task_id = f"{prot.region_id}#st{self._seq}"
        self._stage_owner[task_id] = (call.key, prot.region_id, src_committed)
        try:
            await self._rpc(
                self.cluster.workers[prot.holder].url,
                "stage_region",
                {
                    "task_id": task_id,
                    "region_id": prot.region_id,
                    "arena": prot.arena,
                    "slot_offset": src_offset,
                    "token_lo": token_lo,
                    "token_hi": src_committed,
                },
            )
        except (ControlError, WorkerDied):
            prot.region_requested = token_lo

    def _log_usage(self) -> None:
        self.log_event(
            "usage",
            dram={w: p.used for w, p in self.dram_pools.items()},
            staging={w: p.used for w, p in self.staging_pools.items()},
            hbm=dict(self.region_slots_used),
            pool={
                w: s.get("total", 0) - s.get("available", 0)
                for w, s in self.stats.items()
            },
        )

    async def _poll_loop(self) -> None:
        last_usage = 0.0
        while not self._stopped:
            if self.now() - last_usage >= 1.0:
                last_usage = self.now()
                self._log_usage()
            alive = [w for w in self.cluster.workers if w.alive]
            await asyncio.gather(*(self._poll_one(w) for w in alive))
            for call in self.running.values():
                if call.prot is None and not self.alive(call.holder):
                    call.holder = self.pick_holder(call.exec_w)
            await self._advance_regions()
            await self.policy.tick(self)
            await asyncio.sleep(self.cfg.poll_interval_s)

    # ------------------------------------------------------------------ #
    # Failure injection
    # ------------------------------------------------------------------ #

    async def _failure_injector(self) -> None:
        cfg = self.cfg
        if cfg.fail_worker is None or cfg.fail_at_s is None:
            return
        await asyncio.sleep(max(0.0, cfg.fail_at_s - self.now()))
        worker = self.cluster.workers[cfg.fail_worker]
        self.fail_time = self.now()
        lost_arena = worker.arena_path
        victims = [c.key for c in self.running.values() if c.exec_w == worker.index]
        self.log_event("failure", worker=worker.index, victims=victims)
        await self.cluster.kill_async(worker)
        await self._drop_lost_backups(worker.index, lost_arena)
        await asyncio.sleep(cfg.rejoin_after_s)
        self.cluster.start(worker)
        await self.cluster.wait_ready(worker)
        self._reset_channel(worker.url)
        await self._rpc(worker.url, "reset", {})
        self._reset_pools(worker.index)
        await self._map_arenas()
        self.log_event("rejoin", worker=worker.index)

    async def _map_arenas(self) -> None:
        paths = [w.arena_path for w in self.cluster.workers if w.alive]
        await asyncio.gather(
            *(
                self._rpc(w.url, "map_arenas", {"paths": paths})
                for w in self.cluster.workers
                if w.alive
            )
        )

    async def _drop_lost_backups(self, w: int, arena: str) -> None:
        for other in self.cluster.workers:
            if other.alive:
                try:
                    await self._rpc(other.url, "drop_arena", {"path": arena})
                except (ControlError, WorkerDied):
                    pass
        for call in self.running.values():
            if call.holder == w and call.exec_w != w:
                if call.prot is not None:
                    for tid in (f"{call.prot.tag}#d", f"{call.prot.tag}#s"):
                        self._target_owner.pop(tid, None)
                call.prot = None
                call.holder = self.pick_holder(call.exec_w)
                self.log_event("holder_lost", call=call.key, new_holder=call.holder)

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #

    def _write_results(self) -> None:
        self._events.close()
        calls = [
            {
                "key": c.key,
                "workflow": c.workflow,
                "role": c.role,
                "prompt_len": len(c.prompt),
                "max_new": c.max_new,
                "output_len": len(c.output),
                "t_submit": c.t_submit,
                "t_first": c.t_first,
                "t_done": c.t_done,
                "submit_stats": c.submit_stats,
                "failures": c.failures,
                "output_hash": hash(tuple(c.output)),
            }
            for c in self.calls.values()
        ]
        result = {
            "config": msgspec.to_builtins(self.cfg),
            "policy": self.policy.name,
            "fail_time": self.fail_time,
            "workflows": self.workflow_times,
            "calls": calls,
        }
        with open(os.path.join(self.cfg.out_dir, "result.json"), "w") as f:
            json.dump(result, f)
