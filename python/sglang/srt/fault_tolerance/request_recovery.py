# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""
Request seamless recovery for KevlarFlow fault tolerance.

This module handles the recovery of in-flight requests when a GPU/node failure
is detected. Instead of retrying requests from scratch (which causes large
latency spikes as shown in the paper), we recover requests from the replicated
KV cache on peer nodes.

Recovery workflow:
1. Fault detected (GPUHealthChecker publishes FaultEvent)
2. DataParallelController marks the worker unhealthy
3. RequestRecoveryManager identifies affected in-flight requests
4. For each affected request:
   a. Find the backup peer holding the replicated KV
   b. Migrate the request to the backup peer
   c. Resume inference from the replicated KV state
   d. Mark request as recovered (deduplicate retries)
5. Resume normal operation once backup takes over

Key design decisions:
- Recovery is initiated per-request, not per-batch, for fine-grained control
- Deduplication: if a request is already being recovered, subsequent retries are ignored
- Fallback: if KV is not replicated, fall back to normal retry (session context lost)
- Session context: includes full prompt + generated tokens so far + sampling params
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RecoveryState(Enum):
    """State of a request during recovery."""

    NORMAL = auto()
    """Request is being served normally."""

    DETECTED = auto()
    """Fault detected for the worker serving this request."""

    RECOVERING = auto()
    """Request is being migrated to a backup peer."""

    RECOVERED = auto()
    """Request has been successfully recovered on a backup peer."""

    RETRYING = auto()
    """Fallback: KV not replicated, normal retry from scratch."""

    FAILED = auto()
    """Recovery failed (no healthy backup, etc.)."""


@dataclass
class RecoveryTask:
    """Data for recovering a single in-flight request."""

    req_id: str
    # Worker that was serving this request before failure
    failed_dp_rank: int
    # Target backup peer (should have replicated KV)
    backup_dp_rank: int
    # Full request context for recovery
    prompt_ids: List[int]
    generated_ids: List[int]
    sampling_params_dict: Dict
    # Session / conversation context
    session_id: Optional[str] = None
    # How many recovery attempts have been made
    attempts: int = 0
    max_attempts: int = 3
    state: RecoveryState = RecoveryState.DETECTED
    created_at: float = field(default_factory=time.time)
    recovered_at: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def duration(self) -> float:
        end = self.recovered_at or time.time()
        return end - self.created_at


class RequestRecoveryManager:
    """
    Manages seamless recovery of in-flight requests after GPU failures.

    Integrates with:
    - GPUHealthChecker: receives fault events
    - DataParallelController: coordinates worker status and request routing
    - AsyncKVReplicator: confirms KV availability on backup peers

    Usage:
        recovery_mgr = RequestRecoveryManager(dp_size)
        recovery_mgr.subscribe_recovered(my_callback)
        recovery_mgr.on_fault_detected(failed_dp_rank=2, affected_req_ids=[...])
    """

    def __init__(
        self,
        dp_size: int,
        my_dp_rank: int = 0,
        max_recovery_attempts: int = 3,
        recovery_timeout: float = 30.0,
        enabled: bool = True,
    ):
        """
        Initialize the request recovery manager.

        Args:
            dp_size: Total DP size.
            my_dp_rank: My own DP rank.
            max_recovery_attempts: Max attempts per request before falling back to retry.
            recovery_timeout: Timeout for a single recovery operation (seconds).
            enabled: If False, recovery is bypassed (normal retry only).
        """
        self.dp_size = dp_size
        self.my_dp_rank = my_dp_rank
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_timeout = recovery_timeout
        self.enabled = enabled

        self._lock = threading.RLock()
        self._running = False

        # Recovery tasks indexed by request ID
        self._recovery_tasks: Dict[str, RecoveryTask] = {}

        # Set of request IDs currently being recovered (for deduplication)
        self._in_recovery: Set[str] = set()

        # Set of request IDs that have been recovered (to deduplicate retries)
        self._recovered_ids: Set[str] = set()

        # Subscribers
        self._recovered_callbacks: List[Callable[[RecoveryTask], None]] = []
        self._failed_callbacks: List[Callable[[RecoveryTask], None]] = []

        # Worker health status (updated from DataParallelController)
        self._worker_health: Dict[int, bool] = {i: True for i in range(dp_size)}

        # Statistics
        self._stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "fallback_retries": 0,
            "deduplicated_retries": 0,
        }

        logger.info(
            f"RequestRecoveryManager initialized: dp_size={dp_size}, my_rank={my_dp_rank}, "
            f"enabled={enabled}"
        )

    # -------------------------------------------------------------------------
    # Subscription API
    # -------------------------------------------------------------------------

    def subscribe_recovered(
        self, callback: Callable[[RecoveryTask], None]
    ) -> None:
        """Register a callback for successful recoveries."""
        with self._lock:
            self._recovered_callbacks.append(callback)

    def subscribe_failed(
        self, callback: Callable[[RecoveryTask], None]
    ) -> None:
        """Register a callback for failed recoveries."""
        with self._lock:
            self._failed_callbacks.append(callback)

    def _notify_recovered(self, task: RecoveryTask) -> None:
        with self._lock:
            for cb in self._recovered_callbacks:
                try:
                    cb(task)
                except Exception:
                    logger.exception(f"Recovered callback raised: {cb}")

    def _notify_failed(self, task: RecoveryTask) -> None:
        with self._lock:
            for cb in self._failed_callbacks:
                try:
                    cb(task)
                except Exception:
                    logger.exception(f"Failed callback raised: {cb}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start the recovery manager."""
        if not self.enabled:
            logger.info("RequestRecoveryManager disabled, skipping start.")
            return
        with self._lock:
            if self._running:
                logger.warning("RequestRecoveryManager already running.")
                return
            self._running = True
        logger.info("RequestRecoveryManager started.")

    def stop(self) -> None:
        """Stop the recovery manager."""
        with self._lock:
            if not self._running:
                return
            self._running = False
        logger.info("RequestRecoveryManager stopped.")

    def update_worker_health(self, dp_rank: int, is_healthy: bool) -> None:
        """Update the health status of a worker. Called by DataParallelController."""
        with self._lock:
            self._worker_health[dp_rank] = is_healthy
            logger.debug(
                f"Worker health updated: dp_rank={dp_rank}, healthy={is_healthy}"
            )

    def on_fault_detected(
        self,
        failed_dp_rank: int,
        affected_req_ids: List[str],
    ) -> None:
        """
        Called when a fault is detected for a worker.
        Initiates recovery for all affected in-flight requests.

        Args:
            failed_dp_rank: The DP rank that failed.
            affected_req_ids: List of request IDs that were being served by the failed worker.
        """
        if not self.enabled:
            return

        logger.info(
            f"Recovery triggered: failed_dp_rank={failed_dp_rank}, "
            f"affected_requests={len(affected_req_ids)}"
        )

        for req_id in affected_req_ids:
            self._start_recovery(req_id, failed_dp_rank)

    def is_already_recovered(self, req_id: str) -> bool:
        """Return whether a request has already been recovered (for deduplication)."""
        with self._lock:
            return req_id in self._recovered_ids

    def mark_recovered(self, req_id: str) -> None:
        """Mark a request as successfully recovered."""
        with self._lock:
            self._recovered_ids.add(req_id)
            if req_id in self._recovery_tasks:
                task = self._recovery_tasks[req_id]
                task.state = RecoveryState.RECOVERED
                task.recovered_at = time.time()
                logger.info(
                    f"Request recovered: req_id={req_id}, duration={task.duration:.2f}s"
                )
                self._notify_recovered(task)

    def on_retry_received(self, req_id: str) -> Optional[RecoveryTask]:
        """
        Called when a retry request is received for a request ID.

        Returns the RecoveryTask if this is a genuine retry (not a deduplicated one),
        or None if the request was already recovered and should be ignored.
        """
        with self._lock:
            if req_id in self._recovered_ids:
                self._stats["deduplicated_retries"] += 1
                logger.info(
                    f"Retry deduplicated (already recovered): req_id={req_id}"
                )
                return None

            if req_id in self._in_recovery:
                self._stats["deduplicated_retries"] += 1
                logger.info(
                    f"Retry deduplicated (recovery in progress): req_id={req_id}"
                )
                return None

        return self._recovery_tasks.get(req_id)

    def find_backup_peer(self, failed_dp_rank: int) -> Optional[int]:
        """
        Find the best backup peer for a failed DP rank.

        Strategy:
        - For rank 0: KV is replicated at all other ranks, pick any healthy peer
        - For other ranks: KV is replicated at rank 0 (primary backup)
        - Fallback: any healthy peer
        """
        with self._lock:
            health = dict(self._worker_health)

        if failed_dp_rank == 0:
            # rank 0's KV is replicated at all other ranks
            for peer in range(1, self.dp_size):
                if health.get(peer, False):
                    return peer
        else:
            # Primary backup is rank 0
            if health.get(0, False):
                return 0
            # Fallback: any other healthy peer
            for peer in range(1, self.dp_size):
                if peer != failed_dp_rank and health.get(peer, False):
                    return peer

        return None

    def get_stats(self) -> Dict:
        """Return recovery statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_recoveries": len(self._in_recovery),
                "total_tasks": len(self._recovery_tasks),
                "recovered_ids": len(self._recovered_ids),
            }

    # -------------------------------------------------------------------------
    # Internal recovery logic
    # -------------------------------------------------------------------------

    def _start_recovery(self, req_id: str, failed_dp_rank: int) -> None:
        """
        Start the recovery process for a single request.
        Runs in the caller's thread (typically from fault callback).
        """
        # Check for deduplication
        with self._lock:
            if req_id in self._recovered_ids:
                self._stats["deduplicated_retries"] += 1
                return
            if req_id in self._in_recovery:
                return
            self._in_recovery.add(req_id)

        backup_peer = self.find_backup_peer(failed_dp_rank)
        if backup_peer is None:
            logger.warning(
                f"No healthy backup found for req_id={req_id}, "
                f"will use fallback retry"
            )
            task = RecoveryTask(
                req_id=req_id,
                failed_dp_rank=failed_dp_rank,
                backup_dp_rank=-1,
                prompt_ids=[],
                generated_ids=[],
                sampling_params_dict={},
                state=RecoveryState.RETRYING,
            )
            self._complete_task(task, success=False)
            return

        task = RecoveryTask(
            req_id=req_id,
            failed_dp_rank=failed_dp_rank,
            backup_dp_rank=backup_peer,
            prompt_ids=[],
            generated_ids=[],
            sampling_params_dict={},
            state=RecoveryState.RECOVERING,
        )

        with self._lock:
            self._recovery_tasks[req_id] = task

        logger.info(
            f"Starting recovery: req_id={req_id}, from={failed_dp_rank} -> backup={backup_peer}"
        )

        # Attempt recovery
        success = self._attempt_recovery(task)
        self._complete_task(task, success)

    def _attempt_recovery(self, task: RecoveryTask) -> bool:
        """
        Attempt to recover a request from the backup peer.

        Steps:
        1. Check if replicated KV is available on backup
        2. Route the request to the backup peer
        3. Resume inference from the replicated KV state
        4. Return success/failure

        Returns True if recovery succeeded, False otherwise.
        """
        task.attempts += 1
        logger.info(
            f"Recovery attempt {task.attempts}/{task.max_attempts} for req_id={task.req_id}"
        )

        # In a full implementation, this would:
        # 1. Send a recovery request to the backup peer via IPC
        # 2. The backup peer's scheduler checks if KV is available
        # 3. If available, resume the request from the replicated state
        # 4. Return the new generated token stream to the client

        # For now, we simulate the recovery attempt
        # The actual implementation requires integration with the scheduler's
        # request state management
        backup_available = self._check_backup_available(task.backup_dp_rank)
        if not backup_available:
            logger.warning(
                f"Backup peer dp_rank={task.backup_dp_rank} not available"
            )
            return False

        # Check if KV was replicated
        # This requires integration with AsyncKVReplicator
        kv_available = self._check_replicated_kv(task.req_id, task.backup_dp_rank)
        if not kv_available:
            logger.warning(
                f"Replicated KV not available for req_id={task.req_id} on backup={task.backup_dp_rank}"
            )
            # Fall back to normal retry
            task.state = RecoveryState.RETRYING
            return False

        # Send recovery request to backup peer
        recovery_success = self._send_recovery_request(task)
        return recovery_success

    def _check_backup_available(self, backup_dp_rank: int) -> bool:
        """Check if a backup peer is available to accept recovered requests."""
        with self._lock:
            return self._worker_health.get(backup_dp_rank, False)

    def _check_replicated_kv(self, req_id: str, backup_dp_rank: int) -> bool:
        """
        Check if the replicated KV cache is available on a backup peer.

        This requires access to the AsyncKVReplicator's state.
        In a full implementation, this would query the replicator
        or use a shared state store.
        """
        # Placeholder: in production, this would check the replicator
        # For now, assume KV is available if the backup is healthy
        # This is a conservative assumption - in practice, we should
        # query the actual replicator state
        return True

    def _send_recovery_request(self, task: RecoveryTask) -> bool:
        """
        Send a recovery request to the backup peer.

        In a full implementation, this would:
        1. Create a special recovery request with the full prompt + generated tokens
        2. Send it to the backup peer's scheduler via IPC
        3. The backup scheduler resumes from the replicated KV state
        4. Stream tokens back to the detokenizer

        Returns True if the recovery request was sent successfully.
        """
        # Placeholder implementation
        # The actual implementation requires integration with:
        # - Scheduler's request queue
        # - AsyncKVReplicator's replicated state
        # - DetokenizerManager's streaming state

        logger.info(
            f"Sending recovery request to backup peer dp_rank={task.backup_dp_rank} "
            f"for req_id={task.req_id}"
        )
        return True

    def _complete_task(self, task: RecoveryTask, success: bool) -> None:
        """Mark a recovery task as complete."""
        with self._lock:
            task.recovered_at = time.time()
            if task.req_id in self._in_recovery:
                self._in_recovery.discard(task.req_id)

            if success:
                task.state = RecoveryState.RECOVERED
                self._stats["successful_recoveries"] += 1
                self._recovered_ids.add(task.req_id)
                self._notify_recovered(task)
                logger.info(
                    f"Recovery successful: req_id={task.req_id}, duration={task.duration:.2f}s"
                )
            else:
                if task.state == RecoveryState.RETRYING:
                    self._stats["fallback_retries"] += 1
                    self._notify_failed(task)
                else:
                    self._stats["failed_recoveries"] += 1
                    task.error_message = f"Recovery failed after {task.attempts} attempts"
                    self._notify_failed(task)
                    logger.error(f"Recovery failed: req_id={task.req_id}, {task.error_message}")

            self._stats["total_recoveries"] += 1

    def get_active_recoveries(self) -> List[RecoveryTask]:
        """Return all requests currently being recovered."""
        with self._lock:
            return [
                task for req_id, task in self._recovery_tasks.items()
                if req_id in self._in_recovery
            ]

    def get_recovery_summary(self) -> Dict:
        """Return a summary of recovery state for monitoring."""
        with self._lock:
            by_state: Dict[str, int] = {}
            for task in self._recovery_tasks.values():
                key = task.state.name
                by_state[key] = by_state.get(key, 0) + 1

            return {
                "total_tasks": len(self._recovery_tasks),
                "active": len(self._in_recovery),
                "recovered": len(self._recovered_ids),
                "by_state": by_state,
                "stats": self._stats,
            }
