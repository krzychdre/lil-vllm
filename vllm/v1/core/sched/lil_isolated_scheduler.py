# SPDX-License-Identifier: Apache-2.0
"""Isolated round-robin scheduler for vLLM (GLM-5.3 full, KK 20261011), phase P3a.

Goal: the result of a request must not depend on what other sessions compute.
The scheduler NEVER puts tokens of two different requests into one model step,
but rotates requests step by step (no "wait for the whole answer" as with
--max-num-seqs 1).

Mechanism (no copy of the scheduling logic; the stock AsyncScheduler.schedule()
does all the work, it is only shown one request per step):

* one request is selected per step (round-robin by arrival order, quantum in
  steps, see the env vars below);
* every other RUNNING request is hidden from the running pass through the
  existing per-request gate ``next_decode_eligible_step`` (scheduler.py:1066),
  set to ``step + 1`` for the duration of schedule() and restored afterwards;
* admission from the waiting queues is limited through
  ``max_num_running_reqs`` (scheduler.py:1366-1368): equal to the current
  number of slots on a running turn (nothing admitted), one more on a waiting
  turn (exactly one admitted);
* the compute-share controller and parallel prefills are disabled: with one
  request per step there is no contention to arbitrate, and these are the
  only places where the budget/chunk of a request depends on other requests.

Enable: ``--scheduler-cls=lil_isolated_scheduler.IsolatedAsyncScheduler`` with
this file importable in the engine process (e.g. mounted into site-packages).

Environment (read at import):
  VLLM_ISOLATED_SCHED_QUANTUM          steps per turn (default 1)
  VLLM_ISOLATED_SCHED_DECODE_QUANTUM   steps per turn for a decoding request
                                       (default = QUANTUM); >1 lets decodes run
                                       several steps between prefill chunks
  VLLM_ISOLATED_SCHED_CHECK            1 (default): check "one request per
                                       step" on every step (O(1)), WARNING with
                                       request ids on violation, periodic INFO
                                       summary; 0: silent
  VLLM_ISOLATED_SCHED_LOG_INTERVAL     seconds between summaries (default 60)
"""

from __future__ import annotations

import os
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger("vllm.lil_isolated_scheduler")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {raw!r}")
    return value


QUANTUM = _env_int("VLLM_ISOLATED_SCHED_QUANTUM", 1)
DECODE_QUANTUM = _env_int("VLLM_ISOLATED_SCHED_DECODE_QUANTUM", QUANTUM)
CHECK = os.getenv("VLLM_ISOLATED_SCHED_CHECK", "1").strip() not in ("0", "false", "")
LOG_INTERVAL_S = float(os.getenv("VLLM_ISOLATED_SCHED_LOG_INTERVAL", "60"))

_RUN = "run"
_WAIT = "wait"


class IsolatedAsyncScheduler(AsyncScheduler):
    """AsyncScheduler that schedules at most one request per model step."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # With one request per step there is no prefill/decode contention, and
        # these controllers are the only places that shrink or defer a
        # request's work because OTHER requests exist (scheduler.py:898-1006,
        # 979-988, 1104-1110, 1641-1667). A lone request never triggers them,
        # so disabling them keeps "isolated" == "alone on the server".
        self._iso_disabled_compute_share = self.compute_share_controller is not None
        self.compute_share_controller = None
        self.prefill_fairness_max_tokens = None
        self.max_parallel_prefills = 1

        self._iso_seq: dict[str, int] = {}
        self._iso_next_seq = 0
        self._iso_last_seq = -1  # rotation pointer (arrival seq of last served)
        self._iso_holder: str | None = None  # request holding the current turn
        self._iso_left = 0  # remaining steps of the holder's turn
        # Free-block count at the last failed waiting turn (admission back-off).
        self._iso_wait_fail_free: int | None = None

        self._iso_stat_steps = 0
        self._iso_stat_nonempty = 0
        self._iso_stat_empty_selected = 0
        self._iso_stat_max_reqs = 0
        self._iso_stat_violations = 0
        self._iso_stat_last_log = time.monotonic()

        logger.info(
            "IsolatedAsyncScheduler active: one request per step, round-robin, "
            "quantum=%d decode_quantum=%d check=%s; prefill_compute_share %s, "
            "max_parallel_prefills forced to 1 (configured %s)",
            QUANTUM,
            DECODE_QUANTUM,
            CHECK,
            "DISABLED (was %s)" % self.scheduler_config.prefill_compute_share
            if self._iso_disabled_compute_share
            else "off",
            self.scheduler_config.max_parallel_prefills,
        )

    # ------------------------------------------------------------------ queue
    def add_request(self, request: Request) -> None:
        super().add_request(request)
        rid = request.request_id
        if rid in self.requests and rid not in self._iso_seq:
            self._iso_seq[rid] = self._iso_next_seq
            self._iso_next_seq += 1

    def _free_request(self, request: Request, delay_free_blocks: bool = False):
        result = super()._free_request(request, delay_free_blocks)
        self._iso_seq.pop(request.request_id, None)
        if self._iso_holder == request.request_id:
            self._iso_holder = None
            self._iso_left = 0
        # Freed blocks may unblock a waiting request.
        self._iso_wait_fail_free = None
        return result

    def set_prefill_fairness(self, config):  # dev API, would re-create the controller
        raise ValueError(
            "prefill fairness is disabled by IsolatedAsyncScheduler "
            "(one request per step)"
        )

    # -------------------------------------------------------------- selection
    def _iso_seq_of(self, request: Request) -> int:
        seq = self._iso_seq.get(request.request_id)
        if seq is None:  # defensive: request added through another path
            seq = self._iso_seq[request.request_id] = self._iso_next_seq
            self._iso_next_seq += 1
        return seq

    def _iso_running_ready(self, request: Request, step: int) -> bool:
        """Mirror of the cheap skip checks of the running pass
        (scheduler.py:1049-1121, 1153); allocation is not predicted."""
        if step < request.next_decode_eligible_step:
            return False
        if (
            request.num_output_placeholders > 0
            and request.num_computed_tokens + 2 - request.num_output_placeholders
            >= request.num_prompt_tokens + request.max_tokens
        ):
            return False
        num_new = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        num_new = min(
            num_new,
            self.max_model_len
            - request.num_computed_tokens
            - self.num_sampled_tokens_per_step,
        )
        if num_new <= 0:
            return False
        return not self._ec_transfer_pending(
            request, request.num_computed_tokens - request.num_output_placeholders
        )

    def _iso_waiting_admissible(self, request: Request) -> bool:
        status = request.status
        if status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            so = request.structured_output_request
            if so is None:
                return False
            grammar = so.grammar
            return grammar is not None and not isinstance(grammar, Exception)
        if status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            return request.request_id in self.finished_recving_kv_req_ids
        if status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
            return False
        return not (
            request.num_stale_output_tokens > 0 and not request.drop_stale_output
        )

    def _iso_waiting_candidate(self) -> Request | None:
        if self._pause_state != PauseState.UNPAUSED:
            return None
        if (
            len(self.running) + self.num_waiting_for_streaming_input
            >= self.max_num_running_reqs
        ):
            return None
        best: Request | None = None
        best_seq = 0
        for queue in (self.skipped_waiting, self.waiting):
            for request in queue:
                if not self._iso_waiting_admissible(request):
                    continue
                seq = self._iso_seq_of(request)
                if best is None or seq < best_seq:
                    best, best_seq = request, seq
        return best

    def _iso_select(self, step: int) -> tuple[Request | None, str | None]:
        cands: list[tuple[int, str, Request, bool]] = []
        for request in self.running:
            cands.append(
                (
                    self._iso_seq_of(request),
                    _RUN,
                    request,
                    self._iso_running_ready(request, step),
                )
            )
        waiting = self._iso_waiting_candidate()
        if waiting is not None:
            ready = (
                self._iso_wait_fail_free is None
                or self.kv_cache_manager.block_pool.get_num_free_blocks()
                > self._iso_wait_fail_free
            )
            cands.append((self._iso_seq_of(waiting), _WAIT, waiting, ready))
        if not cands:
            return None, None

        # Continue the current turn (quantum > 1).
        if self._iso_holder is not None and self._iso_left > 0:
            for _, kind, request, ready in cands:
                if kind == _RUN and ready and request.request_id == self._iso_holder:
                    return request, kind

        cands.sort(key=lambda c: c[0])
        last = self._iso_last_seq
        ordered = [c for c in cands if c[0] > last] + [c for c in cands if c[0] <= last]
        for _, kind, request, ready in ordered:
            if ready:
                return request, kind
        # Nothing predicted runnable: still try the next one in rotation, so a
        # wrong prediction can cost at most one empty step per candidate,
        # never a stall.
        _, kind, request, _ = ordered[0]
        return request, kind

    # ---------------------------------------------------------------- schedule
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        step = self.current_step + 1  # value schedule() is about to use
        selected, kind = self._iso_select(step)

        saved_max = self.max_num_running_reqs
        slots = len(self.running) + self.num_waiting_for_streaming_input
        gated: list[tuple[Request, int]] = []
        try:
            for request in self.running:
                if request is selected:
                    continue
                gated.append((request, request.next_decode_eligible_step))
                if request.next_decode_eligible_step <= step:
                    request.next_decode_eligible_step = step + 1
            self.max_num_running_reqs = min(
                saved_max, slots + 1 if kind == _WAIT else slots
            )
            output = super().schedule(throttle_prefills)
        finally:
            self.max_num_running_reqs = saved_max
            for request, value in gated:
                request.next_decode_eligible_step = value

        self._iso_after_schedule(output, selected, kind)
        return output

    def _iso_after_schedule(
        self, output: SchedulerOutput, selected: Request | None, kind: str | None
    ) -> None:
        scheduled = output.num_scheduled_tokens
        num_reqs = len(scheduled)
        self._iso_stat_steps += 1
        if num_reqs:
            self._iso_stat_nonempty += 1
            if num_reqs > self._iso_stat_max_reqs:
                self._iso_stat_max_reqs = num_reqs
        if num_reqs > 1:
            self._iso_stat_violations += 1
            if CHECK:
                logger.warning(
                    "isolated-scheduler: R1 VIOLATION at step %d: %d requests in "
                    "one step: %s",
                    self.current_step,
                    num_reqs,
                    dict(scheduled),
                )

        if num_reqs:
            rid = next(iter(scheduled))
            is_decode = rid in output.scheduled_spec_decode_tokens or (
                scheduled[rid] == 1
            )
            if rid == self._iso_holder:
                self._iso_left -= 1
            else:
                self._iso_holder = rid
                self._iso_left = (DECODE_QUANTUM if is_decode else QUANTUM) - 1
            seq = self._iso_seq.get(rid)
            if seq is not None:
                self._iso_last_seq = seq
            if kind == _WAIT:
                self._iso_wait_fail_free = None
        else:
            self._iso_holder = None
            self._iso_left = 0
            if selected is not None:
                self._iso_stat_empty_selected += 1
                seq = self._iso_seq.get(selected.request_id)
                if seq is not None:
                    self._iso_last_seq = seq  # move past it
                if kind == _WAIT:
                    self._iso_wait_fail_free = (
                        self.kv_cache_manager.block_pool.get_num_free_blocks()
                    )

        if CHECK and LOG_INTERVAL_S > 0:
            now = time.monotonic()
            if now - self._iso_stat_last_log >= LOG_INTERVAL_S:
                if self._iso_stat_nonempty:
                    logger.info(
                        "isolated-scheduler: %d steps (%d non-empty), max requests "
                        "per step %d, R1 violations %d, empty selected steps %d, "
                        "running %d, waiting %d",
                        self._iso_stat_steps,
                        self._iso_stat_nonempty,
                        self._iso_stat_max_reqs,
                        self._iso_stat_violations,
                        self._iso_stat_empty_selected,
                        len(self.running),
                        len(self.waiting) + len(self.skipped_waiting),
                    )
                self._iso_stat_last_log = now
                self._iso_stat_steps = 0
                self._iso_stat_nonempty = 0
                self._iso_stat_max_reqs = 0
                self._iso_stat_empty_selected = 0
                # violations are cumulative on purpose
