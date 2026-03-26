# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ToolCallEstimator: Estimates tool-call execution delay for per-turn scheduling.

This module is ported from vllm-continuum and adapted for SGLang.
It tracks function-call execution times and predicts how long to pin
KV cache after a request finishes, based on the tool that was called.

Usage in the scheduler:
  1. On request arrival:
       estimator.request_arrives(req)
  2. On request finish (after decoding):
       estimator.request_finished(req, decoded_text)
  3. On scheduling decision:
       pin_duration = estimator.get_pin_duration(req)
       is_last = estimator.get_is_last_step(req)
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Default pin duration (seconds). If the next tool's avg execution time
# exceeds this threshold, we skip pinning (the wait would be too long).
FIXED_PIN_DURATION: float = 2.0


class ToolCallParser:
    """
    Parse function / tool calls from LLM output text.

    Currently supports:
    - Bash code blocks: ```bash ... ```
    - Generic code blocks: ``` ... ```
    - General function names (first word in code blocks)

    The parser returns the command / function name, not the full invocation.
    This is intentionally simple — a real system would parse JSON tool calls
    (e.g., BFCL format) or structured outputs.
    """

    # Match bash code blocks and extract content
    _BASH_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
    # Match any code block
    _CODE_RE = re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL)

    def parse(self, text: str) -> Optional[str]:
        """
        Extract the first command / function name from LLM output text.

        Returns:
            The tool name (first word of the first code block), or None.
        """
        if not text:
            return None

        # Try bash blocks first
        matches = self._BASH_RE.findall(text)
        if matches:
            cmd = matches[0].strip().split()[0] if matches[0].strip() else None
            return cmd

        # Fall back to generic code blocks
        matches = self._CODE_RE.findall(text)
        if matches:
            cmd = matches[0].strip().split()[0] if matches[0].strip() else None
            return cmd

        return None


class ToolCallEstimator:
    """
    Estimate tool-call execution delay and manage per-turn scheduling metadata.

    Tracks execution times per tool, history per job_id / session_id, and
    provides pin-duration predictions for the Continuum KV management.

    Design notes:
    - job_id takes priority over session_id as the affinity key.
    - session_id is used as a fallback for requests created via Session.create_req.
    - When neither is available, each request is treated independently.
    """

    def __init__(self):
        self.parser = ToolCallParser()

        # func_name → rolling average execution time (seconds)
        self.exec_time_avg: Dict[str, float] = {}

        # func_name → list of recent samples (for rolling average)
        self.exec_time_samples: Dict[str, List[float]] = {}

        # affinity_key (job_id or session_id) → list of history entries
        # Each entry: {"arrival": float, "departure": float, "func_call": str|None}
        self.job_history: Dict[str, List[Dict]] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def request_arrives(self, req: "Req") -> None:
        """
        Called by the scheduler when a new request arrives at the waiting queue.

        Actions:
        - Creates a new history entry for the affinity key (job_id / session_id)
          if it does not yet exist.
        - Sets req.last_func_call from the previous history entry, so the
          ToolCallEstimator already knows which tool was called before.
        - If the previous step's tool call and departure time are available,
          updates the rolling average for that tool.
        """
        affinity_key = self._get_affinity_key(req)
        if not affinity_key:
            return

        if affinity_key not in self.job_history:
            self.job_history[affinity_key] = []
            req.last_func_call = None
        else:
            # Retrieve the previous turn's info
            last_entry = self.job_history[affinity_key][-1]
            req.last_func_call = last_entry.get("func_call")

            # Update rolling average for the previous tool if we now know its duration
            if "departure" in last_entry and "arrival" in last_entry:
                prev_func = last_entry.get("func_call")
                if prev_func:
                    exec_time = last_entry["departure"] - last_entry["arrival"]
                    self._update_exec_time(prev_func, exec_time)
                    logger.debug(
                        "ToolCallEstimator: updated avg exec time for %s = %.3fs (from %.3fs)",
                        prev_func,
                        self.exec_time_avg.get(prev_func, 0),
                        exec_time,
                    )

        # Append new arrival entry (departure will be filled later by request_finished)
        self.job_history[affinity_key].append({
            "arrival": time.time(),
        })

    def request_finished(self, req: "Req", decoded_text: str) -> None:
        """
        Called by the scheduler when a request has finished decoding.

        Actions:
        - Parses the tool name from the decoded text.
        - Records it in req.this_func_call.
        - Updates the history entry's departure time and func_call.
        - Updates req.is_last_step based on whether a tool call was made.
        """
        affinity_key = self._get_affinity_key(req)
        if not affinity_key or affinity_key not in self.job_history:
            req.this_func_call = None
            req.is_last_step = True
            return

        # Parse tool call from output text
        this_func_call = self.parser.parse(decoded_text)
        req.this_func_call = this_func_call

        # If no tool call was made, this is the last step
        req.is_last_step = (this_func_call is None)

        # Update the most recent history entry
        if self.job_history[affinity_key]:
            entry = self.job_history[affinity_key][-1]
            entry["departure"] = time.time()
            entry["func_call"] = this_func_call

            logger.debug(
                "ToolCallEstimator: request %s finished, func=%s, is_last=%s, "
                "exec_time=%.3fs",
                req.rid,
                this_func_call,
                req.is_last_step,
                entry["departure"] - entry["arrival"],
            )

    def get_pin_duration(self, req: "Req") -> float:
        """
        Estimate how long to pin the KV cache after this request finishes.

        The pin keeps the request's KV cache alive on GPU so the next turn
        in the same job can reuse it without reloading.

        Logic (mirrors vllm-continuum):
        - If last_func_call's avg execution time > FIXED_PIN_DURATION,
          skip pinning (the tool is slow; by the time the next turn arrives,
          the KV would have been evicted anyway).
        - Otherwise, pin for FIXED_PIN_DURATION seconds.
        - If last_func_call is None (first turn), pin for FIXED_PIN_DURATION.

        Returns:
            Pin duration in seconds (0 means do not pin).
        """
        if req.last_func_call:
            avg_time = self.exec_time_avg.get(req.last_func_call, 0.0)
            if avg_time > FIXED_PIN_DURATION:
                logger.debug(
                    "ToolCallEstimator: skip pin for %s (avg exec %.3fs > %.1fs)",
                    req.last_func_call,
                    avg_time,
                    FIXED_PIN_DURATION,
                )
                return 0.0
        return FIXED_PIN_DURATION

    def get_is_last_step(self, req: "Req") -> bool:
        """
        Return whether this request is the last step of its job.

        The result is set by request_finished() — if a tool call was parsed,
        there will be another turn, so is_last_step = False.
        """
        return getattr(req, "is_last_step", True)

    def get_last_func_call(self, req: "Req") -> Optional[str]:
        """Return the last tool call name for this request's job."""
        return getattr(req, "last_func_call", None)

    def get_this_func_call(self, req: "Req") -> Optional[str]:
        """Return the tool call name parsed from this request's output."""
        return getattr(req, "this_func_call", None)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_affinity_key(self, req: "Req") -> Optional[str]:
        """Return job_id if available, else session_id, else None."""
        if req.job_id:
            return req.job_id
        if req.session_id:
            return req.session_id
        return None

    def _update_exec_time(self, func: str, exec_time: float) -> None:
        """
        Update rolling average for a tool's execution time.

        Keeps the last 100 samples and computes a simple mean.
        """
        if func not in self.exec_time_samples:
            self.exec_time_samples[func] = []
        self.exec_time_samples[func].append(exec_time)
        if len(self.exec_time_samples[func]) > 100:
            self.exec_time_samples[func] = self.exec_time_samples[func][-100:]
        samples = self.exec_time_samples[func]
        self.exec_time_avg[func] = sum(samples) / len(samples)

    def get_exec_time_avg(self, func: str) -> Optional[float]:
        """Return the rolling average execution time for a tool, or None."""
        return self.exec_time_avg.get(func)
