"""
TTLManager: Manages Time-To-Live based pinning for Continuum KV cache protection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import statistics
import os

from .tree_node import TreeNode
from .radix_tree import RadixTree


@dataclass
class ToolDurationStats:
    """Per-tool duration statistics for CDF calculation."""
    durations: List[float] = field(default_factory=list)

    def add_duration(self, duration: float) -> None:
        """Add a duration sample, keeping window limited."""
        self.durations.append(duration)
        if len(self.durations) > 500:
            self.durations = self.durations[-250:]

    def get_percentile(self, percentile: float) -> Optional[float]:
        """Get percentile value from durations."""
        if not self.durations:
            return None
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * percentile / 100)
        idx = min(idx, len(sorted_durations) - 1)
        return sorted_durations[idx]


@dataclass
class ProgramStats:
    """Per-program statistics."""
    program_id: str
    tool_durations: Dict[str, ToolDurationStats] = field(default_factory=dict)
    idle_gaps: List[float] = field(default_factory=list)

    def add_idle_gap(self, gap: float) -> None:
        """Add an idle gap sample."""
        if gap > 0:
            self.idle_gaps.append(gap)
            if len(self.idle_gaps) > 100:
                self.idle_gaps = self.idle_gaps[-50:]

    def get_tool_stats(self, tool_name: str) -> ToolDurationStats:
        """Get or create tool duration stats."""
        if tool_name not in self.tool_durations:
            self.tool_durations[tool_name] = ToolDurationStats()
        return self.tool_durations[tool_name]


@dataclass
class PinnedEntry:
    program_id: str
    node_id: int
    expire_time: float
    ttl_sec: float
    last_access_time: float
    tool_name: Optional[str] = None


@dataclass
class ToolExecution:
    timestamp: float
    program_id: str
    tool_name: str
    duration: float


class TTLManager:
    """Manages TTL-based pinning for KV cache protection."""

    def __init__(
        self,
        radix_tree: RadixTree,
        default_ttl: float = 5.0,
        min_ttl: float = 0.5,
        max_ttl: float = 120.0,  # Aligned with SimulatorConfig (论文允许较长TTL保护长工具调用)
        history_threshold: int = 100,  # K=100 per Continuum paper cold-start threshold
        enable_adaptive_ttl: bool = True,
        ttl_grid_points: int = 50,
        prefill_latency_ms_per_token: float = 0.01,
        # L1 cache parameters
        l1_reload_penalty: float = 0.3,  # L1 loading time as fraction of prefill time
        write_policy: str = "write_back",  # "write_back" or "write_through" (论文推荐write_back)
        high_pressure_threshold: float = 0.8,  # Memory pressure threshold to disable TTL
    ):
        self.radix_tree = radix_tree
        self.default_ttl = default_ttl
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.history_threshold = history_threshold
        self.enable_adaptive_ttl = enable_adaptive_ttl

        # Grid search parameters: use min_ttl/max_ttl as grid bounds
        self.ttl_grid_points = ttl_grid_points

        # Hardware parameters for utility calculation
        self.prefill_latency_ms_per_token = prefill_latency_ms_per_token

        # L1 cache parameters
        self.l1_reload_penalty = l1_reload_penalty  # Fraction of prefill time
        self.write_policy = write_policy  # "write_back" or "write_through"
        self.high_pressure_threshold = high_pressure_threshold

        self.pinned_entries: Dict[str, PinnedEntry] = {}
        self.history: List[ToolExecution] = []
        self.stats = TTLStats()
        
        # Timeline tracking for failure analysis
        self.timeline_tracker = TTLTimelineTracker()

        # Adaptive TTL statistics tracking
        self.program_stats: Dict[str, ProgramStats] = {}
        self.global_tool_durations: Dict[str, List[float]] = {}
        self.global_idle_gaps: List[float] = []
        self.global_queueing_delays: List[float] = []  # Track actual queueing delays for T calculation
        self.memoryfulness_factor: float = 1.0

        # Track TTL values used for adaptive TTL
        self.adaptive_ttl_history: List[Tuple[str, float, str]] = []  # (program_id, ttl, strategy)
    
    def pin_program(
        self,
        program_id: str,
        node: TreeNode,
        ttl_sec: Optional[float] = None,
        current_time: Optional[float] = None,
        tool_name: Optional[str] = None
    ) -> float:
        if ttl_sec is None:
            ttl_sec = self.default_ttl
        
        ttl_sec = max(self.min_ttl, min(self.max_ttl, ttl_sec))
        
        if current_time is None:
            current_time = 0.0
        
        # Only remove the specific entry for this (program_id, node_id) pair
        # This prevents one program from affecting another program's pins
        key = (program_id, node.node_id)
        if key in self.pinned_entries:
            del self.pinned_entries[key]
        
        # Set TTL on node directly (matching real SGLang)
        node.pin(ttl_sec, current_time, program_id)
        
        # Lock the node and its ancestors
        self.radix_tree.inc_lock_ref(node, current_time)
        
        entry = PinnedEntry(
            program_id=program_id,
            node_id=node.node_id,
            expire_time=node.ttl_expiry_time,
            ttl_sec=ttl_sec,
            last_access_time=current_time,
            tool_name=tool_name
        )
        # Use tuple key to allow multiple pinned nodes per program
        self.pinned_entries[(program_id, node.node_id)] = entry
        
        if tool_name:
            self.history.append(ToolExecution(
                timestamp=current_time,
                program_id=program_id,
                tool_name=tool_name,
                duration=ttl_sec
            ))
            if len(self.history) > 1000:
                self.history = self.history[-500:]
        
        self.stats.total_pins += 1
        
        # Record timeline event
        self.timeline_tracker.record_pin(
            program_id=program_id,
            node_id=node.node_id,
            ttl_sec=ttl_sec,
            current_time=current_time,
            reason=tool_name or "program pin"
        )
        return ttl_sec
    
    def unpin_program(self, program_id: str, current_time: float = 0.0, node_id: int = None) -> bool:
        # Handle tuple key format: (program_id, node_id)
        if node_id is not None:
            key = (program_id, node_id)
            if key not in self.pinned_entries:
                return False
            entry = self.pinned_entries[key]
            if entry.node_id in self.radix_tree.nodes:
                node = self.radix_tree.nodes[entry.node_id]
                if node.is_pinned and node.program_id == program_id:
                    node.unpin(current_time)
                    self.radix_tree.dec_lock_ref(node, current_time)
            del self.pinned_entries[key]
            self.stats.total_unpins += 1
            return True
        
        # Legacy: unpin by program_id only (first entry found)
        # Find all entries for this program and unpin them
        keys_to_remove = [k for k in self.pinned_entries.keys() if k[0] == program_id]
        if not keys_to_remove:
            return False
        
        for key in keys_to_remove:
            entry = self.pinned_entries[key]
            if entry.node_id in self.radix_tree.nodes:
                node = self.radix_tree.nodes[entry.node_id]
                if node.is_pinned and node.program_id == program_id:
                    node.unpin(current_time)
                    self.radix_tree.dec_lock_ref(node, current_time)
            del self.pinned_entries[key]
        
        self.stats.total_unpins += 1
        return True
    
    def cleanup_expired(self, current_time: float) -> List[str]:
        expired_programs = []
        
        for key, entry in list(self.pinned_entries.items()):
            if current_time >= entry.expire_time:
                # Extract program_id from key (key may be tuple or string)
                if isinstance(key, tuple):
                    prog_id = key[0]
                else:
                    prog_id = key
                if prog_id not in expired_programs:
                    expired_programs.append(prog_id)
        
        for prog_id in expired_programs:
            self.unpin_program(prog_id, current_time)
            self.stats.expired_pins += 1
            # Record timeline event for expiration
            for key, entry in list(self.pinned_entries.items()):
                if isinstance(key, tuple) and key[0] == prog_id:
                    self.timeline_tracker.record_expire(prog_id, entry.node_id, current_time)
        
        # Also clean up radix tree nodes with TTL expiry times
        self.radix_tree.cleanup_expired_ttls(current_time)
        
        return expired_programs
    
    def is_pinned(self, program_id: str, current_time: float = None) -> bool:
        """Check if program is currently pinned (and not expired)."""
        # Find any entry for this program
        matching_keys = [k for k in self.pinned_entries.keys() 
                       if (isinstance(k, tuple) and k[0] == program_id) or k == program_id]

        if not matching_keys:
            return False
        
        # Check all entries for this program
        for key in matching_keys:
            entry = self.pinned_entries[key]
            
            # Check if node still exists
            if entry.node_id not in self.radix_tree.nodes:
                continue
            
            node = self.radix_tree.nodes[entry.node_id]
            
            # Check if TTL has expired using simulation time
            if node.ttl_expiry_time is not None and node.is_pinned:
                # Use simulation time for expiration check
                if current_time is not None:
                    if current_time >= node.ttl_expiry_time:
                        continue
                else:
                    # Fall back to wall clock time for backwards compatibility
                    import time
                    if time.time() > node.ttl_expiry_time:
                        continue
            
            return True
        
        return False
    
    def get_remaining_ttl(self, program_id: str, current_time: float) -> Optional[float]:
        # Find any entry for this program
        for key, entry in self.pinned_entries.items():
            key_program = key[0] if isinstance(key, tuple) else key
            if key_program == program_id and entry.node_id in self.radix_tree.nodes:
                remaining = entry.expire_time - current_time
                return max(0.0, remaining)
        return 0.0
    
    def get_pinned_programs(self) -> List[str]:
        programs = set()
        for key in self.pinned_entries.keys():
            programs.add(key[0] if isinstance(key, tuple) else key)
        return list(programs)
    
    def get_pinned_programs_sorted(self, current_time: float) -> List[Tuple[str, float]]:
        """Get all pinned programs sorted by last access time (oldest first).
        
        Returns list of (program_id, last_access_time) tuples.
        """
        pinned = set()
        for key, entry in self.pinned_entries.items():
            program_id = key[0] if isinstance(key, tuple) else key
            # Verify entry is still valid (node exists and not expired)
            if entry.node_id in self.radix_tree.nodes:
                node = self.radix_tree.nodes[entry.node_id]
                if node.is_pinned and not node.check_expired(current_time):
                    pinned.add((program_id, entry.last_access_time))
        # Sort by last_access_time (oldest first = earliest arrival)
        pinned_list = list(pinned)
        pinned_list.sort(key=lambda x: x[1])
        return pinned_list
    
    def unpin_victims(
        self,
        radix_tree: RadixTree,
        current_time: float,
        max_victims: int = 1
    ) -> List[str]:
        """Unpin the oldest pinned programs to free up space for eviction.
        
        According to the paper's deadlock prevention mechanism: when cache is full
        and all nodes are pinned, proactively unpin programs in arrival order
        (oldest first) to allow eviction.
        
        Args:
            radix_tree: The radix tree to free nodes from
            current_time: Current simulation time
            max_victims: Maximum number of programs to unpin
            
        Returns:
            List of program_ids that were unpinned
        """
        pinned_sorted = self.get_pinned_programs_sorted(current_time)
        
        if not pinned_sorted:
            return []
        
        # Select victims: start with oldest programs
        victims = pinned_sorted[:max_victims]
        
        unpinned = []
        for program_id, _ in victims:
            # Find all nodes for this program and mark them as recently unpinned
            # This prevents them from being spilled to L1 (write_back optimization)
            for key, entry in list(self.pinned_entries.items()):
                if entry.program_id == program_id and entry.node_id in radix_tree.nodes:
                    node = radix_tree.nodes[entry.node_id]
                    # Mark as recently unpinned so scheduler won't spill it
                    node._force_unpinned_at = current_time
            
            if self.unpin_program(program_id, current_time):
                unpinned.append(program_id)
                self.stats.forced_unpins += 1
        
        return unpinned
    
    def get_stats(self) -> Dict:
        return {
            "total_pins": self.stats.total_pins,
            "total_unpins": self.stats.total_unpins,
            "expired_pins": self.stats.expired_pins,
            "active_pins": len(self.pinned_entries),
            "forced_unpins": self.stats.forced_unpins,
            "adaptive_samples": sum(
                len(durations) for durations in self.global_tool_durations.values()
            ),
        }

    def record_tool_execution(
        self,
        program_id: str,
        tool_name: str,
        duration: float,
        idle_gap: float = 0.0,
        queueing_delay: float = 0.0,
    ) -> None:
        """Record tool execution for CDF calculation.

        Args:
            program_id: The program ID
            tool_name: The tool that was executed
            duration: How long the tool execution took
            idle_gap: Time gap between this turn and the next
            queueing_delay: Actual queueing delay (waiting time) for this request
        """
        # Record global tool durations
        if tool_name not in self.global_tool_durations:
            self.global_tool_durations[tool_name] = []
        self.global_tool_durations[tool_name].append(duration)
        if len(self.global_tool_durations[tool_name]) > 500:
            self.global_tool_durations[tool_name] = self.global_tool_durations[tool_name][-250:]

        # Record global idle gaps
        if idle_gap > 0:
            self.global_idle_gaps.append(idle_gap)
            if len(self.global_idle_gaps) > 1000:
                self.global_idle_gaps = self.global_idle_gaps[-500:]

        # Record queueing delay for T calculation
        if queueing_delay > 0:
            self.global_queueing_delays.append(queueing_delay)
            if len(self.global_queueing_delays) > 1000:
                self.global_queueing_delays = self.global_queueing_delays[-500:]

        # Record program-specific stats
        if program_id not in self.program_stats:
            self.program_stats[program_id] = ProgramStats(program_id=program_id)

        prog_stats = self.program_stats[program_id]
        prog_stats.get_tool_stats(tool_name).add_duration(duration)
        prog_stats.add_idle_gap(idle_gap)
    
    def record_ttl_failure(self, program_id: str, current_time: float, 
                          reason: str = "TTL expired") -> None:
        """Record a TTL failure event for timeline analysis."""
        self.timeline_tracker.record_miss(program_id, current_time, reason)
    
    def get_timeline_data(self) -> List[Dict]:
        """Get TTL timeline data for visualization."""
        return self.timeline_tracker.get_timeline_data()
    
    def get_ttl_lifecycle_summary(self) -> Dict:
        """Get TTL lifecycle summary statistics."""
        return self.timeline_tracker.get_lifecycle_summary()
    
    def get_ttl_failure_analysis(self) -> Dict:
        """Get TTL failure analysis."""
        return self.timeline_tracker.get_failure_analysis()
    
    def record_prediction(
        self,
        program_id: str,
        tool_name: str,
        predicted_ttl: float,
        actual_duration: float,
        cdf_samples: int,
        strategy: str,
        cache_hit: bool,
        current_time: float,
        memory_pressure: float = 0.0
    ) -> None:
        """Record TTL prediction vs actual outcome for analysis."""
        prediction = TTLPrediction(
            timestamp=current_time,
            program_id=program_id,
            tool_name=tool_name,
            predicted_ttl=predicted_ttl,
            actual_duration=actual_duration,
            cdf_samples=cdf_samples,
            strategy=strategy,
            cache_hit=cache_hit,
            memory_pressure=memory_pressure
        )
        self.timeline_tracker.record_prediction(prediction)
    
    def get_prediction_data(self) -> List[Dict]:
        """Get TTL prediction data for visualization."""
        return [
            {
                'timestamp': p.timestamp,
                'program': p.program_id,
                'tool': p.tool_name,
                'predicted_ttl': p.predicted_ttl,
                'actual_duration': p.actual_duration,
                'cdf_samples': p.cdf_samples,
                'strategy': p.strategy,
                'cache_hit': p.cache_hit,
                'memory_pressure': p.memory_pressure,
                'error': p.predicted_ttl - p.actual_duration if p.predicted_ttl > 0 else 0,
                'error_pct': (p.predicted_ttl - p.actual_duration) / p.actual_duration * 100 if p.actual_duration > 0 else 0
            }
            for p in self.timeline_tracker.predictions
        ]
    
    def get_prediction_stats(self) -> Dict:
        """Get prediction statistics."""
        predictions = self.timeline_tracker.predictions
        if not predictions:
            return {}
        
        errors = [p.predicted_ttl - p.actual_duration for p in predictions if p.predicted_ttl > 0]
        error_pcts = [(p.predicted_ttl - p.actual_duration) / p.actual_duration * 100 
                     for p in predictions if p.actual_duration > 0 and p.predicted_ttl > 0]
        
        ttl_underestimates = sum(1 for p in predictions if p.predicted_ttl > 0 and p.predicted_ttl < p.actual_duration)
        ttl_overestimates = sum(1 for p in predictions if p.predicted_ttl > 0 and p.predicted_ttl > p.actual_duration)
        
        cache_hits = sum(1 for p in predictions if p.cache_hit)
        
        return {
            'total_predictions': len(predictions),
            'avg_predicted_ttl': sum(p.predicted_ttl for p in predictions if p.predicted_ttl > 0) / max(len([p for p in predictions if p.predicted_ttl > 0]), 1),
            'avg_actual_duration': sum(p.actual_duration for p in predictions) / len(predictions),
            'avg_error': sum(errors) / len(errors) if errors else 0,
            'avg_error_pct': sum(error_pcts) / len(error_pcts) if error_pcts else 0,
            'ttl_underestimates': ttl_underestimates,
            'ttl_overestimates': ttl_overestimates,
            'ttl_underestimate_rate': ttl_underestimates / max(len([p for p in predictions if p.predicted_ttl > 0]), 1),
            'cache_hit_rate': cache_hits / len(predictions),
        }

    def calc_adaptive_ttl(
        self,
        program_id: str,
        tool_name: Optional[str],
        miss_tokens: int,
        node_size: int,
        turn_index: int,
        total_turns: int,
        current_time: float,
        l2_enabled: bool = False,
        l2_reload_penalty: float = None,
    ) -> Tuple[float, str, int]:
        """Calculate optimal TTL using paper's utility model with L1 penalty.

        τ* = argmax_τ  P(τ, f) × (T·η + Prefill-Reload(r)) - (MemUsage(r)/M) × τ

        Extended to consider L1 penalty, L2 state, and write policy:
        - TTL + write_back: Protect node in L0, no spill
        - TTL + write_through: Protect node, but spill if evicted
        - no_ttl: Don't pin, always L0 miss

        L2-aware cost model (per paper Section 4.1.3):
        - Without L2: Prefill-Reload = prefill cost (full recomputation)
        - With L2: Prefill-Reload = L2 reload cost (typically smaller than prefill)

        Args:
            program_id: The program ID
            tool_name: The tool that was executed (or None for general)
            miss_tokens: Number of tokens that need prefill
            node_size: Size of the node in tokens
            turn_index: Current turn index (0-based)
            total_turns: Total turns in this program
            current_time: Current simulation time
            l2_enabled: Whether L2 cache is enabled (affects cost model)
            l2_reload_penalty: L2 reload cost as fraction of prefill cost.
                              If None, uses self.l1_reload_penalty.

        Returns:
            (ttl_seconds, strategy_name, cdf_samples)
        """
        if not self.enable_adaptive_ttl:
            return self.default_ttl, "disabled"

        # 1. Calculate base parameters
        T = self._calculate_T()
        eta = self._calculate_memoryfulness(turn_index, total_turns)

        # Calculate prefill cost (seconds)
        prefill_cost = miss_tokens * (self.prefill_latency_ms_per_token / 1000.0)

        # L2-aware cost model: Prefill vs Reload
        # - Without L2: Miss = full prefill (Prefill-Reload = prefill_cost)
        # - With L2: Miss = L2 reload cost (typically < prefill)
        # Use l2_reload_penalty as multiplier on prefill cost
        if l2_enabled:
            reload_cost_penalty = l2_reload_penalty if l2_reload_penalty is not None else self.l1_reload_penalty
            reload_cost = prefill_cost * reload_cost_penalty
        else:
            # Without L2, reload cost = prefill cost (no offloading available)
            reload_cost = prefill_cost

        mem_usage = node_size / max(self.radix_tree.allocator.capacity, 1)

        # 2. Get CDF durations
        cdf_durations, cdf_strategy = self._get_cdf_durations(program_id, tool_name)

        # 3. Calculate memory pressure using cache utilization
        capacity = self.radix_tree.allocator.capacity
        used = self.radix_tree.allocator.used()
        memory_pressure = used / capacity if capacity > 0 else 0.0

        # 4. Use strategy comparison with L2-aware costs
        if cdf_durations:
            best_ttl, best_strategy = self._compare_strategies(
                miss_tokens, node_size, T, eta, prefill_cost, reload_cost,
                cdf_durations, memory_pressure, l2_enabled
            )

            # Apply memory pressure adjustment to TTL
            best_ttl = self._apply_memory_pressure(best_ttl)

            # Enforce bounds
            best_ttl = max(self.min_ttl, min(self.max_ttl, best_ttl))

            # Track adaptive TTL history
            self.adaptive_ttl_history.append((program_id, best_ttl, best_strategy))

            return best_ttl, best_strategy, len(cdf_durations)

        # Track default TTL usage
        self.adaptive_ttl_history.append((program_id, self.default_ttl, "default"))
        return self.default_ttl, "default", 0

    def _calculate_T(self) -> float:
        """T: Unit memory average queueing delay (ms per token).

        Per Continuum paper: T represents the average queueing delay per unit memory.
        Formula: T = avg_queueing_delay / avg_memory_per_active_request

        We track actual queueing delays (waiting time when requests are blocked)
        and normalize by average memory usage per active request.
        """
        if self.global_queueing_delays:
            try:
                avg_queueing = statistics.mean(self.global_queueing_delays) * 1000  # Convert to ms
                avg_memory_per_request = self._get_avg_memory_per_request()
                if avg_memory_per_request > 0:
                    return avg_queueing / avg_memory_per_request
            except statistics.StatisticsError:
                pass
        # Fallback to idle gaps if no queueing data (for cold-start)
        if self.global_idle_gaps:
            try:
                avg_gap = statistics.mean(self.global_idle_gaps)
                avg_gap_ms = avg_gap * 1000  # Convert to ms
                capacity = self.radix_tree.allocator.capacity
                if capacity > 0:
                    avg_usage = self.radix_tree.allocator.used() / capacity
                    return avg_gap_ms / max(avg_usage, 0.01)
            except statistics.StatisticsError:
                pass
        # Fallback: return 0 per paper (论文规定T初始化为0)
        return 0.0

    def _get_avg_memory_per_request(self) -> float:
        """Calculate average memory per active request for T calculation."""
        active = self.radix_tree.allocator.used()
        num_programs = max(len(self.get_pinned_programs()), 1)
        return active / num_programs

    def _calculate_memoryfulness(self, turn_index: int, total_turns: int) -> float:
        """η: Memoryfulness factor = -Corr(k, N-k).

        Paper definition: η measures the negative correlation between current progress
        and remaining work. Higher η means more benefit from preserving order.

        - Earlier turns (smaller turn_index) need more protection → higher η
        - Later turns have less remaining work → lower η

        Simplified approximation for simulation:
        - η ≈ remaining_turns / total_turns
        - Range: 1.0 (first turn) → ~0.0 (last turn)
        """
        if total_turns <= 1:
            return 1.0
        # Simplified formula: η = (remaining turns) / (total turns - 1)
        # This gives η = 1.0 for first turn, η → 0 for last turn
        remaining = total_turns - turn_index
        decay = turn_index / max(total_turns - 1, 1)
        # Range: 0.05 → 1.0 (never quite reach 0 to ensure some protection)
        return max(0.05, 1.0 - decay * 0.95)

    def _apply_memory_pressure(self, ttl: float) -> float:
        """Apply memory pressure penalty when cache is highly utilized.

        When cache usage > 50%, reduce TTL proportionally:
        50% → no penalty, 75% → 0.875x, 100% → 0.75x
        """
        capacity = self.radix_tree.allocator.capacity
        if capacity > 0:
            pressure = self.radix_tree.allocator.used() / capacity
            if pressure > 0.5:
                # Linear penalty from 50% to 100%
                penalty = 1.0 - (pressure - 0.5) * 0.5
                return ttl * max(0.5, penalty)
        return ttl

    def _calculate_memory_pressure(self) -> float:
        """
        Calculate memory pressure considering active programs and token usage.
        
        Returns:
            Memory pressure as a fraction (0.0 to 1.0+)
        
        Formula: (active_programs × avg_tokens_per_program) / capacity
        """
        capacity = self.radix_tree.allocator.capacity
        if capacity <= 0:
            return 0.0
        
        # Count active programs (those with pinned entries)
        active_programs = len(self.get_pinned_programs())
        
        # Also count programs with recent activity (have entries in radix tree)
        total_tokens = self.radix_tree.allocator.used()
        
        # Estimate avg tokens per program
        avg_tokens_per_prog = total_tokens / max(active_programs, 1)
        
        # Calculate pressure based on active programs and their token usage
        pressure = (active_programs * avg_tokens_per_prog) / capacity
        
        return min(pressure, 1.5)  # Cap at 1.5 to handle overflow

    def _compare_strategies(
        self,
        miss_tokens: int,
        node_size: int,
        T: float,
        eta: float,
        prefill_cost: float,
        reload_cost: float,
        cdf_durations: List[float],
        memory_pressure: float = 0.0,
        l2_enabled: bool = False,
    ) -> Tuple[float, str]:
        """
        Compare TTL vs no-TTL strategies using paper's utility model.

        Paper formula (Section 4.1.3):
            τ* = argmax_τ  P(τ, f) × (T·η + Prefill-Reload(r)) - (MemUsage(r)/M) × τ

        Where:
        - P(τ, f) = CDF(tool_time) = probability tool finishes within TTL
        - T·η = queueing delay savings (T = unit queueing delay, η = memoryfulness)
        - Prefill-Reload(r) = cost saved on cache hit
          * Without L2: = prefill_cost (full recomputation)
          * With L2: = reload_cost (L2 load-back, typically faster)
        - (MemUsage(r)/M) × τ = memory holding cost

        L2-aware model:
        - L2 disabled: Prefill-Reload = prefill_cost
        - L2 enabled: Prefill-Reload = reload_cost (L2 reload is faster than prefill)
        """
        if not cdf_durations:
            return self.default_ttl, "default"

        capacity = self.radix_tree.allocator.capacity

        # Calculate costs in ms
        prefill_cost_ms = prefill_cost * 1000
        reload_cost_ms = reload_cost * 1000

        # Per paper: Prefill-Reload depends on L2 state
        # - Without L2: Prefill-Reload = prefill_cost (miss = full recompute)
        # - With L2: Prefill-Reload = reload_cost (miss = L2 load-back, faster)
        if l2_enabled:
            prefill_reload = reload_cost_ms
        else:
            prefill_reload = prefill_cost_ms

        # Calculate memory holding cost per τ seconds
        # Per paper: Cost(τ, r) = (MemUsage(r)/M) × τ
        # where MemUsage(r)/M = node_size/capacity (relative size of pinned request)
        # This represents the "number of average requests blocked" by pinning this request
        mem_usage_ratio = node_size / max(capacity, 1)

        # No-TTL strategy: always miss, pay reload/prefill cost
        # But we don't add this to optimization - TTL is only better if positive
        utility_no_ttl = 0.0

        # Paper Section 4.1.4: "枚举 S[f] 中所有唯一的工具调用时长作为候选（包括 τ=0）"
        # Use unique historical duration values as τ candidates
        tau_candidates = self._generate_tau_grid(cdf_durations)

        # TTL strategy: grid search over τ values
        best_ttl = 0.0
        best_value = utility_no_ttl
        best_strategy = "no_ttl"

        for tau in tau_candidates:
            # P(τ, f): CDF value at τ - probability tool finishes within τ
            P_tau = self._compute_cdf(cdf_durations, tau)

            # Per paper formula:
            # Benefit = P(τ,f) × (T·η + Prefill-Reload)
            #
            # Component 1: Queueing delay savings
            # When tool finishes within TTL, we get cache hit and save queueing delay
            queue_delay_savings = T * eta
            benefit_queue = P_tau * queue_delay_savings

            # Component 2: Prefill/Reload savings
            # If tool finishes within TTL, we get cache hit and save computation
            # Per paper: Prefill-Reload is the cost we save (depends on L2 state)
            benefit_prefill_reload = P_tau * prefill_reload

            # Total benefit per paper formula
            benefit = benefit_queue + benefit_prefill_reload

            # Cost = memory holding cost × τ
            # Per paper: Cost = (MemUsage(r)/M) × τ (unitless coefficient)
            cost = mem_usage_ratio * tau

            # TTL + write_back utility
            utility_ttl_wb = benefit - cost

            # TTL + write_through utility
            # When evicted during TTL, we pay reload penalty
            # P(evicted) ≈ eviction_prob
            eviction_prob = self.radix_tree.allocator.used() / max(capacity, 1)
            eviction_prob = min(1.0, eviction_prob)
            miss_penalty = (1 - P_tau) * eviction_prob * prefill_reload
            utility_ttl_wt = benefit - cost - miss_penalty

            # Compare utilities
            if utility_ttl_wb > best_value:
                best_value = utility_ttl_wb
                best_ttl = tau
                best_strategy = "ttl_write_back"

            if utility_ttl_wt > best_value:
                best_value = utility_ttl_wt
                best_ttl = tau
                best_strategy = "ttl_write_through"

        # Only return non-zero TTL if it's better than no_ttl
        if best_value <= utility_no_ttl:
            return 0.0, "no_ttl"

        return best_ttl, best_strategy

    def _get_cdf_durations(
        self,
        program_id: str,
        tool_name: Optional[str],
    ) -> Tuple[List[float], str]:
        """Get durations for CDF calculation with fallback chain.

        Priority:
        1. Tool-specific durations (if tool_name provided)
        2. Program-specific durations (if available)
        3. Global tool durations (if tool_name provided)
        4. All global durations (fallback)

        Returns:
            (durations_list, strategy_name)
        """
        # Debug: log the lookup
        debug = os.environ.get('DEBUG_CDF', '0') == '1'
        if debug:
            print(f"[CDF DEBUG] program={program_id}, tool={tool_name}")
            print(f"  program_stats keys: {list(self.program_stats.keys())[:5]}")
            if program_id in self.program_stats:
                print(f"  program_tools: {list(self.program_stats[program_id].tool_durations.keys())}")
            print(f"  global_tool_durations keys: {list(self.global_tool_durations.keys())}")

        # Try program-specific tool durations first
        if program_id in self.program_stats:
            prog_stats = self.program_stats[program_id]
            if tool_name and tool_name in prog_stats.tool_durations:
                durations = prog_stats.tool_durations[tool_name].durations
                if len(durations) >= self.history_threshold:
                    if debug:
                        print(f"  -> program_tool:{tool_name}, {len(durations)} samples")
                    return durations, f"program_tool:{tool_name}"

        # Try program-specific all tools
        if program_id in self.program_stats:
            prog_stats = self.program_stats[program_id]
            all_durations = []
            for tool_stat in prog_stats.tool_durations.values():
                all_durations.extend(tool_stat.durations)
            if len(all_durations) >= self.history_threshold:
                if debug:
                    print(f"  -> program_all, {len(all_durations)} samples")
                return all_durations, "program_all"

        # Try global tool-specific durations
        if tool_name and tool_name in self.global_tool_durations:
            durations = self.global_tool_durations[tool_name]
            if len(durations) >= self.history_threshold:
                if debug:
                    print(f"  -> global_tool:{tool_name}, {len(durations)} samples")
                return durations, f"global_tool:{tool_name}"

        # Fallback: all global durations
        all_global = []
        for durations in self.global_tool_durations.values():
            all_global.extend(durations)
        if len(all_global) >= self.history_threshold:
            if debug:
                print(f"  -> global_all, {len(all_global)} samples")
            return all_global, "global_all"

        # Not enough data
        if debug:
            print(f"  -> insufficient_data")
        return [], "insufficient_data"

    def _compute_cdf(self, durations: List[float], tau: float) -> float:
        """Compute P(τ, f) - probability that duration <= τ.

        Empirical CDF: fraction of durations <= tau.
        """
        if not durations:
            return 0.0
        count = sum(1 for d in durations if d <= tau)
        return count / len(durations)

    def _generate_tau_grid(self, cdf_durations: List[float] = None) -> List[float]:
        """Generate grid of τ values for grid search.

        Per paper Section 4.1.4:
        "枚举 S[f] 中所有唯一的工具调用时长作为候选（包括 τ=0），
         选择期望奖励最高的。"

        Uses unique historical duration values as candidates,
        plus additional points for better coverage.

        Args:
            cdf_durations: Historical tool call durations to use as candidates
        """
        candidates = [0.0]  # τ=0 means no pinning (always included per paper)

        # Paper: enumerate unique tool call durations from S[f] as candidates
        if cdf_durations:
            unique_durations = sorted(set(cdf_durations))
            candidates.extend(unique_durations)

        # If we don't have enough candidates, add linear grid points
        if len(candidates) < 3:
            if self.ttl_grid_points <= 1:
                return [self.default_ttl]
            step = (self.max_ttl - self.min_ttl) / max(self.ttl_grid_points - 1, 1)
            for i in range(self.ttl_grid_points):
                tau = self.min_ttl + i * step
                if tau not in candidates:
                    candidates.append(tau)

        # Sort and deduplicate
        return sorted(set(candidates))


@dataclass
class TTLStats:
    total_pins: int = 0
    total_unpins: int = 0
    expired_pins: int = 0
    ttl_extensions: int = 0
    cache_hits_while_pinned: int = 0
    forced_unpins: int = 0


@dataclass
class TTLEvent:
    """Record of a TTL lifecycle event for timeline tracking."""
    timestamp: float
    event_type: str  # 'pin', 'unpin', 'expire', 'hit', 'miss', 'extend'
    program_id: str
    node_id: int
    ttl_sec: float
    remaining_ttl: float = 0.0
    reason: str = ""
    cache_hit: bool = False


@dataclass
class TTLPrediction:
    """Record of TTL prediction vs actual tool execution."""
    timestamp: float
    program_id: str
    tool_name: str
    predicted_ttl: float
    actual_duration: float
    cdf_samples: int
    strategy: str
    cache_hit: bool = False
    memory_pressure: float = 0.0


class TTLTimelineTracker:
    """Tracks TTL lifecycle events for timeline visualization and failure analysis."""
    
    def __init__(self):
        self.events: List[TTLEvent] = []
        self.pinned_lifecycles: Dict[str, Dict] = {}  # program_id -> lifecycle info
        # Track predictions vs actuals
        self.predictions: List[TTLPrediction] = []
        
    def record_prediction(self, prediction: TTLPrediction) -> None:
        """Record a TTL prediction with actual outcome."""
        self.predictions.append(prediction)
        if len(self.predictions) > 10000:  # Keep last 10000 predictions
            self.predictions = self.predictions[-5000:]
    
    def record_pin(self, program_id: str, node_id: int, ttl_sec: float, 
                   current_time: float, reason: str = "") -> None:
        """Record a TTL pin event."""
        self.events.append(TTLEvent(
            timestamp=current_time,
            event_type='pin',
            program_id=program_id,
            node_id=node_id,
            ttl_sec=ttl_sec,
            remaining_ttl=ttl_sec,
            reason=reason
        ))
        # Track lifecycle
        if program_id not in self.pinned_lifecycles:
            self.pinned_lifecycles[program_id] = {
                'pin_time': current_time,
                'ttl_sec': ttl_sec,
                'expire_time': current_time + ttl_sec,
                'hit_count': 0,
                'miss_count': 0,
                'unpin_time': None,
                'unpin_reason': None
            }
        else:
            # Update TTL (may have been extended)
            self.pinned_lifecycles[program_id]['ttl_sec'] = ttl_sec
            self.pinned_lifecycles[program_id]['expire_time'] = current_time + ttl_sec
    
    def record_hit(self, program_id: str, current_time: float) -> None:
        """Record a cache hit while TTL is active."""
        self.events.append(TTLEvent(
            timestamp=current_time,
            event_type='hit',
            program_id=program_id,
            node_id=-1,
            ttl_sec=0.0,
            remaining_ttl=0.0,
            cache_hit=True
        ))
        if program_id in self.pinned_lifecycles:
            self.pinned_lifecycles[program_id]['hit_count'] += 1
    
    def record_miss(self, program_id: str, current_time: float, 
                    reason: str = "TTL expired") -> None:
        """Record a cache miss due to TTL failure."""
        self.events.append(TTLEvent(
            timestamp=current_time,
            event_type='miss',
            program_id=program_id,
            node_id=-1,
            ttl_sec=0.0,
            remaining_ttl=0.0,
            reason=reason,
            cache_hit=False
        ))
        if program_id in self.pinned_lifecycles:
            self.pinned_lifecycles[program_id]['miss_count'] += 1
    
    def record_expire(self, program_id: str, node_id: int, current_time: float) -> None:
        """Record TTL expiration event."""
        self.events.append(TTLEvent(
            timestamp=current_time,
            event_type='expire',
            program_id=program_id,
            node_id=node_id,
            ttl_sec=0.0,
            remaining_ttl=0.0,
            reason="TTL timer expired"
        ))
        if program_id in self.pinned_lifecycles:
            self.pinned_lifecycles[program_id]['unpin_time'] = current_time
            self.pinned_lifecycles[program_id]['unpin_reason'] = "expired"
    
    def record_unpin(self, program_id: str, node_id: int, current_time: float,
                     reason: str = "") -> None:
        """Record explicit unpin event."""
        self.events.append(TTLEvent(
            timestamp=current_time,
            event_type='unpin',
            program_id=program_id,
            node_id=node_id,
            ttl_sec=0.0,
            remaining_ttl=0.0,
            reason=reason
        ))
        if program_id in self.pinned_lifecycles:
            self.pinned_lifecycles[program_id]['unpin_time'] = current_time
            self.pinned_lifecycles[program_id]['unpin_reason'] = reason
    
    def get_timeline_data(self) -> List[Dict]:
        """Get timeline data for visualization."""
        return [
            {
                'timestamp': e.timestamp,
                'event': e.event_type,
                'program': e.program_id,
                'node': e.node_id,
                'ttl': e.ttl_sec,
                'remaining': e.remaining_ttl,
                'reason': e.reason
            }
            for e in self.events
        ]
    
    def get_lifecycle_summary(self) -> Dict:
        """Get summary statistics for TTL lifecycles."""
        if not self.pinned_lifecycles:
            return {}
        
        lifecycles = list(self.pinned_lifecycles.values())
        durations = []
        hit_rates = []
        
        for lc in lifecycles:
            if lc['unpin_time']:
                duration = lc['unpin_time'] - lc['pin_time']
                durations.append(duration)
            total = lc['hit_count'] + lc['miss_count']
            if total > 0:
                hit_rates.append(lc['hit_count'] / total)
        
        return {
            'total_pins': len(lifecycles),
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'avg_hit_rate': sum(hit_rates) / len(hit_rates) if hit_rates else 0,
            'lifecycles': self.pinned_lifecycles
        }
    
    def get_failure_analysis(self) -> Dict:
        """Analyze TTL failure patterns."""
        miss_events = [e for e in self.events if e.event_type == 'miss']
        expire_events = [e for e in self.events if e.event_type == 'expire']
        
        failure_reasons = {}
        for e in miss_events:
            reason = e.reason or "unknown"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            'total_misses': len(miss_events),
            'total_expires': len(expire_events),
            'failure_reasons': failure_reasons,
            'failure_rate': len(miss_events) / max(len(self.events), 1)
        }
