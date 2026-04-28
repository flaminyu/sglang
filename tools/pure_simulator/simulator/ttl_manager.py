"""
TTLManager: Manages Time-To-Live based pinning for Continuum KV cache protection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import statistics

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
        max_ttl: float = 15.0,
        history_threshold: int = 1,  # Start adaptive TTL immediately (tool intervals are fixed)
        enable_adaptive_ttl: bool = True,
        ttl_grid_min: float = 1.0,
        ttl_grid_max: float = 60.0,
        ttl_grid_points: int = 50,
        prefill_latency_ms_per_token: float = 0.01,
        # L1 cache parameters
        l1_reload_penalty: float = 0.3,  # L1 loading time as fraction of prefill time
        write_policy: str = "write_through",  # "write_back" or "write_through"
        high_pressure_threshold: float = 0.8,  # Memory pressure threshold to disable TTL
    ):
        self.radix_tree = radix_tree
        self.default_ttl = default_ttl
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.history_threshold = history_threshold
        self.enable_adaptive_ttl = enable_adaptive_ttl

        # CDF grid search parameters
        self.ttl_grid_min = ttl_grid_min
        self.ttl_grid_max = ttl_grid_max
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

        # Adaptive TTL statistics tracking
        self.program_stats: Dict[str, ProgramStats] = {}
        self.global_tool_durations: Dict[str, List[float]] = {}
        self.global_idle_gaps: List[float] = []
        self.memoryfulness_factor: float = 1.0
    
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
        expired = []
        
        for key, entry in list(self.pinned_entries.items()):
            if current_time >= entry.expire_time:
                expired.append(key[0] if isinstance(key, tuple) else key)
        
        for key in expired:
            if isinstance(key, tuple):
                self.unpin_program(key[0], current_time, key[1])
            else:
                self.unpin_program(key, current_time)
            self.stats.expired_pins += 1
        
        # Also clean up radix tree nodes with TTL expiry times
        self.radix_tree.cleanup_expired_ttls(current_time)
        
        return expired
    
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
    ) -> None:
        """Record tool execution for CDF calculation.

        Args:
            program_id: The program ID
            tool_name: The tool that was executed
            duration: How long the tool execution took
            idle_gap: Time gap between this turn and the next
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

        # Record program-specific stats
        if program_id not in self.program_stats:
            self.program_stats[program_id] = ProgramStats(program_id=program_id)

        prog_stats = self.program_stats[program_id]
        prog_stats.get_tool_stats(tool_name).add_duration(duration)
        prog_stats.add_idle_gap(idle_gap)

    def calc_adaptive_ttl(
        self,
        program_id: str,
        tool_name: Optional[str],
        miss_tokens: int,
        node_size: int,
        turn_index: int,
        total_turns: int,
        current_time: float,
    ) -> Tuple[float, str]:
        """Calculate optimal TTL using paper's utility model with L1 penalty.

        τ* = argmax_τ  P(τ, f) × (T·η + Prefill-Reload(r)) - (MemUsage(r)/M) × τ

        Extended to consider L1 penalty and write policy:
        - TTL + write_back: Protect node in L0, no spill
        - TTL + write_through: Protect node, but spill if evicted
        - no_ttl: Don't pin, always L0 miss

        Args:
            program_id: The program ID
            tool_name: The tool that was executed (or None for general)
            miss_tokens: Number of tokens that need prefill
            node_size: Size of the node in tokens
            turn_index: Current turn index (0-based)
            total_turns: Total turns in this program
            current_time: Current simulation time

        Returns:
            (ttl_seconds, strategy_name)
        """
        if not self.enable_adaptive_ttl:
            return self.default_ttl, "disabled"

        # 1. Calculate base parameters
        T = self._calculate_T()
        eta = self._calculate_memoryfulness(turn_index, total_turns)
        prefill_reload = miss_tokens * (self.prefill_latency_ms_per_token / 1000.0)
        mem_usage = node_size / max(self.radix_tree.allocator.capacity, 1)

        # 2. Get CDF durations
        cdf_durations, cdf_strategy = self._get_cdf_durations(program_id, tool_name)

        # 3. Calculate memory pressure using cache utilization
        capacity = self.radix_tree.allocator.capacity
        used = self.radix_tree.allocator.used()
        memory_pressure = used / capacity if capacity > 0 else 0.0

        # 4. Use strategy comparison
        if cdf_durations:
            best_ttl, best_strategy = self._compare_strategies(
                miss_tokens, node_size, T, eta, prefill_reload, cdf_durations, memory_pressure
            )

            # Apply memory pressure adjustment to TTL
            best_ttl = self._apply_memory_pressure(best_ttl)

            # Enforce bounds
            best_ttl = max(self.min_ttl, min(self.max_ttl, best_ttl))

            return best_ttl, best_strategy

        return self.default_ttl, "default"

    def _calculate_T(self) -> float:
        """T: Unit memory average queueing delay (ms per token).

        Based on idle gaps between requests.
        """
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
        return self.default_ttl * 500  # Fallback: default TTL in ms

    def _calculate_memoryfulness(self, turn_index: int, total_turns: int) -> float:
        """η: Memoryfulness factor = -Corr(k, N-k).

        - Earlier turns (smaller turn_index) need more protection → higher η
        - Later turns have less remaining work → lower η
        """
        if total_turns <= 1:
            return 1.0
        # Linear decay: η = 1 - (turn / (total-1)) * 0.8
        # Range: 1.0 (first turn) to 0.2 (last turn)
        decay = (turn_index / max(total_turns - 1, 1)) * 0.8
        return max(0.1, 1.0 - decay)

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
        prefill_reload: float,
        cdf_durations: List[float],
        memory_pressure: float = 0.0,
    ) -> Tuple[float, str]:
        """
        Compare TTL vs no-TTL strategies.

        Simplified utility model:
        - Benefit = P(hit) × prefill_savings + L1_reload_avoidance
        - Cost = memory_holding_cost × τ

        Where:
        - P(hit) = CDF(tool_time) = probability tool finishes within TTL
        - prefill_savings = miss_tokens × prefill_latency (what we save on hit)
        - L1_reload_avoidance = P(evicted) × L1_reload_penalty × prefill_cost
        - memory_holding_cost = (node_size / capacity) × base_cost_per_sec
        """
        if not cdf_durations:
            return self.default_ttl, "default"

        capacity = self.radix_tree.allocator.capacity

        # Calculate prefill cost in ms
        prefill_cost = prefill_reload * 1000  # Convert to ms

        # Calculate memory cost per second (in ms)
        # This is the opportunity cost of holding node_size tokens in memory
        # Base cost: 1ms per second per 1% memory usage (much lower than before)
        mem_usage = node_size / max(capacity, 1)
        cost_per_sec_ms = mem_usage * 10  # 10ms per second per 100% memory usage

        # Strategy 1: no TTL (baseline)
        utility_no_ttl = 0.0

        # Strategy 2: TTL with grid search
        best_ttl = 0.0
        best_value = utility_no_ttl
        best_strategy = "no_ttl"

        for tau in self._generate_tau_grid():
            # P(τ, f): CDF value at τ - probability tool finishes within τ
            P_tau = self._compute_cdf(cdf_durations, tau)

            # Benefit 1: Prefill savings on cache hit
            # If tool finishes within TTL, we get a cache hit and save prefill time
            prefill_savings = prefill_cost
            benefit1 = P_tau * prefill_savings

            # Benefit 2: L1 reload avoidance
            # If TTL protects node from eviction, we avoid L1 reload penalty
            # P(evicted) ≈ memory_pressure (fraction of nodes likely to be evicted)
            memory_pressure = self.radix_tree.allocator.used() / max(capacity, 1)
            eviction_prob = min(1.0, memory_pressure)
            l1_reload_avoidance = eviction_prob * self.l1_reload_penalty * prefill_cost
            benefit2 = P_tau * l1_reload_avoidance

            # Total benefit
            benefit = benefit1 + benefit2

            # Cost = memory_holding_cost × τ
            cost = cost_per_sec_ms * tau

            # TTL + write_back utility
            utility_ttl_wb = benefit - cost

            # TTL + write_through utility
            # When evicted, we pay L1 reload penalty
            miss_penalty = (1 - P_tau) * self.l1_reload_penalty * prefill_cost * eviction_prob
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
        # Try program-specific tool durations first
        if program_id in self.program_stats:
            prog_stats = self.program_stats[program_id]
            if tool_name and tool_name in prog_stats.tool_durations:
                durations = prog_stats.tool_durations[tool_name].durations
                if len(durations) >= self.history_threshold:
                    return durations, f"program_tool:{tool_name}"

        # Try program-specific all tools
        if program_id in self.program_stats:
            prog_stats = self.program_stats[program_id]
            all_durations = []
            for tool_stat in prog_stats.tool_durations.values():
                all_durations.extend(tool_stat.durations)
            if len(all_durations) >= self.history_threshold:
                return all_durations, "program_all"

        # Try global tool-specific durations
        if tool_name and tool_name in self.global_tool_durations:
            durations = self.global_tool_durations[tool_name]
            if len(durations) >= self.history_threshold:
                return durations, f"global_tool:{tool_name}"

        # Fallback: all global durations
        all_global = []
        for durations in self.global_tool_durations.values():
            all_global.extend(durations)
        if len(all_global) >= self.history_threshold:
            return all_global, "global_all"

        # Not enough data
        return [], "insufficient_data"

    def _compute_cdf(self, durations: List[float], tau: float) -> float:
        """Compute P(τ, f) - probability that duration <= τ.

        Empirical CDF: fraction of durations <= tau.
        """
        if not durations:
            return 0.0
        count = sum(1 for d in durations if d <= tau)
        return count / len(durations)

    def _generate_tau_grid(self) -> List[float]:
        """Generate grid of τ values for grid search."""
        if self.ttl_grid_points <= 1:
            return [self.default_ttl]
        step = (self.ttl_grid_max - self.ttl_grid_min) / (self.ttl_grid_points - 1)
        return [self.ttl_grid_min + i * step for i in range(self.ttl_grid_points)]


@dataclass
class TTLStats:
    total_pins: int = 0
    total_unpins: int = 0
    expired_pins: int = 0
    ttl_extensions: int = 0
    cache_hits_while_pinned: int = 0
    forced_unpins: int = 0
