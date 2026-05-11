"""
Main: Pure Python KV Cache Simulator entry point.

This is the main simulator class that orchestrates all components.
Supports two modes:
1. Batch mode: Original approach - processes requests by arrival_time order
2. Event-driven mode: Processes requests by completion order with TTL-aware scheduling
"""

import heapq
import json
import random
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .cache_allocator import CacheAllocator
from .radix_key import RadixKey
from .radix_tree import RadixTree
from .ttl_manager import TTLManager
from .eviction import create_eviction_policy
from .request import Request
from .scheduler import RequestScheduler
from .request_log import RequestLogReader, RequestLogEntry
from .stats import CacheStats, SimulationResult, StatisticsCollector
from .l2_cache import L2CacheManager
from .state_manager import StateManager


@dataclass
class SimulatorConfig:
    cache_capacity: int = 100_000
    l2_capacity: int = 500_000  # L2 cache capacity (Host Memory)
    l2_enabled: bool = True    # Enable L2 cache (default enabled)
    default_ttl: float = 5.0
    min_ttl: float = 0.5
    max_ttl: float = 120.0  # Allow longer TTL (2 minutes)
    eviction_policy: str = "lru"
    schedule_policy: str = "fcfs"
    max_batch_size: int = 32
    enable_ttl: bool = True
    enable_adaptive_ttl: bool = True
    page_size: int = 1
    simulation_mode: str = "event_driven"  # "batch" or "event_driven"
    poisson_lambda: float = 5.0  # Lambda for Poisson distribution of first request arrival
    tool_execution_time: float = 0.5  # Default tool execution time for next turn arrival
    # Adaptive TTL parameters
    history_threshold: int = 100  # K=100 per Continuum paper cold-start threshold
    ttl_grid_points: int = 50       # CDF search grid points
    prefill_latency_ms_per_token: float = 0.01  # Prefill latency per token
    # L2 cache parameters
    l2_reload_penalty: float = 0.3   # L2 loading time as fraction of prefill time
    write_policy: str = "write_back"  # "write_back" or "write_through" (论文推荐write_back)
    high_pressure_threshold: float = 0.8  # Memory pressure threshold to disable TTL


class KVCacheSimulator:
    """
    Pure Python KV Cache Simulator.

    Replicates SGLang's KV cache behavior including:
    - Radix prefix cache with prefix matching
    - TTL-based pinning (Continuum)
    - Cache eviction (LRU/LFU/etc.)
    - L1 cache (Host Memory) for load-back
    - Request scheduling

    Supports two simulation modes:
    - batch: Processes requests by arrival_time order (original)
    - event_driven: Processes by completion order with TTL-aware scheduling
    """

    def __init__(
        self,
        cache_capacity: int = 100_000,
        l2_capacity: int = 500_000,
        l2_enabled: bool = False,
        default_ttl: float = 5.0,
        min_ttl: float = 0.5,
        max_ttl: float = 15.0,
        eviction_policy: str = "lru",
        schedule_policy: str = "fcfs",
        max_batch_size: int = 32,
        enable_ttl: bool = True,
        enable_adaptive_ttl: bool = True,
        page_size: int = 1,
        config: Optional[SimulatorConfig] = None,
        simulation_mode: str = "event_driven",
        poisson_lambda: float = 5.0,
        tool_execution_time: float = 0.5,
        random_seed: Optional[int] = None,
        # Hardware config - can pass HardwareConfig or TimingCalculator for accurate timing
        hardware_config = None,  # HardwareConfig or TimingCalculator
        # L2 cache parameters
        l2_reload_penalty: float = 0.3,
        write_policy: str = "write_through",
        high_pressure_threshold: float = 0.8,
        # Timing calculator - if passed, overrides hardware_config timing
        timing_config = None,
        # Adaptive TTL parameters
        history_threshold: int = 100,
        ttl_grid_points: int = 50,
    ):
        if config:
            self.config = config
            l2_enabled = config.l2_enabled
            l2_capacity = config.l2_capacity
            simulation_mode = getattr(config, 'simulation_mode', 'event_driven')
            poisson_lambda = getattr(config, 'poisson_lambda', 5.0)
            tool_execution_time = getattr(config, 'tool_execution_time', 0.5)
        else:
            self.config = SimulatorConfig(
                cache_capacity=cache_capacity,
                l2_capacity=l2_capacity,
                l2_enabled=l2_enabled,
                default_ttl=default_ttl,
                min_ttl=min_ttl,
                max_ttl=max_ttl,
                eviction_policy=eviction_policy,
                schedule_policy=schedule_policy,
                max_batch_size=max_batch_size,
                enable_ttl=enable_ttl,
                enable_adaptive_ttl=enable_adaptive_ttl,
                page_size=page_size,
                simulation_mode=simulation_mode,
                poisson_lambda=poisson_lambda,
                tool_execution_time=tool_execution_time,
                # L2 cache parameters
                l2_reload_penalty=l2_reload_penalty,
                write_policy=write_policy,
                high_pressure_threshold=high_pressure_threshold,
                # Adaptive TTL parameters
                history_threshold=history_threshold,
                ttl_grid_points=ttl_grid_points,
            )

        self.simulation_mode = simulation_mode
        self.poisson_lambda = poisson_lambda  # Mean inter-arrival time (seconds)
        self.jps = 1.0 / poisson_lambda if poisson_lambda > 0 else 0  # Arrivals per second
        self.tool_execution_time = tool_execution_time
        self.random_seed = random_seed

        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        # Reset StateManager for clean simulation
        StateManager.reset()

        self.allocator = CacheAllocator(capacity=self.config.cache_capacity)

        self.ttl_manager = None
        if self.config.enable_ttl:
            self.ttl_manager = TTLManager(
                radix_tree=None,  # Will be set after radix_tree creation
                default_ttl=self.config.default_ttl,
                min_ttl=self.config.min_ttl,
                max_ttl=self.config.max_ttl,
                history_threshold=getattr(self.config, 'history_threshold', 100),  # K=100 per paper
                enable_adaptive_ttl=self.config.enable_adaptive_ttl,
                ttl_grid_points=getattr(self.config, 'ttl_grid_points', 50),
                prefill_latency_ms_per_token=getattr(
                    self.config, 'prefill_latency_ms_per_token', 0.01
                ),
                # L1 cache parameters
                l1_reload_penalty=getattr(self.config, 'l1_reload_penalty', 0.3),
                write_policy=getattr(self.config, 'write_policy', 'write_back'),  # 论文推荐write_back
                high_pressure_threshold=getattr(self.config, 'high_pressure_threshold', 0.8),
            )

        self.radix_tree = RadixTree(
            allocator=self.allocator,
            capacity=self.config.cache_capacity,
            page_size=self.config.page_size,
            eviction_policy=self.config.eviction_policy,
            ttl_manager=self.ttl_manager,
        )

        # Set radix_tree reference in ttl_manager (bidirectional reference)
        if self.ttl_manager:
            self.ttl_manager.radix_tree = self.radix_tree

        # L2 cache for spilled entries
        self.l2_cache = None
        if self.config.l2_enabled:
            self.l2_cache = L2CacheManager(
                capacity_tokens=self.config.l2_capacity,
                write_policy=getattr(self.config, 'write_policy', 'write_through'),
            )

        # Initialize timing calculator
        timing_calc = None
        if timing_config is not None:
            # Use passed-in timing calculator
            timing_calc = timing_config
        elif hardware_config is not None:
            from .timing import TimingCalculator
            if isinstance(hardware_config, TimingCalculator):
                timing_calc = hardware_config
            else:
                timing_calc = TimingCalculator(hw=hardware_config)

        self.scheduler = RequestScheduler(
            radix_tree=self.radix_tree,
            ttl_manager=self.ttl_manager,
            eviction_policy=self.config.eviction_policy,
            schedule_policy=self.config.schedule_policy,
            max_batch_size=self.config.max_batch_size,
            l2_cache=self.l2_cache,
            timing=timing_calc,
        )

        self.collector = StatisticsCollector()
        self.current_time = 0.0
        self.request_reader: Optional[RequestLogReader] = None

        # Event-driven mode state
        self.waiting_queue: List[Request] = []  # All pending requests
        self._active_request_ids: set = set()   # IDs of requests still in queue
        self.program_max_turns: Dict[str, int] = {}  # program_id -> max_turn_index
        self.program_completed_turns: Dict[str, int] = {}  # program_id -> completed_turn_index
        self.entries_by_program: Dict[str, List[RequestLogEntry]] = {}  # program_id -> list of entries

    def load_requests(self, log_path: str) -> int:
        self.request_reader = RequestLogReader(log_path)
        return self.request_reader.load()

    def run(
        self,
        end_time: Optional[float] = None,
        max_requests: Optional[int] = None,
        verbose: bool = False,
        enable_batch: bool = False,
        batch_size: int = 4,
    ) -> SimulationResult:
        if self.simulation_mode == "event_driven":
            return self._run_event_driven(end_time, max_requests, verbose, enable_batch, batch_size)
        else:
            return self._run_batch(end_time, max_requests, verbose)

    def _run_batch(
        self,
        end_time: Optional[float] = None,
        max_requests: Optional[int] = None,
        verbose: bool = False,
    ) -> SimulationResult:
        """Original batch mode: processes requests by arrival_time order."""
        if not self.request_reader:
            raise RuntimeError("No requests loaded. Call load_requests() first.")

        self.current_time = 0.0
        self.collector = StatisticsCollector()
        self.collector.start(self.current_time)

        request_iter = self.request_reader.iter_requests()
        requests_processed = 0

        if verbose:
            print(f"Starting BATCH simulation with {len(self.request_reader)} requests...")

        for entry in request_iter:
            if max_requests and requests_processed >= max_requests:
                break

            req = self._entry_to_request(entry)

            if req.arrival_time > self.current_time:
                time_delta = req.arrival_time - self.current_time
            else:
                time_delta = 0.0

            if time_delta > 0:
                self.current_time += time_delta
                if self.ttl_manager:
                    self.ttl_manager.cleanup_expired(self.current_time)

            if end_time and self.current_time > end_time:
                break

            self.scheduler.submit_request(req)

            if self.scheduler.get_queue_length() > 0:
                batch_result = self.scheduler.process_batch(self.current_time)

                for result in batch_result.results:
                    self.collector.record_request(
                        rid=result.rid,
                        cache_hit=result.hit,
                        matched_tokens=result.matched_tokens,
                        total_tokens=result.matched_tokens + result.new_tokens,
                        ttl_pin=bool(self.ttl_manager and result.node_for_pin),
                        prefill_ms=result.prefill_ms,
                        decode_ms=result.decode_ms,
                        h2d_ms=result.h2d_ms,
                        queue_delay_ms=result.queue_delay_ms,
                        total_ms=result.inference_time * 1000,
                        hit_type=result.hit_type,
                    )

            requests_processed += 1

        self.collector.end(self.current_time)

        self._copy_stats()

        result = self.collector.get_result()
        result.config = self.config.__dict__
        result.duration = self.scheduler.total_time

        if self.l2_cache:
            result.l2_stats = self.l2_cache.get_stats()

        if verbose:
            print(f"Batch simulation complete. Processed {requests_processed} requests "
                  f"in {self.scheduler.total_time:.2f}s")

        return result

    def _run_event_driven(
        self,
        end_time: Optional[float] = None,
        max_requests: Optional[int] = None,
        verbose: bool = False,
        enable_batch: bool = False,
        batch_size: int = 4,
    ) -> SimulationResult:
        """
        Event-driven simulation mode.

        Key differences from batch mode:
        1. Requests are processed by completion order, not arrival order
        2. Next turn's arrival time = current_time + tool_execution_time
        3. TTL-aware priority scheduling: pinned programs get highest priority
        4. Program's first turn arrival time follows Poisson distribution
        5. Batch mode (enable_batch=True): Process multiple requests together
        """
        if not self.request_reader:
            raise RuntimeError("No requests loaded. Call load_requests() first.")

        # Reset state
        self.current_time = 0.0
        self.collector = StatisticsCollector()
        self.collector.start(self.current_time)
        self.waiting_queue = []
        self.program_max_turns = {}
        self.program_completed_turns = defaultdict(int)

        # Statistics
        requests_processed = 0

        # Group entries by program and find max turn
        self.entries_by_program = defaultdict(list)
        for entry in self.request_reader:
            self.entries_by_program[entry.program_id].append(entry)

        for program_id in self.entries_by_program:
            entries = self.entries_by_program[program_id]
            max_turn = max(e.turn_index for e in entries)
            self.program_max_turns[program_id] = max_turn

        if verbose:
            print(f"Starting EVENT-DRIVEN simulation with {len(self.request_reader)} requests...")
            print(f"  Programs: {len(self.entries_by_program)}")
            print(f"  Poisson lambda: {self.poisson_lambda}")
            print(f"  Tool execution time: {self.tool_execution_time}s")
            if enable_batch:
                print(f"  Batch mode enabled: batch_size={batch_size}")

        # Initialize waiting queue with turn 0 of each program
        # Arrival time follows Poisson distribution (only turn 0 has real arrival_time)
        for program_id, entries in self.entries_by_program.items():
            # Find turn 0 entry (it has non-negative arrival_time)
            for entry in entries:
                if entry.turn_index == 0:
                    # Generate Poisson arrival time (overriding the stored value)
                    # poisson_lambda is the RATE (arrivals per second)
                    # Inter-arrival time ~ Exp(rate), so mean = 1/rate seconds
                    # For fast arrival (all at once), use large rate (e.g., 1000 = 0.001s mean)
                    # Note: random.expovariate(rate) gives mean = 1/rate
                    poisson_arrival = random.expovariate(self.poisson_lambda)
                    req = self._entry_to_request(entry)
                    req.arrival_time = poisson_arrival
                    req.program_arrival_time = poisson_arrival  # For FCFS ordering
                    req.queue_start_time = poisson_arrival  # Set when request enters queue (for queue wait tracking)
                    self.waiting_queue.append(req)
                    break

            # Override program_max_turns from entries (this controls next turn generation)
            entries = self.entries_by_program.get(program_id, [])
            max_turn = max(e.turn_index for e in entries) + 1 if entries else 0
            self.program_max_turns[program_id] = max_turn

        # Sort by arrival time
        self.waiting_queue.sort(key=lambda r: r.arrival_time)

        # Start at the earliest arrival time
        if self.waiting_queue:
            self.current_time = self.waiting_queue[0].arrival_time

        # Sort by arrival time
        self.waiting_queue.sort(key=lambda r: r.arrival_time)

        # Initialize pending turns tracking
        self._pending_turns = set()

        # Main event loop
        iteration = 0

        # GPU availability tracking for proper queueing simulation
        gpu_available_time = 0.0  # Time when GPU becomes free
        last_batch_end_time = 0.0  # Track when last batch finished

        # Batch processing state
        self._batch_mode_enabled = enable_batch
        self._batch_size = batch_size
        self._event_queue = []  # Event queue for event-driven batch processing

        while self.waiting_queue and (max_requests is None or requests_processed < max_requests):
            iteration += 1

            # 1. Cleanup expired TTL pins
            if self.ttl_manager:
                self.ttl_manager.cleanup_expired(self.current_time)

            if enable_batch:
                # Batch mode: process ready requests in batches
                # Step 1: Filter ready requests (arrival_time <= current_time)
                ready_requests = [req for req in self.waiting_queue if req.arrival_time <= self.current_time]
                
                if not ready_requests:
                    # No ready requests, advance time to next arrival (same as single mode)
                    if self.waiting_queue:
                        next_arrival = min(req.arrival_time for req in self.waiting_queue)
                        if self.current_time < next_arrival:
                            self.current_time = next_arrival
                    continue
                
                # Step 2: Sort ready requests by arrival_time (FIFO)
                sorted_ready = sorted(ready_requests, key=lambda r: r.arrival_time)
                
                # Step 3: Take min(batch_size, len(ready_requests)) for batch
                actual_batch_size = min(batch_size, len(sorted_ready))
                batch = sorted_ready[:actual_batch_size]
                
                # Remove batch from waiting queue
                for req in batch:
                    self.waiting_queue.remove(req)
                
                # Step 4: Adjust current_time to GPU availability
                if self.current_time < gpu_available_time:
                    self.current_time = gpu_available_time
                
                # Note: We don't advance current_time because all batch requests are ready
                # (arrival_time <= current_time), so no waiting needed
                
                # Calculate batch timing
                batch_result = self._run_batch_event(batch, gpu_available_time)
                
                # Update GPU availability and time
                batch_time = batch_result.inference_time
                gpu_available_time = self.current_time + batch_time
                self.current_time = gpu_available_time
                
                # Record statistics for each request
                for result in batch_result.results:
                    self.collector.record_request(
                        rid=result.rid,
                        cache_hit=result.hit,
                        matched_tokens=result.matched_tokens,
                        total_tokens=result.matched_tokens + result.new_tokens,
                        ttl_pin=bool(self.ttl_manager and result.node_for_pin),
                        prefill_ms=result.prefill_ms,
                        decode_ms=result.decode_ms,
                        h2d_ms=result.h2d_ms,
                        queue_delay_ms=result.queue_delay_ms,
                        total_ms=result.inference_time * 1000,
                        hit_type=result.hit_type,
                    )
                    requests_processed += 1
                
                # Handle completions for batch
                for req_idx, req in enumerate(batch):
                    result = batch_result.results[req_idx] if req_idx < len(batch_result.results) else batch_result.results[0]
                    self._on_request_complete(req, batch_result, result)
            
            else:
                # Single request mode (event-driven)
                # 2. Find ready requests (arrival_time <= current_time)
                ready_requests = [req for req in self.waiting_queue
                               if req.arrival_time <= self.current_time]

                if not ready_requests:
                    # No ready requests, advance time to next arrival
                    next_arrival = min(req.arrival_time for req in self.waiting_queue)
                    self.current_time = next_arrival
                    continue

                # 3. Select next request using TTL-aware priority scheduling
                req = self._select_next_request()

                if req is None:
                    # No schedulable request (memory full, deadlock)
                    if verbose:
                        print(f"  [Warning] No schedulable request at time {self.current_time:.3f}")
                    break

                # 4. Check end_time
                if end_time and self.current_time > end_time:
                    break

                # 5. Handle GPU availability (queueing simulation)
                # If GPU is busy, request must wait
                if self.current_time < gpu_available_time:
                    # GPU is busy, advance time to when GPU becomes available
                    self.current_time = gpu_available_time

                # Calculate queue wait time for this request
                queue_wait_time = max(0.0, gpu_available_time - req.arrival_time) if req.arrival_time > 0 else 0.0

                # 6. Calculate number of waiting requests for timing calculation
                num_waiting = len([r for r in self.waiting_queue if r.arrival_time < self.current_time])

                # 7. Process the request (passing waiting queue length for analysis)
                batch_result = self.scheduler.process_single(req, self.current_time, num_waiting=num_waiting)

                # 8. Record statistics
                if batch_result.results:
                    result = batch_result.results[0]
                    self.collector.record_request(
                        rid=result.rid,
                        cache_hit=result.hit,
                        matched_tokens=result.matched_tokens,
                        total_tokens=result.matched_tokens + result.new_tokens,
                        ttl_pin=bool(self.ttl_manager and result.node_for_pin),
                        prefill_ms=result.prefill_ms,
                        decode_ms=result.decode_ms,
                        h2d_ms=result.h2d_ms,
                        queue_delay_ms=result.queue_delay_ms,
                        total_ms=result.inference_time * 1000,
                        hit_type=result.hit_type,
                    )

                    # 9. Advance time and update GPU availability
                    inference_time = result.inference_time
                    gpu_available_time = self.current_time + inference_time
                    self.current_time = gpu_available_time

                # 10. Handle request completion
                self._on_request_complete(req, batch_result)

                requests_processed += 1

            if verbose and iteration % 100 == 0:
                print(f"  Processed {requests_processed} requests, "
                      f"current_time={self.current_time:.3f}, "
                      f"queue_size={len(self.waiting_queue)}, "
                      f"pinned={len(self.ttl_manager.pinned_entries) if self.ttl_manager else 0}")

        self.collector.end(self.current_time)

        self._copy_stats()

        # Get result with scheduling log
        result = self.collector.get_result_with_log(self.scheduler.scheduling_log)
        result.config = self.config.__dict__
        result.duration = self.current_time

        if self.l2_cache:
            result.l2_stats = self.l2_cache.get_stats()

        if verbose:
            print(f"Event-driven simulation complete. Processed {requests_processed} requests "
                  f"in {self.current_time:.2f}s")

        return result

    def _run_batch_event(self, batch: List[Request], gpu_available_time: float) -> Any:
        """
        Run a batch of requests in event-driven mode.

        This method implements batch processing with proper time advancement.
        Key: We do NOT call process_single here because it advances time internally.
        Instead, we calculate timing directly and let the caller handle time advancement.

        Args:
            batch: List of requests to process
            gpu_available_time: Time when GPU becomes available

        Returns:
            BatchResult with timing information
        """
        # Adjust current_time to GPU availability
        if self.current_time < gpu_available_time:
            self.current_time = gpu_available_time

        # Calculate batch timing using the new batch timing methods
        batch_info = {
            'num_requests': len(batch),
            'avg_input_tokens': sum(len(req.token_ids) for req in batch) / len(batch),
            'total_output_tokens': sum(req.output_len for req in batch),
            'l1_cached_tokens': 0,  # Will be computed in batch_match_cache
            'l2_cached_tokens': 0,
            'is_first_batch': False,
        }

        # Batch cache matching
        match_results = self.scheduler.batch_match_cache(batch, self.current_time)

        batch_info['l1_cached_tokens'] = match_results['total_l1_tokens']
        batch_info['l2_cached_tokens'] = match_results['total_l2_tokens']

        import os
        DEBUG = os.environ.get('DEBUG_BATCH', '0') == '1'
        if DEBUG:
            print(f"\n[DEBUG _run_batch_event] batch_size={len(batch)} l1_tokens={match_results['total_l1_tokens']} l2_tokens={match_results['total_l2_tokens']}")
            for i, r in enumerate(match_results['results']):
                print(f"  [DEBUG] req {batch[i].rid}: l1={r['l1_matched']} l2={r['l2_matched']} new={r['new_tokens']} total={r['total_tokens']}")
            print(f"  [DEBUG] allocator before eviction: used={self.radix_tree.allocator.used()}, available={self.radix_tree.allocator.available()}")

        # Batch eviction
        self.scheduler.batch_evict_for_requests(batch, match_results['results'], self.current_time)

        if DEBUG:
            print(f"  [DEBUG] allocator after eviction: used={self.radix_tree.allocator.used()}, available={self.radix_tree.allocator.available()}")
            print(f"  [DEBUG] radix tree nodes: {len(self.radix_tree.nodes)}")

        # Batch L2 loading
        l2_load_time, l2_hits = self.scheduler.batch_load_l2_to_l1(batch, match_results['results'])

        # Calculate batch timing
        batch_times = self.scheduler.timing.calculate_batch_time(batch_info)

        # Process each request and record results
        # Key fix: We do NOT call process_single here because it advances time
        # internally via StateManager.step_global_clock. Instead, we directly
        # calculate timing and update cache state without time advancement.
        from .request import BatchResult as ReqBatchResult, ProcessResult

        results = []
        for req_idx, req in enumerate(batch):
            # Get the updated match results from batch_evict_for_requests
            # (which includes L2->L1 restoration)
            batch_result = match_results['results'][req_idx]
            l1_matched_from_batch = batch_result['l1_matched']
            l2_matched_from_batch = batch_result['l2_matched']
            
            # Directly call _process_request without time advancement
            ttl_sec = req.ttl_sec or (self.ttl_manager.default_ttl if self.ttl_manager else 5.0)
            tool_name = req.tool_name

            if self.ttl_manager:
                self.ttl_manager.cleanup_expired(self.current_time)

            # skip_eviction=True because eviction was already done by batch_evict_for_requests
            # Pass l2_matched_override to handle L2 restoration properly
            process_result = self.scheduler._process_request(
                req, self.current_time, ttl_sec=ttl_sec, tool_name=tool_name,
                skip_eviction=True, l2_matched_override=l2_matched_from_batch
            )

            # Determine hit type using the updated batch_result (which has L2->L1 restoration applied)
            # Note: Stats are already updated by _update_batch_hit_stats in batch_evict_for_requests
            is_ttl_hit = (
                l1_matched_from_batch > 0 and
                req.is_tool_call and
                self.ttl_manager and
                self.ttl_manager.is_pinned(req.program_id, self.current_time)
            )

            hit_type = "L1+L2 Miss"
            l2_tokens = 0
            l1_tokens = 0

            if is_ttl_hit:
                hit_type = "TTL"
                l1_tokens = l1_matched_from_batch
                # Stats already updated by _update_batch_hit_stats, just recount as TTL
                self.radix_tree.stats.ttl_hits += 1
                self.radix_tree.stats.l1_hits -= 1  # Correct the miscount
            elif l2_matched_from_batch > 0:
                # L2 data was restored to L1 during batch_evict_for_requests
                # But we still report it as L2 hit for proper accounting
                hit_type = "L2"
                l2_tokens = l2_matched_from_batch
                l1_tokens = l1_matched_from_batch
            elif l1_matched_from_batch > 0:
                hit_type = "L1"
                l1_tokens = l1_matched_from_batch
            else:
                hit_type = "L1+L2 Miss"
                l1_tokens = 0

            # Update process_result with values from batch_result (which has L2->L1 restoration)
            process_result.matched_tokens = l1_matched_from_batch
            process_result.new_tokens = batch_result['new_tokens']
            
            # Get time breakdown
            total_input_tokens = l1_matched_from_batch + batch_result['new_tokens']
            time_breakdown = self.scheduler.timing.get_time_breakdown(
                total_input_tokens=total_input_tokens,
                output_tokens=req.output_len,
                hit_type=hit_type,
                num_waiting=0,
                l2_tokens=l2_tokens,
                l1_tokens=l1_tokens,
            )

            inf_time = time_breakdown.total_ms / 1000.0

            # Set timing on result (don't call step_global_clock here)
            process_result.inference_time = inf_time
            process_result.prefill_ms = time_breakdown.prefill_ms
            process_result.decode_ms = time_breakdown.decode_ms
            process_result.h2d_ms = time_breakdown.h2d_ms
            process_result.queue_delay_ms = time_breakdown.queue_delay_ms
            process_result.hit_type = hit_type

            # Log scheduling with batch start time
            finish_time = self.current_time + inf_time
            self.scheduler._log_scheduling(
                req, self.current_time, finish_time, hit_type, is_ttl_hit,
                0, process_result, time_breakdown
            )

            self.scheduler._record_request_completion(req, process_result, self.current_time, hit_type, inf_time)

            # Handle TTL pinning for cache hits
            if req.is_tool_call and self.ttl_manager:
                if process_result.last_node and process_result.hit:
                    node = process_result.last_node
                    need_repin = (
                        not self.ttl_manager.is_pinned(req.program_id, self.current_time) or
                        node.ttl_expiry_time is None or
                        node.is_expired_at(self.current_time)
                    )
                    if need_repin:
                        self.ttl_manager.pin_program(
                            req.program_id,
                            node,
                            ttl_sec,
                            self.current_time,
                            tool_name
                        )
                        self.scheduler.stats.ttl_pins += 1
                else:
                    if process_result.last_node:
                        self.radix_tree.dec_lock_ref(process_result.last_node, self.current_time)
            else:
                if process_result.last_node:
                    self.radix_tree.dec_lock_ref(process_result.last_node, self.current_time)

            results.append(process_result)

        # Calculate batch time (max of all request times = parallel execution)
        # Add L2 load time for L2->L1 migration cost
        total_time = 0.0
        for r in results:
            if hasattr(r, 'inference_time'):
                total_time = max(total_time, r.inference_time)
        
        # Add L2->L1 migration time to batch total
        l2_load_seconds = l2_load_time / 1000.0 if l2_load_time else 0.0
        total_time += l2_load_seconds

        return ReqBatchResult(
            results=results,
            inference_time=total_time,
            total_new_tokens=sum(r.new_tokens for r in results if hasattr(r, 'new_tokens')),
            total_cached_tokens=match_results['total_l1_tokens'] + match_results['total_l2_tokens'],
            eviction_count=0,
        )

    def _select_next_request(self) -> Optional[Request]:
        """
        Select the next request using TTL-aware priority scheduling.

        Returns the selected request, or None if no request can be scheduled.

        Priority rules (per Continuum paper Section 4.3):
        1. Preempted status (被抢占的请求优先) - HIGHEST priority
        2. Pinned status (TTL 窗口内的 pinned 请求)
        3. FCFS by program arrival time (earliest first)
        """
        if not self.waiting_queue:
            return None

        # Use pre-sorted waiting queue by arrival_time
        # Fast path: check first request
        first_req = self.waiting_queue[0]
        if first_req.arrival_time > self.current_time:
            # Advance time to next arrival
            self.current_time = first_req.arrival_time

        # Find ready requests - use binary search for efficiency
        # waiting_queue is sorted by arrival_time
        ready_idx = 0
        for i, req in enumerate(self.waiting_queue):
            if req.arrival_time <= self.current_time:
                ready_idx = i + 1
            else:
                break

        if ready_idx == 0:
            return None

        ready_requests = self.waiting_queue[:ready_idx]

        # Group by priority per paper:
        # Priority 1: Preempted requests (论文 Section 4.3)
        # Priority 2: Pinned programs (TTL not expired)
        # Priority 3: FCFS by program_arrival_time
        preempted = []
        pinned = []
        unpinned = []

        for req in ready_requests:
            if getattr(req, 'is_preempted', False):
                preempted.append(req)
            elif (self.ttl_manager and
                  self.ttl_manager.is_pinned(req.program_id, self.current_time) and
                  self.ttl_manager.get_remaining_ttl(req.program_id, self.current_time) > 0):
                pinned.append(req)
            else:
                unpinned.append(req)

        # Priority 1: preempted (FCFS by program_arrival_time)
        if preempted:
            preempted.sort(key=lambda r: r.program_arrival_time)
            selected = preempted[0]
        # Priority 2: pinned (FCFS by program_arrival_time)
        elif pinned:
            pinned.sort(key=lambda r: r.program_arrival_time)
            selected = pinned[0]
        # Priority 3: unpinned (FCFS by program_arrival_time)
        elif unpinned:
            unpinned.sort(key=lambda r: r.program_arrival_time)
            selected = unpinned[0]
        else:
            return None

        # Remove selected from waiting_queue
        self._active_request_ids.discard(selected.rid)
        # Remove selected from anywhere in the queue
        self.waiting_queue = [r for r in self.waiting_queue if r.rid != selected.rid]
        # Remove from pending_turns tracking
        if hasattr(self, '_pending_turns'):
            self._pending_turns.discard(selected.rid)
        return selected

    def _advance_time_to_next_event(self) -> None:
        """Advance simulation time to the next relevant event.

        This handles the timing correctly:
        1. Advance to next request arrival time if GPU is idle
        2. Advance to GPU availability if GPU is busy
        3. Process pinned requests immediately when they become ready
        """
        if not self.waiting_queue:
            return

        # Check if there are pinned requests ready to be scheduled
        pinned_ready = []
        for req in self.waiting_queue:
            if req.arrival_time <= self.current_time:
                if (self.ttl_manager and
                    self.ttl_manager.is_pinned(req.program_id, self.current_time) and
                    self.ttl_manager.get_remaining_ttl(req.program_id, self.current_time) > 0):
                    pinned_ready.append(req)

        # If there are pinned requests ready, we don't need to advance time
        # They will be scheduled immediately in the next iteration
        if pinned_ready:
            return

        # No pinned requests ready, advance time normally
        next_arrival = min(req.arrival_time for req in self.waiting_queue)
        if next_arrival > self.current_time:
            self.current_time = next_arrival

    def _on_request_complete(self, req: Request, batch_result, result=None) -> None:
        """
        Handle request completion:
        1. Calculate adaptive TTL (using historical CDF only)
        2. Pin the program if it's a tool call (TTL mechanism)
        3. Record tool execution for adaptive TTL (for NEXT request's CDF)
        4. Generate next turn request if available

        IMPORTANT: record_tool_execution MUST come after calc_adaptive_ttl
        to ensure TTL decisions use only historical data, not current request.
        """
        if not batch_result.results:
            return

        # Use provided result or default to first result
        if result is None:
            result = batch_result.results[0]

        # Step 1: Calculate adaptive TTL if enabled
        # This uses historical CDF, NOT including current request's duration
        ttl_sec = req.ttl_sec or self.config.default_ttl
        strategy = 'disabled'  # Track actual strategy for prediction recording
        
        if self.ttl_manager and self.config.enable_adaptive_ttl:
            # Calculate adaptive TTL using paper's utility model
            miss_tokens = result.new_tokens
            node_size = len(req.token_ids) if req.token_ids else 0
            total_turns = self.program_max_turns.get(req.program_id, 1)

            # Get L2 state from simulator config
            l2_enabled = self.config.l2_enabled
            l2_reload_penalty = getattr(self.config, 'l2_reload_penalty', 0.3)

            adaptive_ttl, strategy, cdf_samples = self.ttl_manager.calc_adaptive_ttl(
                program_id=req.program_id,
                tool_name=req.tool_name,
                miss_tokens=miss_tokens,
                node_size=node_size,
                turn_index=req.turn_index,
                total_turns=total_turns,
                current_time=self.current_time,
                l2_enabled=l2_enabled,
                l2_reload_penalty=l2_reload_penalty,
            )
            ttl_sec = adaptive_ttl

        # Step 2: Pin the program based on calculated TTL
        # Note: Pinning is now done immediately in scheduler._process_request
        # This only handles re-pinning for cache hits where the node wasn't pinned
        if self.ttl_manager and ttl_sec > 0:
            if result.last_node and result.hit:
                # Re-pin on hit if TTL expired
                node = result.last_node
                need_repin = (
                    not self.ttl_manager.is_pinned(req.program_id, self.current_time) or
                    node.ttl_expiry_time is None or
                    node.is_expired_at(self.current_time)
                )
                if need_repin:
                    self.ttl_manager.pin_program(
                        req.program_id,
                        node,
                        ttl_sec,
                        self.current_time,
                        req.tool_name
                    )
                    self.scheduler.stats.ttl_pins += 1
            # For misses, the node was already pinned in scheduler._process_request

        # Calculate idle gap for later use (time until next turn arrives)
        next_turn_index = req.turn_index + 1
        idle_gap = 0.0
        if next_turn_index <= self.program_max_turns.get(req.program_id, 0):
            actual_duration = req.actual_tool_duration if req.actual_tool_duration else self.tool_execution_time
            idle_gap = actual_duration

        # Calculate queueing delay (waiting time before execution started)
        # Queueing delay = time from arrival to start of execution
        queueing_delay = 0.0
        if req.start_time is not None and req.arrival_time is not None:
            queueing_delay = max(0.0, req.start_time - req.arrival_time)

        # Track pending turns to avoid duplicates
        if not hasattr(self, '_pending_turns'):
            self._pending_turns = set()

        # Step 3: Generate next turn request (MUST be before record_tool_execution)
        # The next request needs correct arrival_time which uses actual_tool_duration
        if next_turn_index <= self.program_max_turns.get(req.program_id, 0):
            # Check for duplicate before adding
            next_rid = f"{req.program_id}_turn_{next_turn_index}"
            if next_rid not in self._pending_turns:
                # Find the next turn entry from entries_by_program
                for entry in self.entries_by_program.get(req.program_id, []):
                    if entry.turn_index == next_turn_index:
                        next_req = self._entry_to_request(entry)
                        # Next turn arrives at: finish_time + actual tool execution time
                        actual_duration = req.actual_tool_duration if req.actual_tool_duration else self.tool_execution_time
                        next_arrival_base = req.finish_time + actual_duration
                        next_req.arrival_time = next_arrival_base
                        next_req.program_arrival_time = req.program_arrival_time  # Keep original ordering
                        # Set queue_start_time when request enters queue (for proper queue wait tracking)
                        next_req.queue_start_time = next_arrival_base
                        self.waiting_queue.append(next_req)
                        self._pending_turns.add(next_rid)
                        break

        # Step 4: Record tool execution for adaptive TTL CDF calculation
        # IMPORTANT: This MUST be last, so CDF is updated AFTER TTL calculation
        # The recorded duration will be used for the NEXT request's TTL decision
        if self.ttl_manager and req.tool_name:
            actual_dur = req.actual_tool_duration if req.actual_tool_duration else (req.ttl_sec or self.config.default_ttl)
            self.ttl_manager.record_tool_execution(
                program_id=req.program_id,
                tool_name=req.tool_name,
                duration=actual_dur,
                idle_gap=idle_gap,
                queueing_delay=queueing_delay
            )
            
            # Step 5: Record TTL prediction vs actual for analysis
            # Use cdf_samples from calc_adaptive_ttl return value (correct)
            cache_hit = result.hit if result else False
            
            # Get memory pressure
            capacity = self.radix_tree.allocator.capacity
            memory_pressure = self.radix_tree.allocator.used() / max(capacity, 1) if capacity > 0 else 0.0
            
            self.ttl_manager.record_prediction(
                program_id=req.program_id,
                tool_name=req.tool_name,
                predicted_ttl=ttl_sec,
                actual_duration=actual_dur,
                cdf_samples=cdf_samples,
                strategy=strategy,  # Use actual strategy from calc_adaptive_ttl
                cache_hit=cache_hit,
                current_time=self.current_time,
                memory_pressure=memory_pressure
            )

    def _copy_stats(self) -> None:
        """Copy statistics from components to collector."""
        self.collector.cache_stats.tokens_cached = self.radix_tree.allocator.used()
        self.collector.cache_stats.tokens_evicted = self.radix_tree.stats.tokens_evicted
        # cache_hits = l1_hits + l2_hits + ttl_hits (all types of hits)
        self.collector.cache_stats.cache_hits = (
            self.radix_tree.stats.l1_hits +
            self.radix_tree.stats.l2_hits +
            self.radix_tree.stats.ttl_hits
        )
        self.collector.cache_stats.cache_misses = self.radix_tree.stats.cache_misses
        self.collector.cache_stats.l1_hits = self.radix_tree.stats.l1_hits
        self.collector.cache_stats.l2_hits = self.radix_tree.stats.l2_hits
        self.collector.cache_stats.ttl_hits = self.radix_tree.stats.ttl_hits
        self.collector.cache_stats.prefix_hits = self.radix_tree.stats.prefix_hits
        self.collector.cache_stats.full_hits = self.radix_tree.stats.full_hits
        self.collector.cache_stats.ttl_pins = self.scheduler.stats.ttl_pins
        self.collector.cache_stats.ttl_expired = self.radix_tree.stats.ttl_expired

    def _entry_to_request(self, entry: RequestLogEntry) -> Request:
        ttl_sec = None
        # Only use TTL if explicitly enabled in config
        if self.config.enable_ttl and self.ttl_manager:
            # Try to extract TTL from extra_key
            if entry.extra_key:
                ttl_sec = RequestLogReader.extract_ttl(entry.extra_key)
            # Fall back to default TTL for tool calls
            if ttl_sec is None and entry.is_tool_call:
                ttl_sec = self.config.default_ttl

        return Request(
            rid=entry.rid,
            program_id=entry.program_id,
            turn_index=entry.turn_index,
            token_ids=entry.token_ids,
            input_len=entry.input_len,
            output_len=entry.output_len,
            is_tool_call=entry.is_tool_call,
            tool_name=entry.tool_name,
            arrival_time=entry.arrival_time,
            ttl_sec=ttl_sec,
            extra_key=entry.extra_key,
            actual_tool_duration=entry.actual_tool_duration,
        )

    def __repr__(self) -> str:
        return (f"KVCacheSimulator("
                f"capacity={self.config.cache_capacity}, "
                f"ttl={self.config.default_ttl}s, "
                f"policy={self.config.eviction_policy}, "
                f"mode={self.simulation_mode})")


def run_simulation(
    log_path: str,
    output_path: Optional[str] = None,
    cache_capacity: int = 100_000,
    default_ttl: float = 5.0,
    eviction_policy: str = "lru",
    verbose: bool = False,
    enable_ttl: bool = True,
) -> SimulationResult:
    sim = KVCacheSimulator(
        cache_capacity=cache_capacity,
        default_ttl=default_ttl,
        eviction_policy=eviction_policy,
        enable_ttl=enable_ttl,
    )

    count = sim.load_requests(log_path)
    if verbose:
        print(f"Loaded {count} requests from {log_path}")

    result = sim.run(verbose=verbose)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pure Python KV Cache Simulator")
    parser.add_argument("log_path", help="Path to request log file")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("-c", "--capacity", type=int, default=100000)
    parser.add_argument("-t", "--ttl", type=float, default=5.0)
    parser.add_argument("-e", "--eviction", default="lru",
                       choices=["lru", "lfu", "fifo", "continuum"])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--mode", default="batch", choices=["batch", "event_driven"],
                       help="Simulation mode")

    args = parser.parse_args()

    result = run_simulation(
        log_path=args.log_path,
        output_path=args.output,
        cache_capacity=args.capacity,
        default_ttl=args.ttl,
        eviction_policy=args.eviction,
        verbose=args.verbose,
    )

    print(result.summary())
