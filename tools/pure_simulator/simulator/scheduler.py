"""
RequestScheduler: Schedules and processes requests with the radix tree cache.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .radix_tree import RadixTree, MatchResult
from .ttl_manager import TTLManager
from .tree_node import TreeNode
from .request import Request, ProcessResult, BatchResult, RequestMetrics
from .eviction import create_eviction_policy, EvictionPolicy
from .timing import TimingCalculator, A100_80G, TimeBreakdown


class RequestScheduler:
    """Schedules and processes requests with cache operations."""

    def __init__(
        self,
        radix_tree: RadixTree,
        ttl_manager: Optional[TTLManager] = None,
        eviction_policy: str = "lru",
        schedule_policy: str = "fcfs",
        max_batch_size: int = 32,
        timing: Optional[TimingCalculator] = None,
        l2_cache=None,  # L2CacheManager for load-back
    ):
        self.radix_tree = radix_tree
        self.ttl_manager = ttl_manager
        self.eviction_policy = create_eviction_policy(eviction_policy)
        self.schedule_policy = schedule_policy
        self.max_batch_size = max_batch_size
        self.timing = timing or TimingCalculator(A100_80G)
        self.l2_cache = l2_cache  # L2 cache for spilled entries

        self.pending_queue: List[Request] = []
        self.running_batch: List[Request] = []
        self.completed_requests: List[RequestMetrics] = []

        self.stats = SchedulerStats()
        self.total_time = 0.0  # Track total simulation time

        # Track program-level metrics for correct JCT calculation
        self.program_first_arrival: Dict[str, float] = {}  # program_id -> first arrival time
        self.program_last_finish: Dict[str, float] = {}  # program_id -> last turn finish time
        self.program_completed_turns: Dict[str, int] = {}  # program_id -> completed turns count
        self.program_total_turns: Dict[str, int] = {}  # program_id -> total turns (set externally)

        # Detailed scheduling log for debugging
        self.scheduling_log: List[Dict[str, Any]] = []

    def submit_request(self, req: Request) -> None:
        req.queue_start_time = req.arrival_time
        self.pending_queue.append(req)
        self.stats.total_requests += 1

    def _log_scheduling(
        self,
        req: Request,
        start_time: float,
        finish_time: float,
        hit_type: str,
        is_ttl_hit: bool,
        num_waiting: int,
        process_result: ProcessResult,
        time_breakdown: "TimeBreakdown",
    ) -> None:
        """Log detailed scheduling information for debugging."""
        # Track program-level metrics
        program_id = req.program_id

        # Record first arrival time for this program
        if program_id not in self.program_first_arrival:
            self.program_first_arrival[program_id] = req.arrival_time

        # Update last finish time
        self.program_last_finish[program_id] = finish_time

        # Increment completed turns
        if program_id not in self.program_completed_turns:
            self.program_completed_turns[program_id] = 0
        self.program_completed_turns[program_id] += 1

        # Check if this is the last turn of the program
        total_turns = self.program_total_turns.get(program_id, req.turn_index + 1)
        is_last_turn = (self.program_completed_turns[program_id] >= total_turns)

        # Calculate actual queue wait time
        if req.queue_start_time is not None:
            # Queue wait = time request was picked up - time it entered queue
            queue_wait_ms = (start_time - req.queue_start_time) * 1000
            queue_wait_ms = max(0, queue_wait_ms)  # Non-negative
        else:
            queue_wait_ms = 0

        # Calculate per-program JCT (only for last turn)
        program_jct_ms = None
        if is_last_turn and program_id in self.program_first_arrival:
            first_arrival = self.program_first_arrival[program_id]
            program_jct_ms = (finish_time - first_arrival) * 1000

        log_entry = {
            "event": "schedule",
            "start_time": start_time,
            "finish_time": finish_time,
            "time": finish_time,  # Keep for compatibility
            "rid": req.rid,
            "program_id": req.program_id,
            "turn_index": req.turn_index,
            "arrival_time": req.arrival_time,  # This turn's arrival
            "program_first_arrival": self.program_first_arrival.get(program_id),  # Program's first arrival
            "hit_type": hit_type,
            "is_ttl_hit": is_ttl_hit,
            "is_pinned": is_ttl_hit,
            "is_last_turn": is_last_turn,
            "num_waiting": num_waiting,
            # Token info
            "input_len": req.input_len,
            "output_len": req.output_len,
            "miss_tokens": process_result.new_tokens,
            "matched_tokens": process_result.matched_tokens,
            "l2_tokens": process_result.l2_tokens,
            # Time breakdown
            "prefill_ms": time_breakdown.prefill_ms,
            "decode_ms": time_breakdown.decode_ms,
            "h2d_ms": time_breakdown.h2d_ms,
            "queue_delay_ms": time_breakdown.queue_delay_ms,
            "ttft_ms": time_breakdown.ttft_ms,
            "total_ms": time_breakdown.total_ms,
            # Queue info
            "queue_wait_ms": queue_wait_ms,
            # Per-turn JCT (for debugging)
            "turn_jct_ms": (finish_time - req.arrival_time) * 1000,
            # Per-program JCT (only set for last turn)
            "program_jct_ms": program_jct_ms,
            # Cache state
            "eviction_needed": process_result.eviction_needed,
            "num_evicted": len(process_result.evicted_nodes) if process_result.evicted_nodes else 0,
        }
        self.scheduling_log.append(log_entry)
    
    def process_single(
        self, 
        req: Request, 
        current_time: float,
        num_waiting: int = None
    ) -> BatchResult:
        """Process a single request (for event-driven mode).
        
        Args:
            req: Request to process
            current_time: Current simulation time
            num_waiting: Number of requests waiting (external queue length).
                        If None, uses len(self.pending_queue).
        """
        if self.ttl_manager:
            self.ttl_manager.cleanup_expired(current_time)

        result = BatchResult()

        # Record start time for next turn arrival calculation
        req.start_time = current_time

        # Get TTL info for immediate pinning
        ttl_sec = req.ttl_sec or (self.ttl_manager and self.ttl_manager.default_ttl) or 5.0
        tool_name = req.tool_name

        process_result = self._process_request(req, current_time, ttl_sec=ttl_sec, tool_name=tool_name)
        result.add_result(process_result)

        is_ttl_hit = (
            process_result.hit and
            req.is_tool_call and
            self.ttl_manager and
            self.ttl_manager.is_pinned(req.program_id, current_time)
        )

        hit_type = "L1+L2 Miss"
        l2_tokens = 0

        if is_ttl_hit:
            hit_type = "TTL"
            self.radix_tree.stats.ttl_hits += 1
        elif process_result.l2_hit:
            hit_type = "L2"
            l2_tokens = process_result.l2_tokens
            self.radix_tree.stats.l2_hits += 1
        elif process_result.hit:
            hit_type = "L1"
            self.radix_tree.stats.l1_hits += 1
        else:
            self.radix_tree.stats.cache_misses += 1

        # Get detailed time breakdown
        # Use external waiting queue length if provided (for event-driven mode)
        waiting_count = num_waiting if num_waiting is not None else len(self.pending_queue)
        
        # Calculate total input tokens = matched + new (cached tokens don't need processing)
        total_input_tokens = process_result.matched_tokens + process_result.new_tokens
        
        time_breakdown = self.timing.get_time_breakdown(
            total_input_tokens=total_input_tokens,
            output_tokens=req.output_len,
            hit_type=hit_type,
            num_waiting=waiting_count,
            l2_tokens=l2_tokens,
            l1_tokens=process_result.matched_tokens,  # Pass actual matched tokens
        )
        inf_time = time_breakdown.total_ms / 1000.0

        process_result.inference_time = inf_time
        process_result.prefill_ms = time_breakdown.prefill_ms
        process_result.decode_ms = time_breakdown.decode_ms
        process_result.h2d_ms = time_breakdown.h2d_ms
        process_result.queue_delay_ms = time_breakdown.queue_delay_ms
        process_result.hit_type = hit_type

        # Log scheduling for debugging
        # start_time is current_time (when processing starts)
        # finish_time = current_time + inference_time
        finish_time = current_time + inf_time
        self._log_scheduling(
            req, current_time, finish_time, hit_type, is_ttl_hit,
            len(self.pending_queue), process_result, time_breakdown
        )

        self._record_request_completion(req, process_result, current_time, hit_type, inf_time)

        # TTL pinning handled by caller (main.py _on_request_complete)

        return result

    def process_batch(self, current_time: float) -> BatchResult:
        if self.ttl_manager:
            self.ttl_manager.cleanup_expired(current_time)

        batch = self._get_next_batch()
        result = BatchResult()

        for req in batch:
            process_result = self._process_request(req, current_time)
            result.add_result(process_result)

            is_ttl_hit = (
                process_result.hit and
                req.is_tool_call and
                self.ttl_manager and
                self.ttl_manager.is_pinned(req.program_id, current_time)
            )

            # Determine hit type and L2 load
            hit_type = "L1+L2 Miss"
            l2_tokens = 0

            if is_ttl_hit:
                hit_type = "TTL"
                self.radix_tree.stats.ttl_hits += 1
            elif process_result.l2_hit:
                hit_type = "L2"
                l2_tokens = process_result.l2_tokens
                self.radix_tree.stats.l2_hits += 1
            elif process_result.hit:
                hit_type = "L1"
                self.radix_tree.stats.l1_hits += 1
            else:
                self.radix_tree.stats.cache_misses += 1

            # Get detailed time breakdown
            # Calculate total input tokens = matched + new
            total_input_tokens = process_result.matched_tokens + process_result.new_tokens
            
            time_breakdown = self.timing.get_time_breakdown(
                total_input_tokens=total_input_tokens,
                output_tokens=req.output_len,
                hit_type=hit_type,
                num_waiting=len(self.pending_queue),
                l2_tokens=l2_tokens,
            )
            inf_time = time_breakdown.total_ms / 1000.0

            process_result.inference_time = inf_time
            process_result.prefill_ms = time_breakdown.prefill_ms
            process_result.decode_ms = time_breakdown.decode_ms
            process_result.h2d_ms = time_breakdown.h2d_ms
            process_result.queue_delay_ms = time_breakdown.queue_delay_ms
            process_result.hit_type = hit_type

            # Log scheduling for debugging
            # For batch mode, start_time = finish_time = current_time (batch is instantaneous)
            self._log_scheduling(
                req, current_time, current_time, hit_type, is_ttl_hit,
                len(self.pending_queue), process_result, time_breakdown
            )

            self.total_time += inf_time

            self._record_request_completion(req, process_result, current_time, hit_type, inf_time)

            # Handle re-pinning for cache hits (node was already pinned in _process_request for misses)
            if req.is_tool_call and self.ttl_manager:
                if process_result.last_node and process_result.hit:
                    # Cache hit - check if we need to re-pin
                    node = process_result.last_node
                    need_repin = (
                        not self.ttl_manager.is_pinned(req.program_id, current_time) or
                        node.ttl_expiry_time is None or
                        node.is_expired_at(current_time)
                    )
                    if need_repin:
                        self.ttl_manager.pin_program(
                            req.program_id,
                            node,
                            ttl_sec,
                            current_time,
                            tool_name
                        )
                        self.stats.ttl_pins += 1
                # Note: For misses, node was already pinned in _process_request
            else:
                if process_result.last_node:
                    self.radix_tree.dec_lock_ref(process_result.last_node, current_time)

        return result
    
    def _get_next_batch(self) -> List[Request]:
        if not self.pending_queue:
            return []
        
        batch = self.pending_queue[:self.max_batch_size]
        
        for req in batch:
            if req in self.pending_queue:
                self.pending_queue.remove(req)
        
        return batch
    
    def _process_request(
        self,
        req: Request,
        current_time: float,
        ttl_sec: float = None,
        tool_name: str = None,
    ) -> ProcessResult:
        from .radix_key import RadixKey
        
        req.queue_end_time = current_time
        
        key = RadixKey(token_ids=req.token_ids, extra_key=req.extra_key)
        match_result = self.radix_tree.match_prefix(key)
        
        matched_tokens = match_result.matched_tokens
        new_tokens = len(req.token_ids) - matched_tokens
        
        result = ProcessResult(
            rid=req.rid,
            hit=match_result.hit,
            matched_tokens=matched_tokens,
            new_tokens=new_tokens,
            cached_indices=match_result.cached_indices.copy(),
            last_node=match_result.last_node
        )
        
        # Check if we need eviction: either full miss OR partial hit with insufficient space
        need_eviction = False
        
        if match_result.hit:
            # Partial or full hit - check if we need more space
            if new_tokens > 0:
                available = self.radix_tree.allocator.available()
                if new_tokens > available:
                    # Need to evict to make room for new tokens
                    need_eviction = True
                    self.radix_tree.stats.cache_hits += 1
                    self.radix_tree.stats.l1_hits += 1
                    if match_result.last_node:
                        match_result.last_node.access(current_time)
        else:
            # Full miss
            need_eviction = True
            # Check L2 cache for matching tokens (load-back)
            if self.l2_cache:
                # Perform token-level prefix matching against L2
                l2_match_len, l2_matches = self.l2_cache.match_l2(
                    req.token_ids, req.extra_key
                )
                
                if l2_match_len > 0:
                    result.l2_hit = True
                    result.l2_tokens = l2_match_len
                    result.l2_matches = l2_matches
                    
                    # Recalculate matched_tokens and new_tokens
                    matched_tokens = matched_tokens + l2_match_len
                    new_tokens = len(req.token_ids) - matched_tokens
                    
                    # Check if we still need eviction after L2 match
                    if new_tokens <= self.radix_tree.allocator.available():
                        need_eviction = False
                    # If we need more space, eviction will handle it
            
            self.radix_tree.stats.cache_misses += 1
        
        # Eviction logic
        if need_eviction:
            needed_tokens = new_tokens
            unpin_attempted = False
            
            # Eviction loop - keep evicting until we have enough space
            # Note: We DON'T pin current program nodes here because TTL protects against
            # OTHER programs evicting our nodes, not against our own growth
            while needed_tokens > self.radix_tree.allocator.available():
                tokens_to_free = needed_tokens - self.radix_tree.allocator.available()
                
                # Evict from L1, passing current program_id to protect other programs' pinned nodes
                evicted = self.radix_tree.evict(tokens_to_free, current_time, req.program_id)
                
                if not evicted:
                    # No more unpinned nodes to evict - check if we should do proactive unpin
                    if self.ttl_manager and not unpin_attempted:
                        # All nodes may be pinned - proactively unpin oldest programs
                        # Calculate how many victims we might need
                        pinned_programs = self.ttl_manager.get_pinned_programs_sorted(current_time)
                        if pinned_programs:
                            # Estimate: assume average node size to determine victim count
                            avg_tokens_per_pin = 50  # rough estimate
                            min_victims = max(1, (tokens_to_free + avg_tokens_per_pin - 1) // avg_tokens_per_pin)
                            unpinned = self.ttl_manager.unpin_victims(
                                self.radix_tree,
                                current_time,
                                max_victims=min_victims
                            )
                            unpin_count = len(unpinned)
                            if unpin_count > 0:
                                self.stats.forced_unpins += unpin_count
                                unpin_attempted = True
                                continue  # Retry eviction after unpinning
                    
                    # Cannot evict any more
                    break
                
                # Spill evicted nodes to L2 based on write_policy
                if self.l2_cache:
                    for node in evicted:
                        # Check if node is currently TTL-protected
                        node_is_pinned = (
                            node.is_pinned and
                            node.program_id and
                            self.ttl_manager and
                            self.ttl_manager.is_pinned(node.program_id, current_time)
                        )
                        
                        # Check if node was recently unpinned (forced unpin for eviction)
                        # When a node is forced unpinned due to memory pressure,
                        # it was previously protected by TTL but protection was removed
                        was_recently_unpinned = (
                            hasattr(node, '_force_unpinned_at') and
                            current_time - node._force_unpinned_at < 0.1  # Within 100ms
                        )
                        
                        # Check if TTL has expired
                        ttl_expired = (
                            node.is_pinned and
                            hasattr(node, 'ttl_expiry_time') and
                            node.ttl_expiry_time is not None and
                            current_time >= node.ttl_expiry_time
                        )
                        
                        # Determine if we should spill based on write_policy
                        # write_back: Don't spill if node is currently pinned OR was recently unpinned
                        # (recently unpinned means it was protected by TTL, so don't waste L2 space)
                        if self.l2_cache.write_policy == "write_back":
                            should_spill = not (node_is_pinned or was_recently_unpinned)
                        else:
                            should_spill = True
                        
                        if should_spill:
                            self.l2_cache.spill_to_l2(
                                node_id=node.node_id,
                                token_count=len(node.kv_indices),
                                token_ids=node.key.token_ids,
                                extra_key=node.key.extra_key or "",
                                current_time=current_time,
                            )
                
                result.eviction_needed = True
                result.evicted_nodes = evicted
            
            # After eviction, re-match to get updated matched_tokens and cached_indices
            # (some matched prefixes may have been evicted)
            full_key = RadixKey(
                token_ids=req.token_ids,
                extra_key=req.extra_key
            )
            new_match = self.radix_tree.match_prefix(full_key)
            print(f"  [DEBUG] {req.rid}: after eviction re-match: matched_tokens={new_match.matched_tokens}, hit={new_match.hit}")
            if new_match.last_node:
                print(f"  [DEBUG] last_node: id={new_match.last_node.node_id}, tokens={len(new_match.last_node.key)}, pinned={new_match.last_node.is_pinned}")
            else:
                # Debug: check what nodes exist
                print(f"  [DEBUG] last_node is None, checking radix tree...")
                all_nodes = list(self.radix_tree.nodes.values())
                print(f"  [DEBUG] Total nodes: {len(all_nodes)}")
                for n in all_nodes:
                    if n != self.radix_tree.root:
                        print(f"  [DEBUG] Node {n.node_id}: extra_key={n.key.extra_key}, tokens={n.key.token_ids[:5]}...")
            matched_tokens = new_match.matched_tokens
            new_tokens = len(req.token_ids) - matched_tokens
            cached_indices = new_match.cached_indices
            
            # Load L2 data back to L1 if there's an L2 hit
            # This happens after eviction, so we have space to load the L2 data
            if result.l2_hit and result.l2_matches:
                # Track total tokens being loaded from L2
                l2_load_tokens = 0
                l2_kv_indices = []
                
                for l2_node_id, l2_token_count, l2_token_ids in result.l2_matches:
                    # Remove from L2 first
                    self.l2_cache.remove_entry(req.extra_key, l2_node_id)
                    
                    # Allocate kv_indices for L2 data (in freed space after eviction)
                    try:
                        l2_indices = self.radix_tree.allocator.alloc(l2_token_count)
                        l2_kv_indices.extend(l2_indices)
                        l2_load_tokens += l2_token_count
                        
                        # Re-create node and insert into RadixTree
                        l2_key = RadixKey(token_ids=l2_token_ids, extra_key=req.extra_key)
                        self.radix_tree.insert_node(
                            l2_key,
                            kv_indices=l2_indices,
                            node_id=l2_node_id,
                            program_id=req.program_id
                        )
                        
                        # Update stats
                        self.radix_tree.stats.l2_hits += 1
                    except RuntimeError:
                        # Not enough space even after eviction - skip this L2 entry
                        # This shouldn't happen if eviction worked correctly
                        pass
                
                # Update matched_tokens to include L2 loaded tokens
                matched_tokens += l2_load_tokens
                new_tokens = len(req.token_ids) - matched_tokens
                cached_indices = cached_indices + l2_kv_indices
                
                # Update result
                result.matched_tokens = matched_tokens
                result.cached_indices = cached_indices
            
            # Update result
            result.matched_tokens = matched_tokens
            result.new_tokens = new_tokens
            
            # Try to allocate
            try:
                new_indices = self.radix_tree.allocator.alloc(new_tokens)
            except RuntimeError:
                # Not enough space even after eviction - try to allocate what we can
                # This handles the case where a single request is larger than cache capacity
                available = self.radix_tree.allocator.available()
                if available > 0:
                    new_indices = self.radix_tree.allocator.alloc(available)
                    # Reduce new_tokens to what we actually allocated
                    result.new_tokens = available
                    self.radix_tree.stats.cache_misses += 1
                else:
                    new_indices = []
                    self.radix_tree.stats.cache_misses += 1
            
            remaining_key = RadixKey(
                token_ids=req.token_ids[matched_tokens:],
                extra_key=req.extra_key
            )
            insert_result = self.radix_tree.insert(
                remaining_key,
                kv_indices=new_indices,
                create_new_indices=False
            )

            result.kv_indices = cached_indices + new_indices
            result.node_for_pin = insert_result.node

            # IMMEDIATE PINNING: Pin the node right after insertion
            # This ensures the node is protected BEFORE the next request can evict it
            if self.ttl_manager and ttl_sec and ttl_sec > 0 and result.node_for_pin:
                self.ttl_manager.pin_program(
                    req.program_id,
                    result.node_for_pin,
                    ttl_sec,
                    current_time,
                    tool_name
                )
                self.stats.ttl_pins += 1

        # Handle L2 load-back when L1 misses but L2 hits, and NO eviction is needed
        # This can happen when the new tokens fit in available space
        if result.l2_hit and result.l2_matches and not need_eviction:
            l2_load_tokens = 0
            l2_kv_indices = []
            
            for l2_node_id, l2_token_count, l2_token_ids in result.l2_matches:
                # Remove from L2 first
                self.l2_cache.remove_entry(req.extra_key, l2_node_id)
                
                # Allocate kv_indices for L2 data
                try:
                    l2_indices = self.radix_tree.allocator.alloc(l2_token_count)
                    l2_kv_indices.extend(l2_indices)
                    l2_load_tokens += l2_token_count
                    
                    # Re-create node and insert into RadixTree
                    l2_key = RadixKey(token_ids=l2_token_ids, extra_key=req.extra_key)
                    self.radix_tree.insert_node(
                        l2_key,
                        kv_indices=l2_indices,
                        node_id=l2_node_id,
                        program_id=req.program_id
                    )
                    
                    # Update stats
                    self.radix_tree.stats.l2_hits += 1
                except RuntimeError:
                    # Not enough space - this shouldn't happen if need_eviction is correct
                    pass
            
            # Update matched_tokens and cached_indices
            matched_tokens += l2_load_tokens
            new_tokens = len(req.token_ids) - matched_tokens
            
            # Update result
            result.matched_tokens = matched_tokens
            result.new_tokens = new_tokens
            result.cached_indices = match_result.cached_indices.copy() + l2_kv_indices
            result.kv_indices = result.cached_indices + []
        
        req.prefix_match_len = matched_tokens
        req.cached_indices = result.cached_indices
        req.kv_indices = result.kv_indices
        
        return result
    
    def _record_request_completion(
        self,
        req: Request,
        result: ProcessResult,
        current_time: float,
        hit_type: str = "Miss",
        inf_time: float = 0.0
    ) -> None:
        req.finish_time = current_time + inf_time  # Use finish_time, not current_time
        
        metrics = RequestMetrics(
            rid=req.rid,
            program_id=req.program_id,
            turn_index=req.turn_index,
            arrival_time=req.arrival_time,
            queue_start_time=req.queue_start_time,
            queue_end_time=req.queue_end_time,
            finish_time=req.finish_time,
            input_len=req.input_len,
            output_len=req.output_len,
            prefix_match_len=req.prefix_match_len,
            is_tool_call=req.is_tool_call,
            ttl_sec=req.ttl_sec,
            cache_hit=result.hit
        )
        
        self.completed_requests.append(metrics)
        
        if result.hit:
            self.stats.cache_hit_requests += 1
        self.stats.total_tokens += req.input_len
        self.stats.total_hit_tokens += result.matched_tokens
        self.stats.total_queue_time += metrics.queue_time
    
    def get_queue_length(self) -> int:
        return len(self.pending_queue)
    
    def _pin_current_program_nodes(self, program_id: str, current_time: float, ttl_sec: float) -> None:
        """Pin all nodes of the current program to protect them from eviction.

        This is called BEFORE eviction to ensure the current program's nodes are not
        evicted during this turn. This implements the "eviction protection" benefit of TTL.
        """
        if not self.ttl_manager or not program_id:
            return

        from .ttl_manager import PinnedEntry
        
        # We'll iterate through the tree and pin any nodes with matching program_id
        def pin_nodes_in_subtree(node):
            if node.program_id == program_id and node.kv_indices:
                # Pin this node if it's not already pinned or if we need to refresh
                if not node.is_pinned or node.check_expired(current_time):
                    node.pin(ttl_sec, current_time, program_id)
                    self.radix_tree.inc_lock_ref(node, current_time)
                    
                    # Add to pinned_entries for each pinned node (may have multiple per program)
                    # Store with node_id as key to track multiple nodes
                    entry_key = (program_id, node.node_id)
                    self.ttl_manager.pinned_entries[entry_key] = PinnedEntry(
                        program_id=program_id,
                        node_id=node.node_id,
                        expire_time=node.ttl_expiry_time,
                        ttl_sec=ttl_sec,
                        last_access_time=current_time,
                    )
            for child in node.children.values():
                pin_nodes_in_subtree(child)

        pin_nodes_in_subtree(self.radix_tree.root)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self.stats.total_requests,
            "completed_requests": len(self.completed_requests),
            "pending_requests": len(self.pending_queue),
            "cache_hit_requests": self.stats.cache_hit_requests,
            "ttl_pins": self.stats.ttl_pins,
            "forced_unpins": self.stats.forced_unpins,
        }


@dataclass
class SchedulerStats:
    total_requests: int = 0
    cache_hit_requests: int = 0
    total_tokens: int = 0
    total_hit_tokens: int = 0
    total_queue_time: float = 0.0
    ttl_pins: int = 0
    forced_unpins: int = 0
