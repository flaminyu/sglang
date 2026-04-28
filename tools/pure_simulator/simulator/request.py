"""
Request: Represents a request in the KV cache simulator.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class Request:
    rid: str
    program_id: str
    turn_index: int
    token_ids: List[int]
    input_len: int
    output_len: int
    
    is_tool_call: bool = False
    tool_name: Optional[str] = None
    arrival_time: float = 0.0
    
    prefix_match_len: int = 0
    cached_indices: List[int] = field(default_factory=list)
    kv_indices: List[int] = field(default_factory=list)
    
    ttl_sec: Optional[float] = None
    is_pinned: bool = False
    
    queue_start_time: Optional[float] = None
    queue_end_time: Optional[float] = None
    finish_time: Optional[float] = None

    # Event-driven mode: when this turn started processing
    start_time: Optional[float] = None

    extra_key: Optional[str] = None
    
    # Event-driven mode: original program arrival time (for FCFS ordering)
    program_arrival_time: float = 0.0

    # Real tool execution time (seconds) - used for TTL CDF calculation
    actual_tool_duration: Optional[float] = None

    def __post_init__(self):
        if self.input_len == 0:
            self.input_len = len(self.token_ids)
    
    @property
    def cache_hit(self) -> bool:
        return self.prefix_match_len > 0
    
    @property
    def queue_time(self) -> float:
        if self.queue_start_time is None or self.queue_end_time is None:
            return 0.0
        return self.queue_end_time - self.queue_start_time


@dataclass
class ProcessResult:
    rid: str
    hit: bool = False
    matched_tokens: int = 0
    new_tokens: int = 0
    cached_indices: List[int] = field(default_factory=list)
    kv_indices: List[int] = field(default_factory=list)
    last_node: Optional[Any] = None
    node_for_pin: Optional[Any] = None
    eviction_needed: bool = False
    evicted_nodes: List[Any] = field(default_factory=list)
    inference_time: float = 0.0  # Total time spent on this request
    # L2 cache fields
    l2_hit: bool = False
    l2_tokens: int = 0
    l2_matches: List[Tuple[int, int, List[int]]] = field(default_factory=list)  # (node_id, token_count, token_ids)

    # Detailed time breakdown (set by timing calculator)
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    h2d_ms: float = 0.0  # L2 load time
    queue_delay_ms: float = 0.0
    hit_type: str = "L1+L2 Miss"


@dataclass
class BatchResult:
    results: List[ProcessResult] = field(default_factory=list)
    total_new_tokens: int = 0
    total_cached_tokens: int = 0
    eviction_count: int = 0
    inference_time: float = 0.0  # Time spent on the batch
    
    def add_result(self, result: ProcessResult):
        self.results.append(result)
        self.total_new_tokens += result.new_tokens
        self.total_cached_tokens += result.matched_tokens
        if result.eviction_needed:
            self.eviction_count += 1


@dataclass
class RequestMetrics:
    rid: str
    program_id: str
    turn_index: int
    arrival_time: float
    queue_start_time: Optional[float]
    queue_end_time: Optional[float]
    finish_time: Optional[float]
    input_len: int
    output_len: int
    prefix_match_len: int
    is_tool_call: bool
    ttl_sec: Optional[float]
    cache_hit: bool
    
    @property
    def queue_time(self) -> float:
        if self.queue_start_time is None or self.queue_end_time is None:
            return 0.0
        return self.queue_end_time - self.queue_start_time
