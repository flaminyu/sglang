"""
Statistics: Collects and reports simulation statistics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CacheStats:
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    # Detailed hit type breakdown (matching sglang-simulator)
    l1_hits: int = 0
    l2_hits: int = 0
    ttl_hits: int = 0
    prefix_hits: int = 0
    full_hits: int = 0
    tokens_cached: int = 0
    tokens_evicted: int = 0
    ttl_pins: int = 0
    ttl_expired: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.hit_rate,
            # Detailed hit breakdown
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "ttl_hits": self.ttl_hits,
            "prefix_hits": self.prefix_hits,
            "full_hits": self.full_hits,
            # Token stats
            "tokens_cached": self.tokens_cached,
            "tokens_evicted": self.tokens_evicted,
            "ttl_pins": self.ttl_pins,
            "ttl_expired": self.ttl_expired,
        }


@dataclass
class SchedulerStats:
    total_queue_time: float = 0.0
    avg_queue_time: float = 0.0
    avg_latency: float = 0.0
    throughput: float = 0.0
    forced_unpins: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queue_time": self.total_queue_time,
            "avg_queue_time": self.avg_queue_time,
            "avg_latency": self.avg_latency,
            "throughput": self.throughput,
            "forced_unpins": self.forced_unpins,
        }


@dataclass
class SimulationResult:
    cache_stats: CacheStats
    scheduler_stats: SchedulerStats
    config: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    l2_stats: Optional[Dict[str, Any]] = None
    scheduling_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "cache_stats": self.cache_stats.to_dict(),
            "scheduler_stats": self.scheduler_stats.to_dict(),
            "config": self.config,
            "duration": self.duration,
        }
        if self.l2_stats:
            result["l2_stats"] = self.l2_stats
        return result

    def summary(self) -> str:
        return (
            f"Simulation Result:\n"
            f"  Hit Rate: {self.cache_stats.hit_rate:.1%}\n"
            f"  Cache Hits: {self.cache_stats.cache_hits}\n"
            f"  Cache Misses: {self.cache_stats.cache_misses}\n"
            f"  Tokens Cached: {self.cache_stats.tokens_cached}\n"
            f"  Tokens Evicted: {self.cache_stats.tokens_evicted}\n"
            f"  TTL Pins: {self.cache_stats.ttl_pins}\n"
        )


@dataclass
class TimeBreakdownStats:
    """Aggregated time breakdown statistics."""
    total_prefill_ms: float = 0.0
    total_decode_ms: float = 0.0
    total_h2d_ms: float = 0.0
    total_queue_delay_ms: float = 0.0
    total_ttft_ms: float = 0.0
    total_inference_ms: float = 0.0
    count: int = 0

    def add(self, prefill_ms: float, decode_ms: float, h2d_ms: float,
            queue_delay_ms: float, ttft_ms: float, total_ms: float):
        self.total_prefill_ms += prefill_ms
        self.total_decode_ms += decode_ms
        self.total_h2d_ms += h2d_ms
        self.total_queue_delay_ms += queue_delay_ms
        self.total_ttft_ms += ttft_ms
        self.total_inference_ms += total_ms
        self.count += 1

    @property
    def avg_prefill_ms(self) -> float:
        return self.total_prefill_ms / self.count if self.count > 0 else 0.0

    @property
    def avg_decode_ms(self) -> float:
        return self.total_decode_ms / self.count if self.count > 0 else 0.0

    @property
    def avg_h2d_ms(self) -> float:
        return self.total_h2d_ms / self.count if self.count > 0 else 0.0

    @property
    def avg_queue_delay_ms(self) -> float:
        return self.total_queue_delay_ms / self.count if self.count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "total_prefill_ms": self.total_prefill_ms,
            "avg_prefill_ms": self.avg_prefill_ms,
            "total_decode_ms": self.total_decode_ms,
            "avg_decode_ms": self.avg_decode_ms,
            "total_h2d_ms": self.total_h2d_ms,
            "avg_h2d_ms": self.avg_h2d_ms,
            "total_queue_delay_ms": self.total_queue_delay_ms,
            "avg_queue_delay_ms": self.avg_queue_delay_ms,
            "total_inference_ms": self.total_inference_ms,
        }


class StatisticsCollector:
    """Collects statistics during simulation."""

    def __init__(self):
        self.cache_stats = CacheStats()
        self.scheduler_stats = SchedulerStats()
        self.time_breakdown = TimeBreakdownStats()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.request_metrics: List[Dict[str, Any]] = []

    def start(self, current_time: float = 0.0):
        self.start_time = current_time

    def end(self, current_time: float = 0.0):
        self.end_time = current_time
        if self.duration > 0:
            self.scheduler_stats.throughput = self.cache_stats.total_requests / self.duration

    @property
    def duration(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def record_request(
        self,
        rid: str,
        cache_hit: bool,
        matched_tokens: int,
        total_tokens: int,
        is_tool_call: bool = False,
        ttl_pin: bool = False,
        queue_time: float = 0.0,
        latency: float = 0.0,
        prefill_ms: float = 0.0,
        decode_ms: float = 0.0,
        h2d_ms: float = 0.0,
        queue_delay_ms: float = 0.0,
        ttft_ms: float = 50.0,
        total_ms: float = 0.0,
        hit_type: str = "Miss",
    ):
        self.cache_stats.total_requests += 1
        if cache_hit:
            self.cache_stats.cache_hits += 1
        else:
            self.cache_stats.cache_misses += 1

        if ttl_pin:
            self.cache_stats.ttl_pins += 1

        self.scheduler_stats.total_queue_time += queue_time

        # Record time breakdown
        self.time_breakdown.add(prefill_ms, decode_ms, h2d_ms, queue_delay_ms, ttft_ms, total_ms)

        self.request_metrics.append({
            "rid": rid,
            "cache_hit": cache_hit,
            "hit_type": hit_type,
            "matched_tokens": matched_tokens,
            "total_tokens": total_tokens,
            "is_tool_call": is_tool_call,
            "ttl_pin": ttl_pin,
            "queue_time": queue_time,
            "latency": latency,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "h2d_ms": h2d_ms,
            "queue_delay_ms": queue_delay_ms,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
        })

    def compute_averages(self):
        n = len(self.request_metrics)
        if n > 0:
            total_queue = sum(m["queue_time"] for m in self.request_metrics)
            total_latency = sum(m["latency"] for m in self.request_metrics)
            self.scheduler_stats.avg_queue_time = total_queue / n
            self.scheduler_stats.avg_latency = total_latency / n

    def get_result(self) -> SimulationResult:
        self.compute_averages()
        return SimulationResult(
            cache_stats=self.cache_stats,
            scheduler_stats=self.scheduler_stats,
            duration=self.duration
        )

    def get_result_with_log(self, scheduling_log: List[Dict[str, Any]]) -> SimulationResult:
        """Get result including scheduling log."""
        self.compute_averages()
        result = SimulationResult(
            cache_stats=self.cache_stats,
            scheduler_stats=self.scheduler_stats,
            duration=self.duration,
            scheduling_log=scheduling_log,
        )
        result.time_breakdown = self.time_breakdown
        return result

    def compute_program_jct(self, scheduling_log: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute per-program JCT from scheduling log.

        Returns:
            Dict mapping program_id -> JCT in milliseconds
        """
        program_jcts = {}

        for entry in scheduling_log:
            program_id = entry.get('program_id')
            program_jct = entry.get('program_jct_ms')

            if program_id and program_jct is not None:
                # Only record for last turn of each program
                if program_id not in program_jcts:
                    program_jcts[program_id] = program_jct

        return program_jcts

    def compute_program_jct_stats(self, scheduling_log: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute per-program JCT statistics.

        Returns:
            Dict with avg_jct, p50_jct, p95_jct, p99_jct, max_jct
        """
        import statistics
        program_jcts = self.compute_program_jct(scheduling_log)

        if not program_jcts:
            return {}

        jct_values = sorted(program_jcts.values())
        n = len(jct_values)

        return {
            'avg_jct': statistics.mean(jct_values),
            'p50_jct': jct_values[n // 2],
            'p95_jct': jct_values[int(n * 0.95)] if n >= 20 else jct_values[-1],
            'p99_jct': jct_values[int(n * 0.99)] if n >= 100 else jct_values[-1],
            'max_jct': max(jct_values),
            'min_jct': min(jct_values),
            'num_programs': n,
        }
