"""
Pure Python KV Cache Simulator

A standalone simulator that replicates SGLang's KV cache behavior.
"""

from .simulator.cache_allocator import CacheAllocator
from .simulator.radix_key import RadixKey
from .simulator.radix_tree import RadixTree, MatchResult, InsertResult, CacheStats
from .simulator.tree_node import TreeNode
from .simulator.ttl_manager import TTLManager, PinnedEntry, TTLStats
from .simulator.scheduler import RequestScheduler, SchedulerStats
from .simulator.eviction import (
    EvictionPolicy,
    LRUEvictionPolicy,
    LFUEvictionPolicy,
    FIFOEvictionPolicy,
    ContinuumEvictionPolicy,
    create_eviction_policy,
)
from .simulator.request import (
    Request,
    ProcessResult,
    BatchResult,
    RequestMetrics,
)
from .simulator.request_log import RequestLogReader, RequestLogWriter, RequestLogEntry
from .simulator.stats import (
    CacheStats as SimulationCacheStats,
    SimulationResult,
    StatisticsCollector,
)
from .simulator.main import KVCacheSimulator, SimulatorConfig, run_simulation

__all__ = [
    # Main simulator
    "KVCacheSimulator",
    "SimulatorConfig",
    "run_simulation",
    
    # Core components
    "RadixTree",
    "RadixKey",
    "CacheAllocator",
    "TTLManager",
    "RequestScheduler",
    
    # Data classes
    "TreeNode",
    "MatchResult",
    "InsertResult",
    "Request",
    "ProcessResult",
    "BatchResult",
    "RequestMetrics",
    "RequestLogReader",
    "RequestLogWriter",
    "RequestLogEntry",
    "PinnedEntry",
    
    # Stats
    "CacheStats",
    "SimulationCacheStats",
    "SimulationResult",
    "StatisticsCollector",
    "SchedulerStats",
    "TTLStats",
    
    # Eviction policies
    "EvictionPolicy",
    "LRUEvictionPolicy",
    "LFUEvictionPolicy",
    "FIFOEvictionPolicy",
    "ContinuumEvictionPolicy",
    "create_eviction_policy",
]

__version__ = "1.0.0"
