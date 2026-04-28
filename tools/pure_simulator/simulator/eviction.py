"""
Eviction policies for the radix tree cache.

Implements various eviction strategies including LRU, LFU, FIFO, and Continuum-aware eviction.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .tree_node import TreeNode


class EvictionPolicy(ABC):
    """
    Base class for eviction policies.
    
    An eviction policy determines which nodes should be evicted when
    the cache needs to free space.
    """
    
    @abstractmethod
    def compute_priority(self, node: TreeNode, current_time: float) -> float:
        """
        Compute eviction priority for a node.
        
        Lower values = higher priority to evict.
        
        Args:
            node: The node to evaluate
            current_time: Current simulation time
            
        Returns:
            Priority value (lower = evict first)
        """
        pass
    
    def name(self) -> str:
        """Return the policy name."""
        return self.__class__.__name__.replace("EvictionPolicy", "").lower()


class LRUEvictionPolicy(EvictionPolicy):
    """
    Least Recently Used eviction policy.
    
    Evicts nodes that haven't been accessed in the longest time.
    """
    
    def compute_priority(self, node: TreeNode, current_time: float) -> float:
        """
        LRU: Lower last_access_time = higher priority to evict.
        
        We use negative of last_access_time so that older items have lower priority.
        """
        if not node.is_evictable:
            return float('inf')
        return -node.last_access_time


class LFUEvictionPolicy(EvictionPolicy):
    """
    Least Frequently Used eviction policy.
    
    Evicts nodes with the lowest hit count.
    """
    
    def compute_priority(self, node: TreeNode, current_time: float) -> float:
        """LFU: Lower hit count = higher eviction priority."""
        if not node.is_evictable:
            return float('inf')
        return -node.hit_count


class FIFOEvictionPolicy(EvictionPolicy):
    """
    First In First Out eviction policy.
    
    Evicts nodes in order of creation time.
    """
    
    def compute_priority(self, node: TreeNode, current_time: float) -> float:
        """FIFO: Older creation time = higher eviction priority."""
        if not node.is_evictable:
            return float('inf')
        return node.creation_time


class MRUEvictionPolicy(EvictionPolicy):
    """
    Most Recently Used eviction policy.
    
    Evicts the most recently accessed nodes first.
    """
    
    def compute_priority(self, node: TreeNode, current_time: float) -> float:
        """MRU: Higher last_access_time = higher eviction priority."""
        if not node.is_evictable:
            return float('inf')
        return node.last_access_time


class FILOEvictionPolicy(EvictionPolicy):
    """
    First In Last Out eviction policy.
    
    Evicts newest nodes first.
    """
    
    def compute_priority(self, node: TreeNode, current_time: float) -> float:
        """FILO: Newer creation time = higher eviction priority."""
        if not node.is_evictable:
            return float('inf')
        return -node.creation_time


class PriorityEvictionPolicy(EvictionPolicy):
    """
    Priority-based eviction policy.
    
    Evicts nodes based on their explicit priority value.
    """
    
    def compute_priority(self, node: TreeNode, current_time: float) -> float:
        """Priority: Higher node.priority = higher eviction priority."""
        if not node.is_evictable:
            return float('inf')
        return -node.priority


class ContinuumEvictionPolicy(EvictionPolicy):
    """
    Continuum-aware eviction policy.
    
    This policy is designed for the Continuum TTL mechanism:
    1. Never evict locked nodes
    2. Never evict pinned (TTL-active) nodes
    3. Prefer evicting expired TTL nodes
    4. Consider recency and priority
    """
    
    def __init__(self, recency_weight: float = 1.0, priority_weight: float = 0.1):
        self.recency_weight = recency_weight
        self.priority_weight = priority_weight
    
    def compute_priority(self, node: TreeNode, current_time: float) -> float:
        if not node.is_evictable:
            return float('inf')
        
        # Expired TTL should be evicted first
        if node.ttl_expiry_time is not None and current_time >= node.ttl_expiry_time:
            if not node.is_pinned:
                return -1000
            return float('inf')
        
        recency = current_time - node.last_access_time
        return -(recency * self.recency_weight + node.priority * self.priority_weight)


class HybridEvictionPolicy(EvictionPolicy):
    """
    Hybrid eviction policy combining multiple strategies.
    """
    
    def __init__(self, base_policy: str = "lru", priority_boost: float = 0.5):
        self.base_policy_name = base_policy
        self.priority_boost = priority_boost
        self.base_policy = create_eviction_policy(base_policy)
    
    def compute_priority(self, node: TreeNode, current_time: float) -> float:
        if not node.is_evictable:
            return float('inf')
        
        base_priority = self.base_policy.compute_priority(node, current_time)
        priority_factor = -node.priority * self.priority_boost
        return base_priority + priority_factor


def create_eviction_policy(policy_name: str) -> EvictionPolicy:
    """Factory function to create an eviction policy by name."""
    policy_map = {
        "lru": LRUEvictionPolicy,
        "lfu": LFUEvictionPolicy,
        "fifo": FIFOEvictionPolicy,
        "mru": MRUEvictionPolicy,
        "filo": FILOEvictionPolicy,
        "priority": PriorityEvictionPolicy,
        "continuum": ContinuumEvictionPolicy,
        "hybrid": HybridEvictionPolicy,
    }
    
    policy_name_lower = policy_name.lower()
    if policy_name_lower not in policy_map:
        raise ValueError(
            f"Unknown eviction policy: {policy_name}. "
            f"Valid options: {', '.join(policy_map.keys())}"
        )
    
    return policy_map[policy_name_lower]()
