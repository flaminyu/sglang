"""
TreeNode: Represents a node in the radix cache tree.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TreeNode:
    """A node in the radix tree representing a token sequence."""
    
    node_id: int
    key: "RadixKey"  # Forward reference
    parent: Optional["TreeNode"] = None
    
    children: Dict[Any, "TreeNode"] = field(default_factory=dict)
    kv_indices: List[int] = field(default_factory=list)
    num_tokens: int = 0
    
    lock_ref: int = 0
    last_access_time: float = 0.0
    hit_count: int = 0
    
    ttl_expiry_time: Optional[float] = None
    is_pinned: bool = False
    
    # Track which program owns this pinned node
    program_id: Optional[str] = None
    
    priority: float = 0.0
    priority_override: Optional[float] = None
    
    creation_time: float = 0.0
    last_ttl_check: float = 0.0
    
    @property
    def is_root(self) -> bool:
        return self.parent is None
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    @property
    def is_evictable(self) -> bool:
        """
        Check if this node can be evicted.
        
        According to Continuum paper: ALL nodes are evictable,
        including pinned ones. TTL pins only affect scheduling priority,
        not eviction protection.
        
        When L0 is full, the scheduler evicts nodes in program arrival order
        (oldest first) regardless of pin status.
        """
        if self.is_root:
            return False
        # All nodes are evictable - pin status doesn't block eviction
        return True
    
    @property
    def is_expired(self) -> bool:
        """Check if TTL has expired using wall clock time (for real-world use)."""
        if self.ttl_expiry_time is None:
            return False
        current = time.time()
        if current > self.last_ttl_check + 1.0:
            self._is_expired_cached = current > self.ttl_expiry_time
            self.last_ttl_check = current
        return getattr(self, '_is_expired_cached', False)
    
    def is_expired_at(self, current_time: float) -> bool:
        """Check if TTL has expired at the given simulation time."""
        if self.ttl_expiry_time is None:
            return False
        return current_time >= self.ttl_expiry_time
    
    @property
    def is_valid(self) -> bool:
        """A node is valid if it has KV cache and is not expired.
        
        Note: This uses wall clock time. For simulation time, use check_expired(current_time).
        """
        if not self.kv_indices:
            return False
        # Use wall clock time for this property (for real-world use)
        # Simulation code should use check_expired(current_time) instead
        return not self.is_expired
    
    def check_expired(self, current_time: float) -> bool:
        if self.ttl_expiry_time is None:
            return False
        return current_time >= self.ttl_expiry_time
    
    def set_ttl(self, ttl_sec: float):
        """Set TTL expiry time (matching real SGLang API)."""
        self.ttl_expiry_time = time.time() + ttl_sec
        self.last_ttl_check = time.time()
        self._is_expired_cached = False
    
    def get_tokens_in_subtree(self) -> int:
        total = len(self.key)
        for child in self.children.values():
            total += child.get_tokens_in_subtree()
        return total
    
    def get_evictable_leaf_nodes(self) -> List["TreeNode"]:
        if self.is_leaf:
            return [self] if self.is_evictable else []
        leaves = []
        for child in self.children.values():
            leaves.extend(child.get_evictable_leaf_nodes())
        return leaves
    
    def pin(self, ttl_sec: float, current_time: float, program_id: str = None):
        """Pin this node with TTL (matching real SGLang API)."""
        self.is_pinned = True
        self.program_id = program_id
        # Set TTL based on current_time (simulation time), not wall clock
        self.ttl_expiry_time = current_time + ttl_sec
        self.last_ttl_check = current_time
        self._is_expired_cached = False
        self.last_access_time = current_time
    
    def unpin(self, current_time: float) -> bool:
        self.is_pinned = False
        return self.is_evictable
    
    def access(self, current_time: float):
        self.last_access_time = current_time
        self.hit_count += 1
    
    def __repr__(self) -> str:
        pinned_str = " [PINNED]" if self.is_pinned else ""
        locked_str = f" [LOCK:{self.lock_ref}]" if self.lock_ref > 0 else ""
        return (f"TreeNode(id={self.node_id}, tokens={len(self.key)}, "
                f"children={len(self.children)}{pinned_str}{locked_str})")
