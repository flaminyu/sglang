"""
RadixTree: Prefix-based KV cache storage and matching.
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .radix_key import RadixKey
from .tree_node import TreeNode
from .cache_allocator import CacheAllocator


@dataclass
class MatchResult:
    """Result of a prefix match operation."""
    matched_tokens: int = 0
    matched_nodes: List[TreeNode] = field(default_factory=list)
    cached_indices: List[int] = field(default_factory=list)
    last_node: Optional[TreeNode] = None
    hit: bool = False


@dataclass
class InsertResult:
    """Result of an insert operation."""
    node: TreeNode
    new_indices: List[int]


class RadixTree:
    """
    Prefix-based radix tree for KV cache storage.
    
    Key design:
    - Children are indexed by first token ID
    - Each node stores a contiguous token sequence
    - Prefix matching walks through children by first token
    """
    
    def __init__(
        self,
        allocator: Optional[CacheAllocator] = None,
        capacity: int = 100000,
        page_size: int = 1,
        eviction_policy: str = "lru",
        ttl_manager = None  # Reference to TTLManager for eviction cleanup
    ):
        if allocator is None:
            allocator = CacheAllocator(capacity)
        
        self.allocator = allocator
        self.page_size = page_size
        self.eviction_policy = eviction_policy
        self.ttl_manager = ttl_manager  # Reference to TTLManager
        
        self.next_node_id = 1
        self.root = TreeNode(
            node_id=0,
            key=RadixKey(token_ids=[]),
            creation_time=time.time()
        )
        self.nodes: Dict[int, TreeNode] = {0: self.root}
        
        self.token_count = 0
        self.evictable_size = 0
        self.protected_size = 0
        
        # Track evictable leaf node IDs efficiently (matching real SGLang)
        self.evictable_leaves: Set[int] = set()
        
        self.stats = CacheStats()
    
    def _get_next_node_id(self) -> int:
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id
    
    def _make_child_key(self, key: RadixKey) -> Any:
        if not key.token_ids:
            return None
        if key.extra_key is None:
            return key.token_ids[0]
        return (key.extra_key, key.token_ids[0])
    
    def _key_match(self, node_key: RadixKey, search_key: RadixKey) -> int:
        if node_key.extra_key != search_key.extra_key:
            return 0
        
        match_len = 0
        for i in range(min(len(node_key), len(search_key))):
            if node_key.token_ids[i] == search_key.token_ids[i]:
                match_len += 1
            else:
                break
        return match_len
    
    def match_prefix(self, key: RadixKey) -> MatchResult:
        if not key.token_ids:
            return MatchResult(hit=False)
        
        result = MatchResult()
        current = self.root
        
        while True:
            child_key = self._make_child_key(key)
            if child_key is None:
                break
            
            if child_key not in current.children:
                break
            
            child = current.children[child_key]
            match_len = self._key_match(child.key, key)
            
            if match_len == 0:
                break
            
            result.matched_nodes.append(child)
            result.cached_indices.extend(child.kv_indices[:match_len])
            result.matched_tokens += match_len
            
            if match_len < len(child.key):
                break
            
            current = child
            key = RadixKey(token_ids=key.token_ids[match_len:], extra_key=key.extra_key)
            
            if not key.token_ids:
                result.last_node = current
                result.hit = True
                return result
        
        if result.matched_tokens > 0:
            result.hit = True
            result.last_node = result.matched_nodes[-1] if result.matched_nodes else None
        
        return result
    
    def insert(
        self,
        key: RadixKey,
        kv_indices: Optional[List[int]] = None,
        create_new_indices: bool = True
    ) -> InsertResult:
        if not key.token_ids:
            raise ValueError("Cannot insert empty key")
        
        match_result = self.match_prefix(key)
        remaining_tokens = len(key) - match_result.matched_tokens
        
        new_indices = []
        if remaining_tokens > 0:
            if kv_indices and len(kv_indices) > match_result.matched_tokens:
                # Use pre-allocated indices
                new_indices = kv_indices[match_result.matched_tokens:]
                new_indices = new_indices[:remaining_tokens]
            elif create_new_indices:
                # Try to allocate
                try:
                    new_indices = self.allocator.alloc(remaining_tokens)
                except RuntimeError:
                    # Not enough space - caller should have evicted first
                    pass
        
        if remaining_tokens > 0 and new_indices:
            parent = match_result.last_node or self.root
            remaining_key = RadixKey(
                token_ids=key.token_ids[match_result.matched_tokens:],
                extra_key=key.extra_key
            )
            
            new_node = TreeNode(
                node_id=self._get_next_node_id(),
                key=remaining_key,
                parent=parent,
                kv_indices=new_indices,
                num_tokens=len(remaining_key),
                creation_time=time.time()
            )
            
            child_key = self._make_child_key(remaining_key)
            parent.children[child_key] = new_node
            self.nodes[new_node.node_id] = new_node
            
            self.token_count += len(remaining_key)
            self.evictable_size += len(remaining_key)
            
            # Add to evictable leaves (matching real SGLang)
            self._update_leaf_status(new_node)
            
            # tokens_cached is now tracked via allocator.used(), not updated here
            return InsertResult(node=new_node, new_indices=new_indices)
        else:
            # No new tokens to insert (either no remaining tokens or allocation failed)
            # tokens_cached is tracked via allocator.used()
            return InsertResult(
                node=match_result.last_node or self.root,
                new_indices=[]
            )
    
    def insert_node(
        self,
        key: RadixKey,
        kv_indices: List[int],
        node_id: Optional[int] = None,
        program_id: Optional[str] = None
    ) -> InsertResult:
        """
        Directly insert a node without prefix matching.
        
        Used for L2 load-back scenario where we need to restore a node
        that was previously evicted to L2.
        
        Args:
            key: The RadixKey containing token_ids and extra_key
            kv_indices: Pre-allocated KV cache indices
            node_id: Optional node_id (generates new one if not provided)
            program_id: Optional program_id to associate with this node
        
        Returns:
            InsertResult with the new node and indices
        """
        if not key.token_ids:
            raise ValueError("Cannot insert empty key")
        
        if not kv_indices:
            raise ValueError("Cannot insert node without kv_indices")
        
        # Walk the tree to find the correct parent
        # For load-back, we need to find if there are existing nodes
        # that are prefixes of the key we're inserting
        current = self.root
        remaining_key = key
        
        while True:
            child_key = self._make_child_key(remaining_key)
            
            # Check if this child exists
            if child_key not in current.children:
                break
            
            child = current.children[child_key]
            
            # Check how much of the child matches our remaining key
            match_len = self._key_match(child.key, remaining_key)
            
            if match_len < len(child.key):
                # Partial match - we need to split the child
                # This case is complex; for now, we'll just add as a sibling
                break
            
            # Full match of child key, continue down
            if match_len < len(remaining_key):
                current = child
                remaining_key = RadixKey(
                    token_ids=remaining_key.token_ids[match_len:],
                    extra_key=remaining_key.extra_key
                )
            else:
                # Full match - node already exists
                # Update its kv_indices if provided
                if kv_indices:
                    child.kv_indices = kv_indices
                return InsertResult(node=child, new_indices=kv_indices)
        
        # Create new node
        if node_id is None:
            node_id = self._get_next_node_id()
        
        new_node = TreeNode(
            node_id=node_id,
            key=key,
            parent=current,
            kv_indices=kv_indices,
            num_tokens=len(key.token_ids),
            creation_time=time.time(),
            program_id=program_id
        )
        
        child_key = self._make_child_key(key)
        current.children[child_key] = new_node
        self.nodes[new_node.node_id] = new_node
        
        self.token_count += len(key.token_ids)
        self.evictable_size += len(key.token_ids)
        self._update_leaf_status(new_node)
        
        return InsertResult(node=new_node, new_indices=kv_indices)
    
    def _update_leaf_status(self, node: TreeNode) -> None:
        """
        Update evictable_leaves set when node state changes.
        
        According to Continuum paper: ALL nodes are evictable,
        including those with lock_ref > 0. TTL pins only affect
        scheduling priority, not eviction.
        """
        # All nodes are evictable regardless of lock_ref
        # The evict() function will handle eviction of pinned nodes
        pass
    
    def inc_lock_ref(self, node: TreeNode, current_time: float) -> int:
        delta = 0
        current = node
        
        while current is not None and current != self.root:
            if current.lock_ref == 0:
                delta += len(current.key)
                self.evictable_size -= len(current.key)
                self.protected_size += len(current.key)
            
            current.lock_ref += 1
            current.last_access_time = current_time
            
            current = current.parent
        
        return delta
    
    def dec_lock_ref(self, node: TreeNode, current_time: float) -> int:
        delta = 0
        current = node
        
        while current is not None and current != self.root:
            if current.lock_ref == 1:
                delta += len(current.key)
                self.evictable_size += len(current.key)
                self.protected_size -= len(current.key)
            
            current.lock_ref = max(0, current.lock_ref - 1)
            
            current = current.parent
        
        return delta
    
    def evict(self, num_tokens: int, current_time: float = 0.0, current_program_id: str = None) -> List[TreeNode]:
        """
        Evict nodes according to Continuum paper's strategy.

        Priority order:
        1. OLDEST unpinned nodes (always evicted first)
        2. OLDEST pinned nodes from the CURRENT program (if space needed by current program)
        3. OLDEST pinned nodes from OTHER programs (only if ALL other options exhausted)

        This protects OTHER programs' pinned nodes while allowing the current program
        to reclaim its own pinned space if needed.
        
        Returns:
            List of (node, was_forced_unpin) tuples. was_forced_unpin is True if the node
            was pinned and had its pin removed to allow eviction.
        """
        if num_tokens <= 0:
            return []

        evicted = []
        freed_tokens = 0

        while freed_tokens < num_tokens:
            # Get all leaf nodes
            all_leaves = self._get_all_leaf_nodes()
            if not all_leaves:
                break

            # Separate nodes into categories
            unpinned_nodes = []  # (creation_time, node)
            current_pinned = []  # (creation_time, node)
            other_pinned = []   # (creation_time, program_id, node)

            for node in all_leaves:
                if not node.kv_indices:
                    continue
                    
                is_pinned = node.is_pinned and not node.check_expired(current_time)
                is_current = (current_program_id is not None and node.program_id == current_program_id)
                
                if not is_pinned:
                    unpinned_nodes.append((node.creation_time, node))
                elif is_current:
                    current_pinned.append((node.creation_time, node))
                else:
                    # For other programs, track program_id for grouping
                    other_pinned.append((node.creation_time, node.program_id, node))

            # Sort each category by creation time (oldest first)
            unpinned_nodes.sort(key=lambda x: x[0])
            current_pinned.sort(key=lambda x: x[0])
            other_pinned.sort(key=lambda x: (x[0], x[1]))  # Sort by time, then program_id

            # Evict in priority order
            evicted_this_round = False

            # 1. Evict unpinned nodes (highest priority)
            for creation_time, node in unpinned_nodes:
                if freed_tokens >= num_tokens:
                    break
                evicted.append((node, False))  # Not forced unpin
                freed_tokens += len(node.kv_indices)
                self._evict_single_node(node, current_time)
                evicted_this_round = True

            # 2. Evict current program's pinned nodes (forced unpin)
            for creation_time, node in current_pinned:
                if freed_tokens >= num_tokens:
                    break
                evicted.append((node, True))  # Forced unpin
                freed_tokens += len(node.kv_indices)
                self._evict_single_node(node, current_time)
                evicted_this_round = True

            # 3. Only if no progress and still need space, evict other programs' pinned
            if freed_tokens < num_tokens and other_pinned:
                # Group by program and find the oldest program
                program_nodes = {}
                for creation_time, prog_id, node in other_pinned:
                    if prog_id not in program_nodes:
                        program_nodes[prog_id] = []
                    program_nodes[prog_id].append((creation_time, node))

                # Find the oldest program (by earliest node creation time)
                oldest_program = min(program_nodes.keys(), key=lambda p: program_nodes[p][0][0])
                nodes_to_evict = program_nodes[oldest_program]

                for creation_time, node in nodes_to_evict:
                    if freed_tokens >= num_tokens:
                        break
                    evicted.append((node, True))  # Forced unpin
                    freed_tokens += len(node.kv_indices)
                    self._evict_single_node(node, current_time)
                    evicted_this_round = True

            # If no progress was made in this round, break (can't evict more)
            if not evicted_this_round:
                break

        return evicted
    
    def _evict_single_node(self, node: "TreeNode", current_time: float) -> None:
        """Helper to evict a single node."""
        # If this node is pinned, unpin via TTLManager first
        # This will call dec_lock_ref which updates protected/evictable sizes
        if node.is_pinned and node.program_id and self.ttl_manager:
            # pinned_entries uses (program_id, node_id) tuple keys
            entry_key = (node.program_id, node.node_id)
            entry = self.ttl_manager.pinned_entries.get(entry_key)
            if entry:
                self.ttl_manager.unpin_program(node.program_id, current_time, node.node_id)
        
        # Now evictable_size should be correctly updated by dec_lock_ref
        # Free the allocator space and remove from tree
        self.allocator.free(node.kv_indices)
        self._remove_node(node)
        self.token_count -= len(node.key)
        self.stats.tokens_evicted += len(node.key)
    
    def _get_all_leaf_nodes(self) -> List["TreeNode"]:
        """Get all leaf nodes in the tree."""
        leaves = []
        def collect(node: "TreeNode"):
            if not node.children:
                leaves.append(node)
            else:
                for child in node.children.values():
                    collect(child)
        collect(self.root)
        return leaves
    
    def _get_evictable_leaves(self) -> List[TreeNode]:
        def collect(node: TreeNode) -> List[TreeNode]:
            if not node.children:
                return [node] if node.is_evictable else []
            result = []
            for child in node.children.values():
                result.extend(collect(child))
            return result
        return collect(self.root)
    
    def _compute_eviction_priority(self, node: TreeNode) -> float:
        if not node.is_evictable:
            return float('inf')
        
        if self.eviction_policy == "lfu":
            return -node.hit_count
        elif self.eviction_policy == "fifo":
            return node.creation_time
        elif self.eviction_policy == "mru":
            return -node.last_access_time
        else:  # lru
            return node.last_access_time
    
    def _remove_node(self, node: TreeNode) -> None:
        parent = node.parent
        if parent:
            child_key = self._make_child_key(node.key)
            if child_key in parent.children:
                del parent.children[child_key]
        
        if node.node_id in self.nodes:
            del self.nodes[node.node_id]
    
    def cleanup_expired_ttls(self, current_time: float) -> List[TreeNode]:
        """
        Unpin nodes whose TTL has expired.
        Matching real SGLang behavior.
        """
        unpinned = []
        
        def traverse(node: TreeNode):
            nonlocal unpinned
            if node.is_pinned and node.check_expired(current_time):
                node.unpin(current_time)
                if node.lock_ref == 1:
                    self.dec_lock_ref(node, current_time)
                unpinned.append(node)
                self.stats.ttl_expired += 1
                # Update evictable leaves after unpin
                self._update_leaf_status(node)
            
            for child in list(node.children.values()):
                traverse(child)
        
        traverse(self.root)
        return unpinned
    
    def reset(self):
        self.root.children.clear()
        self.nodes = {0: self.root}
        self.next_node_id = 1
        self.token_count = 0
        self.evictable_size = 0
        self.protected_size = 0
        self.evictable_leaves.clear()
        self.stats = CacheStats()
        self.allocator.reset()


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
