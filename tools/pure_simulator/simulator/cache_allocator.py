"""
CacheAllocator: Manages token/slot allocation in the KV cache memory pool.

This is a simplified version of SGLang's TokenToKVPoolAllocator that manages
a pool of token slots without GPU memory constraints.
"""

from typing import Dict, List, Set, Optional


class CacheAllocator:
    """
    Manages allocation and deallocation of token slots in the KV cache pool.
    
    This simulates the memory pool management in SGLang's TokenToKVPoolAllocator
    but in a simplified way suitable for pure Python simulation.
    
    The allocator maintains a free list of available token slots and tracks
    which nodes own which slots.
    
    Attributes:
        capacity: Maximum number of token slots in the pool
        free_slots: List of available (free) token indices
        allocated: Mapping from token index to owning node_id
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the cache allocator.
        
        Args:
            capacity: Maximum number of token slots in the pool
        """
        self.capacity = capacity
        self.free_slots: List[int] = []
        self.allocated: Dict[int, int] = {}  # index -> node_id
        self._init_free_list()
    
    def _init_free_list(self):
        """Initialize the free list with all slots."""
        self.free_slots = list(range(self.capacity))
    
    def alloc(self, num_tokens: int, node_id: int = -1) -> List[int]:
        """
        Allocate token slots from the free list.
        
        Args:
            num_tokens: Number of token slots to allocate
            node_id: ID of the node that will own these slots
            
        Returns:
            List of allocated token indices
            
        Raises:
            RuntimeError: If not enough free slots available
        """
        if num_tokens == 0:
            return []
        
        if len(self.free_slots) < num_tokens:
            raise RuntimeError(
                f"Cache allocator out of memory: need {num_tokens} slots, "
                f"only {len(self.free_slots)} available"
            )
        
        # Pop from free list (FIFO)
        indices = []
        for _ in range(num_tokens):
            idx = self.free_slots.pop(0)
            self.allocated[idx] = node_id
            indices.append(idx)
        
        return indices
    
    def alloc_range(self, num_tokens: int, node_id: int = -1) -> List[int]:
        """
        Allocate consecutive token slots (if available).
        
        Args:
            num_tokens: Number of consecutive slots needed
            node_id: ID of the owning node
            
        Returns:
            List of allocated token indices (consecutive), or empty if not available
        """
        if num_tokens == 0:
            return []
        
        # Find consecutive slots
        start = -1
        count = 0
        for i in range(self.capacity):
            if i in self.allocated:
                # Occupied slot, reset search
                start = -1
                count = 0
            else:
                # Free slot
                if start == -1:
                    start = i
                count += 1
                if count == num_tokens:
                    # Found enough consecutive slots
                    indices = list(range(start, start + num_tokens))
                    for idx in indices:
                        self.allocated[idx] = node_id
                        self.free_slots.remove(idx)
                    return indices
        
        # Not enough consecutive slots, fall back to non-consecutive allocation
        return self.alloc(num_tokens, node_id)
    
    def free(self, indices: List[int]) -> None:
        """
        Free allocated token slots.
        
        Args:
            indices: List of token indices to free
        """
        for idx in indices:
            if idx in self.allocated:
                del self.allocated[idx]
                self.free_slots.append(idx)
        
        # Keep free list sorted for easier debugging
        self.free_slots.sort()
    
    def free_by_node(self, node_id: int) -> List[int]:
        """
        Free all token slots owned by a specific node.
        
        Args:
            node_id: ID of the owning node
            
        Returns:
            List of freed token indices
        """
        to_free = [idx for idx, nid in self.allocated.items() if nid == node_id]
        self.free(to_free)
        return to_free
    
    def available(self) -> int:
        """
        Get number of available (free) token slots.
        
        Returns:
            Number of free slots
        """
        return len(self.free_slots)
    
    def used(self) -> int:
        """
        Get number of used (allocated) token slots.
        
        Returns:
            Number of allocated slots
        """
        return len(self.allocated)
    
    def utilization(self) -> float:
        """
        Get pool utilization as a fraction (0.0 to 1.0).
        
        Returns:
            Fraction of slots in use
        """
        return self.used() / self.capacity
    
    def reset(self):
        """Reset the allocator to initial state."""
        self.free_slots.clear()
        self.allocated.clear()
        self._init_free_list()
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"CacheAllocator(capacity={self.capacity}, "
                f"used={self.used()}, free={self.available()})")


class PagedCacheAllocator(CacheAllocator):
    """
    Cache allocator with page-based allocation for PagedAttention.
    
    In PagedAttention, KV cache is organized into fixed-size pages.
    This allocator manages pages instead of individual tokens.
    """
    
    def __init__(self, capacity: int = 100000, page_size: int = 16):
        """
        Initialize the paged cache allocator.
        
        Args:
            capacity: Maximum number of token slots (will be rounded up to page boundary)
            page_size: Number of tokens per page
        """
        # Round up capacity to page boundary
        self.page_size = page_size
        self.num_pages = (capacity + page_size - 1) // page_size
        actual_capacity = self.num_pages * page_size
        
        super().__init__(actual_capacity)
        
        self.pages_free: List[int] = []  # Free page indices
        self.tokens_in_page: List[List[int]] = []  # Tokens within each page
        
        # Initialize pages
        for page_idx in range(self.num_pages):
            start = page_idx * page_size
            tokens = list(range(start, start + page_size))
            self.tokens_in_page.append(tokens)
            self.pages_free.append(page_idx)
    
    def alloc_pages(self, num_tokens: int, node_id: int = -1) -> List[int]:
        """
        Allocate pages for the requested number of tokens.
        
        Args:
            num_tokens: Number of tokens needed
            node_id: Owning node ID
            
        Returns:
            List of token indices (may be non-consecutive across pages)
        """
        num_pages_needed = (num_tokens + self.page_size - 1) // self.page_size
        
        if len(self.pages_free) < num_pages_needed:
            raise RuntimeError(
                f"Not enough free pages: need {num_pages_needed}, "
                f"only {len(self.pages_free)} available"
            )
        
        tokens = []
        for _ in range(num_pages_needed):
            page_idx = self.pages_free.pop(0)
            for token_idx in self.tokens_in_page[page_idx]:
                self.allocated[token_idx] = node_id
                tokens.append(token_idx)
        
        return tokens[:num_tokens]  # Return only needed tokens
    
    def free_pages_by_node(self, node_id: int) -> List[int]:
        """
        Free all pages owned by a node.
        
        Args:
            node_id: Owning node ID
            
        Returns:
            List of freed token indices
        """
        freed_tokens = []
        pages_to_free = set()
        
        for token_idx, nid in self.allocated.items():
            if nid == node_id:
                freed_tokens.append(token_idx)
                pages_to_free.add(token_idx // self.page_size)
        
        for token_idx in freed_tokens:
            del self.allocated[token_idx]
        
        for page_idx in pages_to_free:
            if page_idx not in self.pages_free:
                self.pages_free.append(page_idx)
                self.pages_free.sort()
        
        return freed_tokens
