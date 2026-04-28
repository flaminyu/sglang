"""
RadixKey: Represents a token sequence in the radix tree with namespace isolation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


@dataclass
class RadixKey:
    """Represents a token sequence with optional namespace (extra_key)."""
    
    token_ids: List[int]
    extra_key: Optional[str] = None
    
    def __post_init__(self):
        """Ensure token_ids is a list (not tuple)."""
        if isinstance(self.token_ids, tuple):
            self.token_ids = list(self.token_ids)
    
    def __hash__(self) -> int:
        """Hash based on extra_key and token_ids."""
        if self.extra_key is None:
            return hash(tuple(self.token_ids))
        return hash((self.extra_key, tuple(self.token_ids)))
    
    def __eq__(self, other: object) -> bool:
        """Two RadixKeys are equal if they have same tokens and extra_key."""
        if not isinstance(other, RadixKey):
            return False
        return (self.extra_key == other.extra_key and 
                self.token_ids == other.token_ids)
    
    def __len__(self) -> int:
        """Return the number of tokens."""
        return len(self.token_ids)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[int, List[int]]:
        """Support indexing and slicing."""
        if isinstance(index, slice):
            return self.token_ids[index]
        return self.token_ids[index]
    
    def prefix_match(self, other: "RadixKey") -> int:
        """Find the longest common prefix length between this key and another."""
        if self.extra_key != other.extra_key:
            return 0
        
        min_len = min(len(self), len(other))
        match_len = 0
        for i in range(min_len):
            if self.token_ids[i] == other.token_ids[i]:
                match_len += 1
            else:
                break
        return match_len
    
    def match_full(self, other: "RadixKey") -> bool:
        """Check if this key fully matches another."""
        return len(self) == len(other) and self.prefix_match(other) == len(self)
    
    @classmethod
    def from_token(cls, token_id: int, extra_key: Optional[str] = None) -> "RadixKey":
        """Create a single-token RadixKey."""
        return cls(token_ids=[token_id], extra_key=extra_key)
    
    @classmethod
    def from_tokens(cls, token_ids: List[int], extra_key: Optional[str] = None) -> "RadixKey":
        """Create a RadixKey from a list of token IDs."""
        return cls(token_ids=token_ids, extra_key=extra_key)
    
    def to_tuple(self) -> Tuple:
        """Convert to a hashable tuple."""
        return (tuple(self.token_ids), self.extra_key)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        prefix = self.token_ids[:5]
        suffix = f"...({len(self.token_ids)} tokens)" if len(self.token_ids) > 5 else ""
        key_preview = str(prefix) + suffix if suffix else str(prefix)
        extra = f", extra_key={self.extra_key!r}" if self.extra_key else ""
        return f"RadixKey({key_preview}{extra})"
