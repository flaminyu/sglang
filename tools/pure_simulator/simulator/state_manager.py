"""
StateManager: Singleton to track HiCache L2 load/backup durations.

This mirrors the StateManager from simulator-sglang-origin, providing
the interface that AIConfigurator needs to integrate with pure_simulator.
"""

from typing import Optional


class StateManager:
    """
    Singleton to track HiCache L2 load/backup durations.
    
    In the original SGLang simulator, these values are set by the C++
    HiCache controller. In pure_simulator, they are set by the L2Cache
    when loading/backup operations occur.
    
    Usage:
        # In L2Cache.load()
        StateManager.inc_hicache_l2_load_dur(h2d_time)
        
        # In L2Cache.backup()
        StateManager.inc_hicache_l2_backup_dur(backup_time)
        
        # In scheduler after batch processing
        l2_load_dur = StateManager.pop_hicache_l2_load_dur()
        l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()
    """
    
    _hicache_l2_load_dur: float = 0.0
    _hicache_l2_backup_dur: float = 0.0
    _last_inference_dur: float = 0.0
    _global_clock: float = 0.0
    _iteration: int = 0
    
    @classmethod
    def reset(cls):
        """Reset all state (call between simulations)."""
        cls._hicache_l2_load_dur = 0.0
        cls._hicache_l2_backup_dur = 0.0
        cls._last_inference_dur = 0.0
        cls._global_clock = 0.0
        cls._iteration = 0
    
    # L2 load/backup tracking
    @classmethod
    def inc_hicache_l2_load_dur(cls, dur: float) -> None:
        """Add to L2 load duration."""
        cls._hicache_l2_load_dur += dur
    
    @classmethod
    def inc_hicache_l2_backup_dur(cls, dur: float) -> None:
        """Add to L2 backup duration."""
        cls._hicache_l2_backup_dur += dur
    
    @classmethod
    def pop_hicache_l2_load_dur(cls) -> float:
        """Get and reset L2 load duration."""
        dur = cls._hicache_l2_load_dur
        cls._hicache_l2_load_dur = 0.0
        return dur
    
    @classmethod
    def pop_hicache_l2_backup_dur(cls) -> float:
        """Get and reset L2 backup duration."""
        dur = cls._hicache_l2_backup_dur
        cls._hicache_l2_backup_dur = 0.0
        return dur
    
    @classmethod
    def get_hicache_l2_load_dur(cls) -> float:
        """Get L2 load duration without resetting."""
        return cls._hicache_l2_load_dur
    
    @classmethod
    def get_hicache_l2_backup_dur(cls) -> float:
        """Get L2 backup duration without resetting."""
        return cls._hicache_l2_backup_dur
    
    # Global clock
    @classmethod
    def set_global_clock(cls, clock: float) -> None:
        """Set the global simulation clock."""
        cls._global_clock = clock
    
    @classmethod
    def get_global_clock(cls) -> float:
        """Get the global simulation clock."""
        return cls._global_clock
    
    @classmethod
    def step_global_clock(cls, dt: float) -> None:
        """Advance the global clock by dt seconds."""
        cls._global_clock += dt
    
    # Inference duration
    @classmethod
    def set_current_inference_dur(cls, dur: float) -> None:
        """Set the current inference duration."""
        cls._last_inference_dur = dur
    
    @classmethod
    def get_current_inference_dur(cls) -> float:
        """Get the current inference duration."""
        return cls._last_inference_dur
    
    @classmethod
    def get_last_inference_dur(cls) -> float:
        """Get the last inference duration (alias)."""
        return cls._last_inference_dur
    
    # Iteration tracking
    @classmethod
    def inc_iteration(cls) -> None:
        """Increment the iteration counter."""
        cls._iteration += 1
    
    @classmethod
    def get_iteration(cls) -> int:
        """Get the current iteration number."""
        return cls._iteration
    
    @classmethod
    def reset_iteration(cls) -> None:
        """Reset the iteration counter."""
        cls._iteration = 0
