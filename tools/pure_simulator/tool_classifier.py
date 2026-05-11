"""
Tool Classifier: Categorizes tools by their resource characteristics and anomaly patterns.

This module provides:
- Tool categories based on resource consumption patterns
- Anomaly distribution profiles for each tool category
- Resource state monitoring interface
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class ToolCategory(Enum):
    """Tool categories based on resource consumption patterns."""
    CPU_BOUND = "cpu_bound"      # Compute-intensive, high retry on failure
    IO_BOUND = "io_bound"        # Disk I/O intensive, timeout on slow storage
    NETWORK_BOUND = "network_bound"  # Network calls, timeout on network issues
    MEMORY_BOUND = "memory_bound"  # Memory intensive, OOM risks
    MIXED = "mixed"              # Mixed workload, variable anomaly patterns
    NORMAL = "normal"            # Normal operation, low anomaly rate


class AnomalyType(Enum):
    """Types of tool execution anomalies."""
    RETRY = "retry"           # Tool takes 2x longer (runs twice)
    TIMEOUT = "timeout"       # Tool takes 5x longer (10s instead of 2s)
    HANG = "hang"             # Tool hangs for 10x longer (20s instead of 2s)
    EARLY_RETURN = "early_return"  # Tool returns early (0.1x = 0.2s)
    NORMAL = "normal"         # Normal execution


@dataclass
class AnomalyProfile:
    """Anomaly distribution profile for a tool or category."""
    anomaly_type: AnomalyType
    probability: float  # Probability of this anomaly type occurring
    duration_multiplier: float  # How much longer/shorter the tool takes
    
    def __repr__(self):
        return f"AnomalyProfile({self.anomaly_type.value}, p={self.probability:.2f}, mult={self.duration_multiplier})"


@dataclass 
class ToolMetadata:
    """Metadata for a tool including category and anomaly patterns."""
    tool_name: str
    category: ToolCategory
    base_duration: float = 2.0  # Base execution time in seconds
    anomaly_profile: List[AnomalyProfile] = field(default_factory=list)
    
    # For adaptive prediction
    observed_anomaly_rate: float = 0.0
    observed_retry_rate: float = 0.0
    observed_timeout_rate: float = 0.0
    observed_hang_rate: float = 0.0
    observed_early_return_rate: float = 0.0
    
    def get_anomaly_distribution(self) -> Dict[AnomalyType, float]:
        """Get probability distribution of anomaly types."""
        dist = {AnomalyType.NORMAL: 1.0}
        for profile in self.anomaly_profile:
            dist[AnomalyType.NORMAL] -= profile.probability
            dist[profile.anomaly_type] = profile.probability
        # Ensure non-negative
        dist[AnomalyType.NORMAL] = max(0.0, dist[AnomalyType.NORMAL])
        return dist
    
    def predict_execution_time(self, anomaly: AnomalyType = None) -> float:
        """Predict execution time given known anomaly type."""
        if anomaly is None:
            # Use expected value
            expected_mult = 1.0
            for profile in self.anomaly_profile:
                expected_mult += profile.probability * (profile.duration_multiplier - 1.0)
            return self.base_duration * expected_mult
        else:
            for profile in self.anomaly_profile:
                if profile.anomaly_type == anomaly:
                    return self.base_duration * profile.duration_multiplier
            return self.base_duration


@dataclass
class ResourceState:
    """Current state of system resources."""
    cpu_utilization: float = 0.5      # 0.0 to 1.0
    memory_pressure: float = 0.5       # 0.0 to 1.0
    io_congestion: float = 0.5        # 0.0 to 1.0
    network_congestion: float = 0.5    # 0.0 to 1.0
    cache_pressure: float = 0.5        # 0.0 to 1.0
    
    def get_overall_pressure(self) -> float:
        """Get overall system pressure (0.0 to 1.0)."""
        return (self.cpu_utilization + self.memory_pressure + 
                self.io_congestion + self.network_congestion + 
                self.cache_pressure) / 5.0
    
    def get_congestion_factor(self, category: ToolCategory) -> float:
        """Get congestion factor specific to tool category."""
        if category == ToolCategory.CPU_BOUND:
            return self.cpu_utilization
        elif category == ToolCategory.IO_BOUND:
            return self.io_congestion
        elif category == ToolCategory.NETWORK_BOUND:
            return self.network_congestion
        elif category == ToolCategory.MEMORY_BOUND:
            return self.memory_pressure
        else:
            return self.get_overall_pressure()


# Default anomaly profiles for each tool category
DEFAULT_CATEGORY_PROFILES: Dict[ToolCategory, List[AnomalyProfile]] = {
    ToolCategory.CPU_BOUND: [
        AnomalyProfile(AnomalyType.RETRY, 0.15, 2.0),  # 15% retry rate
        AnomalyProfile(AnomalyType.NORMAL, 0.85, 1.0),
    ],
    ToolCategory.IO_BOUND: [
        AnomalyProfile(AnomalyType.TIMEOUT, 0.20, 5.0),  # 20% timeout rate
        AnomalyProfile(AnomalyType.NORMAL, 0.80, 1.0),
    ],
    ToolCategory.NETWORK_BOUND: [
        AnomalyProfile(AnomalyType.TIMEOUT, 0.25, 5.0),  # 25% timeout rate (higher)
        AnomalyProfile(AnomalyType.RETRY, 0.10, 2.0),
        AnomalyProfile(AnomalyType.NORMAL, 0.65, 1.0),
    ],
    ToolCategory.MEMORY_BOUND: [
        AnomalyProfile(AnomalyType.HANG, 0.10, 10.0),  # 10% hang rate
        AnomalyProfile(AnomalyType.NORMAL, 0.90, 1.0),
    ],
    ToolCategory.MIXED: [
        AnomalyProfile(AnomalyType.RETRY, 0.10, 2.0),
        AnomalyProfile(AnomalyType.TIMEOUT, 0.10, 5.0),
        AnomalyProfile(AnomalyType.HANG, 0.05, 10.0),
        AnomalyProfile(AnomalyType.EARLY_RETURN, 0.10, 0.1),
        AnomalyProfile(AnomalyType.NORMAL, 0.65, 1.0),
    ],
    ToolCategory.NORMAL: [
        AnomalyProfile(AnomalyType.NORMAL, 1.0, 1.0),
    ],
}


class ToolClassifier:
    """Manages tool classification and anomaly prediction."""
    
    def __init__(self):
        self.tools: Dict[str, ToolMetadata] = {}
        self.category_profiles = DEFAULT_CATEGORY_PROFILES.copy()
        
    def register_tool(
        self, 
        tool_name: str, 
        category: ToolCategory,
        base_duration: float = 2.0,
        custom_profile: List[AnomalyProfile] = None
    ) -> ToolMetadata:
        """Register a tool with its category and anomaly profile."""
        profile = custom_profile if custom_profile else self.category_profiles.get(category, [])
        metadata = ToolMetadata(
            tool_name=tool_name,
            category=category,
            base_duration=base_duration,
            anomaly_profile=profile.copy()
        )
        self.tools[tool_name] = metadata
        return metadata
    
    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a tool."""
        return self.tools.get(tool_name)
    
    def predict_anomaly_probability(
        self, 
        tool_name: str, 
        resource_state: ResourceState = None,
        use_observations: bool = True
    ) -> Dict[AnomalyType, float]:
        """
        Predict anomaly probability for a tool.
        
        Combines:
        1. Historical observations (if use_observations=True)
        2. Resource state adjustments
        3. Tool category base rates
        """
        metadata = self.tools.get(tool_name)
        if not metadata:
            # Default to mixed profile
            metadata = ToolMetadata(
                tool_name=tool_name,
                category=ToolCategory.MIXED,
                anomaly_profile=self.category_profiles[ToolCategory.MIXED].copy()
            )
        
        # Start with base distribution from profile
        base_dist = metadata.get_anomaly_distribution()
        
        if not use_observations:
            return base_dist
        
        # Adjust based on observed anomaly rates (exponential moving average)
        adjusted_dist = base_dist.copy()
        
        # Apply observation-based adjustments
        total_adjustment = 0.0
        if metadata.observed_anomaly_rate > 0:
            # Weight observed anomaly rate
            obs_weight = 0.7  # 70% weight to observations
            for anomaly_type in [AnomalyType.RETRY, AnomalyType.TIMEOUT, 
                               AnomalyType.HANG, AnomalyType.EARLY_RETURN]:
                obs_rate = getattr(metadata, f'observed_{anomaly_type.value}_rate', 0.0)
                base_rate = base_dist.get(anomaly_type, 0.0)
                adjusted_rate = obs_weight * obs_rate + (1 - obs_weight) * base_rate
                adjusted_dist[anomaly_type] = adjusted_rate
                total_adjustment += adjusted_rate
            
            # Normalize
            if total_adjustment > 0:
                for k in adjusted_dist:
                    adjusted_dist[k] /= total_adjustment
        
        # Apply resource state adjustments
        if resource_state is not None:
            congestion = resource_state.get_congestion_factor(metadata.category)
            
            # High congestion increases timeout/hang probability
            if congestion > 0.7:
                timeout_adj = (congestion - 0.7) * 0.5
                adjusted_dist[AnomalyType.TIMEOUT] = min(1.0, 
                    adjusted_dist.get(AnomalyType.TIMEOUT, 0) + timeout_adj)
                adjusted_dist[AnomalyType.NORMAL] -= timeout_adj
            
            # Very low congestion might indicate early returns
            elif congestion < 0.3:
                early_adj = (0.3 - congestion) * 0.3
                adjusted_dist[AnomalyType.EARLY_RETURN] = min(1.0,
                    adjusted_dist.get(AnomalyType.EARLY_RETURN, 0) + early_adj)
                adjusted_dist[AnomalyType.NORMAL] -= early_adj
        
        # Ensure sum is 1.0
        total = sum(adjusted_dist.values())
        if total > 0:
            adjusted_dist = {k: v/total for k, v in adjusted_dist.items()}
        
        return adjusted_dist
    
    def update_observations(
        self, 
        tool_name: str, 
        anomaly_type: AnomalyType,
        execution_time: float,
        expected_time: float
    ) -> None:
        """Update observed anomaly rates for a tool."""
        if tool_name not in self.tools:
            # Auto-register as mixed category
            self.register_tool(tool_name, ToolCategory.MIXED)
        
        metadata = self.tools[tool_name]
        
        # Detect anomaly type from execution time
        if anomaly_type == AnomalyType.NORMAL:
            pass
        else:
            metadata.observed_anomaly_rate = (
                0.9 * metadata.observed_anomaly_rate + 0.1 * 1.0
            )
            obs_attr = f'observed_{anomaly_type.value}_rate'
            if hasattr(metadata, obs_attr):
                current_rate = getattr(metadata, obs_attr)
                setattr(metadata, obs_attr, 0.9 * current_rate + 0.1 * 1.0)
    
    def detect_anomaly_from_execution(
        self, 
        tool_name: str, 
        execution_time: float
    ) -> Tuple[AnomalyType, float]:
        """Detect anomaly type from execution time observation.
        
        Returns (anomaly_type, confidence)
        """
        metadata = self.tools.get(tool_name)
        if not metadata:
            base_time = 2.0
        else:
            base_time = metadata.base_duration
        
        ratio = execution_time / base_time
        
        if ratio > 3.0:
            return AnomalyType.HANG, min(1.0, (ratio - 3.0) / 7.0)
        elif ratio > 2.0:
            return AnomalyType.TIMEOUT, min(1.0, (ratio - 2.0) / 3.0)
        elif ratio > 1.5:
            return AnomalyType.RETRY, min(1.0, (ratio - 1.5) / 0.5)
        elif ratio < 0.3:
            return AnomalyType.EARLY_RETURN, min(1.0, (0.3 - ratio) / 0.3)
        else:
            return AnomalyType.NORMAL, 1.0


# Global classifier instance
_global_classifier: Optional[ToolClassifier] = None


def get_classifier() -> ToolClassifier:
    """Get global tool classifier instance."""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = ToolClassifier()
    return _global_classifier


def reset_classifier() -> None:
    """Reset global classifier."""
    global _global_classifier
    _global_classifier = None


# Convenience functions for common tool registrations
def register_cpu_bound_tool(tool_name: str, base_duration: float = 2.0) -> ToolMetadata:
    """Register a CPU-bound tool."""
    return get_classifier().register_tool(
        tool_name, ToolCategory.CPU_BOUND, base_duration
    )

def register_io_bound_tool(tool_name: str, base_duration: float = 2.0) -> ToolMetadata:
    """Register an I/O-bound tool."""
    return get_classifier().register_tool(
        tool_name, ToolCategory.IO_BOUND, base_duration
    )

def register_network_tool(tool_name: str, base_duration: float = 2.0) -> ToolMetadata:
    """Register a network-bound tool."""
    return get_classifier().register_tool(
        tool_name, ToolCategory.NETWORK_BOUND, base_duration
    )

def register_memory_tool(tool_name: str, base_duration: float = 2.0) -> ToolMetadata:
    """Register a memory-bound tool."""
    return get_classifier().register_tool(
        tool_name, ToolCategory.MEMORY_BOUND, base_duration
    )

def register_mixed_tool(tool_name: str, base_duration: float = 2.0) -> ToolMetadata:
    """Register a mixed workload tool."""
    return get_classifier().register_tool(
        tool_name, ToolCategory.MIXED, base_duration
    )
