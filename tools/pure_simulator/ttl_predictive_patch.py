"""
TTL Predictive Patch: Extends TTLManager with predictive anomaly-aware TTL adjustments.

This module provides:
- PredictiveTTLManager: Extends TTLManager with tool classification and anomaly prediction
- TTL adjustment strategies based on known/predicted anomalies
- Oracle mode for testing with perfect knowledge
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
import random

from tool_classifier import (
    ToolClassifier, ToolCategory, AnomalyType, ToolMetadata,
    ResourceState, get_classifier, AnomalyProfile
)
from simulator.ttl_manager import TTLManager


class PredictionAccuracy(Enum):
    """Prediction accuracy levels for testing."""
    PERFECT = 1.0      # Oracle: 100% accurate prediction
    HIGH = 0.85        # 85% accurate prediction
    MEDIUM = 0.70      # 70% accurate prediction
    LOW = 0.50         # 50% accurate prediction
    RANDOM = 0.0      # Random prediction (no better than baseline)


@dataclass
class TTLAdjustmentConfig:
    """Configuration for TTL adjustment factors."""
    # Multipliers for predicted anomalies
    timeout_extension: float = 3.0      # TTL extension for predicted timeout
    retry_extension: float = 2.0        # TTL extension for predicted retry
    hang_extension: float = 5.0         # TTL extension for predicted hang
    early_return_shorten: float = 0.3   # TTL shorten for predicted early return
    
    # Thresholds
    high_anomaly_threshold: float = 0.7  # Above this: extend TTL
    low_anomaly_threshold: float = 0.3   # Below this: shorten TTL
    
    # Strategy selection
    enable_extension: bool = True       # Enable TTL extension strategy
    enable_shortening: bool = True       # Enable TTL shortening strategy


@dataclass
class AnomalyKnowledge:
    """
    Represents known/predicted anomaly information for a tool call.
    
    In oracle mode: we know exactly what will happen
    In prediction mode: we have probabilistic predictions
    """
    tool_name: str
    predicted_anomaly: Optional[AnomalyType] = None
    anomaly_probability: float = 0.0
    actual_anomaly: Optional[AnomalyType] = None  # For oracle mode
    is_known: bool = False  # Oracle has perfect knowledge


class PredictiveTTLManager:
    """
    Extended TTL Manager with predictive anomaly-aware TTL adjustments.
    
    This manager:
    1. Receives tool classification and anomaly predictions
    2. Adjusts TTL based on predicted anomalies
    3. Supports oracle mode for testing with perfect knowledge
    """
    
    def __init__(
        self,
        ttl_manager: TTLManager,
        tool_classifier: ToolClassifier = None,
        accuracy: PredictionAccuracy = PredictionAccuracy.PERFECT,
        config: TTLAdjustmentConfig = None,
        enable_predictive_patch: bool = True,
    ):
        """
        Initialize PredictiveTTLManager.
        
        Args:
            ttl_manager: The base TTLManager to extend
            tool_classifier: Tool classifier instance
            accuracy: Prediction accuracy level
            config: TTL adjustment configuration
            enable_predictive_patch: Whether to enable predictive adjustments
        """
        self.base_ttl_manager = ttl_manager
        self.tool_classifier = tool_classifier or get_classifier()
        self.accuracy = accuracy
        self.config = config or TTLAdjustmentConfig()
        self.enable_predictive_patch = enable_predictive_patch
        
        # Track predictions and adjustments
        self.predictions: List[AnomalyKnowledge] = []
        self.adjustments: List[Tuple[str, float, float, str]] = []  # (tool_name, base_ttl, adjusted_ttl, reason)
        
        # Oracle mode: stores ground truth anomalies
        self._oracle_knowledge: Dict[str, AnomalyKnowledge] = {}
        
        # Statistics
        self.stats = PredictiveTTLStats()
    
    def inject_oracle_knowledge(
        self, 
        request_id: str,  # Changed from tool_name to request_id
        anomaly: AnomalyType,
        probability: float = 1.0
    ) -> None:
        """Inject oracle knowledge for a tool call (for testing).
        
        Args:
            request_id: Request ID (e.g., 'prog_000_turn_0' or 'p0_t0')
            anomaly: The predicted anomaly type
            probability: Probability of the anomaly (1.0 for oracle)
        """
        self._oracle_knowledge[request_id] = AnomalyKnowledge(
            tool_name=request_id,  # Store request_id as tool_name too
            predicted_anomaly=anomaly,
            anomaly_probability=probability,
            actual_anomaly=anomaly,
            is_known=True
        )
    
    def clear_oracle_knowledge(self) -> None:
        """Clear all oracle knowledge."""
        self._oracle_knowledge.clear()
    
    def predict_anomaly_for_tool(
        self, 
        tool_name: str,
        resource_state: ResourceState = None,
        request_id: str = None  # NEW: specific request ID
    ) -> Tuple[Optional[AnomalyType], float]:
        """
        Predict anomaly for a tool call.
        
        In PERFECT mode (oracle): Only returns prediction if oracle knowledge was injected.
        Otherwise returns (None, 0.0) to indicate no prediction available.
        
        In non-PERFECT mode: Uses tool classifier for prediction.
        
        Returns (predicted_anomaly_type, probability)
        """
        # Check oracle knowledge by request_id FIRST (more specific)
        if request_id and request_id in self._oracle_knowledge:
            oracle = self._oracle_knowledge[request_id]
            if self.accuracy == PredictionAccuracy.PERFECT:
                return oracle.predicted_anomaly, oracle.anomaly_probability
            else:
                # Apply prediction noise
                if random.random() < self.accuracy.value:
                    return oracle.predicted_anomaly, oracle.anomaly_probability
                else:
                    return None, 0.0
        
        # Check oracle knowledge by tool_name (less specific, fallback)
        if tool_name and tool_name in self._oracle_knowledge:
            oracle = self._oracle_knowledge[tool_name]
            if self.accuracy == PredictionAccuracy.PERFECT:
                return oracle.predicted_anomaly, oracle.anomaly_probability
            else:
                if random.random() < self.accuracy.value:
                    return oracle.predicted_anomaly, oracle.anomaly_probability
                else:
                    return None, 0.0
        
        # KEY FIX: In PERFECT mode (oracle), if no oracle knowledge exists,
        # do NOT use tool classifier - just return no prediction
        if self.accuracy == PredictionAccuracy.PERFECT:
            return None, 0.0
        
        # Use tool classifier for prediction (only in non-PERFECT mode)
        dist = self.tool_classifier.predict_anomaly_probability(
            tool_name, resource_state, use_observations=True
        )
        
        # Find most likely anomaly
        max_anomaly = None
        max_prob = 0.0
        for anomaly_type, prob in dist.items():
            if anomaly_type != AnomalyType.NORMAL and prob > max_prob:
                max_prob = prob
                max_anomaly = anomaly_type
        
        return max_anomaly, max_prob
    
    def calculate_adjusted_ttl(
        self,
        base_ttl: float,
        tool_name: str,
        anomaly: Optional[AnomalyType],
        probability: float,
        current_time: float = 0.0
    ) -> Tuple[float, str]:
        """
        Calculate TTL adjusted based on predicted anomaly.
        
        Only applies adjustments when:
        1. Predictive patch is enabled AND
        2. Anomaly is predicted AND
        3. Either:
           - Accuracy is PERFECT (oracle mode with injected knowledge)
           - OR probability is above high_anomaly_threshold (non-oracle mode)
        
        Args:
            base_ttl: Base TTL from original TTLManager
            tool_name: Name of the tool being executed
            anomaly: Predicted anomaly type
            probability: Probability of the anomaly (1.0 for oracle-injected)
            current_time: Current simulation time
            
        Returns:
            (adjusted_ttl, adjustment_reason)
        """
        if not self.enable_predictive_patch:
            return base_ttl, "disabled"
        
        # Key fix: Only apply adjustments if:
        # 1. Anomaly is predicted AND
        # 2. Probability is high (oracle-injected = 1.0, or high confidence)
        # This prevents incorrect adjustments when no oracle knowledge was injected
        if anomaly is None:
            return base_ttl, "no_anomaly"
        
        # Oracle mode: only adjust if this was explicitly injected (probability = 1.0)
        if self.accuracy == PredictionAccuracy.PERFECT:
            if probability < 1.0:
                # No oracle knowledge injected for this request, use original TTL
                return base_ttl, "no_oracle_knowledge"
            # Oracle knowledge exists, apply full extension
            return self._apply_full_extension(base_ttl, anomaly)
        
        # Non-oracle mode: require high confidence
        if probability < self.config.high_anomaly_threshold:
            return base_ttl, "low_confidence"
        
        return self._apply_full_extension(base_ttl, anomaly)
    
    def _apply_full_extension(
        self,
        base_ttl: float,
        anomaly: AnomalyType
    ) -> Tuple[float, str]:
        """Apply full TTL extension based on anomaly type."""
        if anomaly == AnomalyType.TIMEOUT and self.config.enable_extension:
            adjusted = base_ttl * self.config.timeout_extension
            return min(adjusted, 60.0), "predicted_timeout_extend"
        
        elif anomaly == AnomalyType.RETRY and self.config.enable_extension:
            adjusted = base_ttl * self.config.retry_extension
            return min(adjusted, 30.0), "predicted_retry_extend"
        
        elif anomaly == AnomalyType.HANG and self.config.enable_extension:
            adjusted = base_ttl * self.config.hang_extension
            return min(adjusted, 120.0), "predicted_hang_extend"
        
        elif anomaly == AnomalyType.EARLY_RETURN and self.config.enable_shortening:
            adjusted = base_ttl * self.config.early_return_shorten
            return max(adjusted, 0.5), "predicted_early_return_shorten"
        
        return base_ttl, "no_adjustment"
    
    def calc_predictive_ttl(
        self,
        program_id: str,
        tool_name: Optional[str],
        miss_tokens: int,
        node_size: int,
        turn_index: int,
        total_turns: int,
        current_time: float,
        l2_enabled: bool = False,
        l2_reload_penalty: float = None,
        resource_state: ResourceState = None,
    ) -> Tuple[float, str, int, float]:
        """
        Calculate adaptive TTL with predictive anomaly awareness.
        
        This extends the base TTLManager.calc_adaptive_ttl with:
        1. Anomaly prediction for the current tool
        2. TTL adjustment based on predicted anomalies
        
        Returns:
            (adjusted_ttl, strategy, cdf_samples, adjustment_factor)
        """
        # Get base TTL from original TTLManager
        base_ttl, strategy, cdf_samples = self.base_ttl_manager.calc_adaptive_ttl(
            program_id=program_id,
            tool_name=tool_name,
            miss_tokens=miss_tokens,
            node_size=node_size,
            turn_index=turn_index,
            total_turns=total_turns,
            current_time=current_time,
            l2_enabled=l2_enabled,
            l2_reload_penalty=l2_reload_penalty,
        )
        
        # Record prediction
        predicted_anomaly, prob = self.predict_anomaly_for_tool(tool_name, resource_state)
        
        # Track prediction
        knowledge = AnomalyKnowledge(
            tool_name=tool_name or "unknown",
            predicted_anomaly=predicted_anomaly,
            anomaly_probability=prob,
        )
        self.predictions.append(knowledge)
        
        # Calculate adjusted TTL
        adjusted_ttl, reason = self.calculate_adjusted_ttl(
            base_ttl, tool_name, predicted_anomaly, prob, current_time
        )
        
        # Track adjustment
        self.adjustments.append((tool_name, base_ttl, adjusted_ttl, reason))
        
        # Update stats
        if reason != "no_adjustment" and reason != "disabled":
            self.stats.total_adjustments += 1
            if "extend" in reason:
                self.stats.extensions += 1
            elif "shorten" in reason:
                self.stats.shortenings += 1
        
        adjustment_factor = adjusted_ttl / max(base_ttl, 0.1)
        self.stats.avg_adjustment_factor = (
            (self.stats.avg_adjustment_factor * (self.stats.total_adjustments - 1) + adjustment_factor)
            / max(self.stats.total_adjustments, 1)
        )
        
        return adjusted_ttl, f"{strategy}+{reason}", cdf_samples, adjustment_factor
    
    def record_actual_outcome(
        self,
        tool_name: str,
        actual_duration: float,
        predicted_anomaly: Optional[AnomalyType],
        was_correct: bool = None
    ) -> None:
        """Record actual outcome for analysis."""
        if was_correct is None:
            # Detect actual anomaly from duration
            detected, _ = self.tool_classifier.detect_anomaly_from_execution(
                tool_name, actual_duration
            )
            was_correct = (detected == predicted_anomaly)
        
        if was_correct:
            self.stats.correct_predictions += 1
        else:
            self.stats.incorrect_predictions += 1
        
        self.stats.total_recorded += 1
    
    def get_prediction_accuracy(self) -> float:
        """Get current prediction accuracy."""
        if self.stats.total_recorded == 0:
            return 0.0
        return self.stats.correct_predictions / self.stats.total_recorded
    
    def get_adjustment_summary(self) -> Dict[str, Any]:
        """Get summary of TTL adjustments made."""
        extensions = sum(1 for _, _, _, r in self.adjustments if "extend" in r)
        shortenings = sum(1 for _, _, _, r in self.adjustments if "shorten" in r)
        
        return {
            'total_adjustments': len(self.adjustments),
            'extensions': extensions,
            'shortenings': shortenings,
            'avg_adjustment_factor': self.stats.avg_adjustment_factor,
            'prediction_accuracy': self.get_prediction_accuracy(),
        }


@dataclass
class PredictiveTTLStats:
    """Statistics for predictive TTL manager."""
    total_adjustments: int = 0
    extensions: int = 0
    shortenings: int = 0
    avg_adjustment_factor: float = 1.0
    correct_predictions: int = 0
    incorrect_predictions: int = 0
    total_recorded: int = 0


class TTLPatchFactory:
    """Factory for creating predictive TTL managers with different configurations."""
    
    @staticmethod
    def create_oracle_manager(ttl_manager: TTLManager) -> PredictiveTTLManager:
        """Create oracle manager with perfect prediction."""
        return PredictiveTTLManager(
            ttl_manager=ttl_manager,
            accuracy=PredictionAccuracy.PERFECT,
            config=TTLAdjustmentConfig(
                timeout_extension=3.0,
                retry_extension=2.0,
                hang_extension=5.0,
                early_return_shorten=0.3,
            ),
            enable_predictive_patch=True,
        )
    
    @staticmethod
    def create_high_accuracy_manager(
        ttl_manager: TTLManager,
        accuracy: float = 0.85
    ) -> PredictiveTTLManager:
        """Create manager with specified accuracy."""
        pred_acc = PredictionAccuracy.HIGH
        if accuracy <= 0.5:
            pred_acc = PredictionAccuracy.LOW
        elif accuracy <= 0.7:
            pred_acc = PredictionAccuracy.MEDIUM
        elif accuracy <= 0.85:
            pred_acc = PredictionAccuracy.HIGH
            
        return PredictiveTTLManager(
            ttl_manager=ttl_manager,
            accuracy=pred_acc,
            enable_predictive_patch=True,
        )
    
    @staticmethod
    def create_conservative_manager(ttl_manager: TTLManager) -> PredictiveTTLManager:
        """Create conservative manager with smaller adjustments."""
        return PredictiveTTLManager(
            ttl_manager=ttl_manager,
            accuracy=PredictionAccuracy.HIGH,
            config=TTLAdjustmentConfig(
                timeout_extension=1.5,
                retry_extension=1.3,
                hang_extension=2.0,
                early_return_shorten=0.7,
            ),
            enable_predictive_patch=True,
        )
    
    @staticmethod
    def create_aggressive_manager(ttl_manager: TTLManager) -> PredictiveTTLManager:
        """Create aggressive manager with larger adjustments."""
        return PredictiveTTLManager(
            ttl_manager=ttl_manager,
            accuracy=PredictionAccuracy.HIGH,
            config=TTLAdjustmentConfig(
                timeout_extension=5.0,
                retry_extension=3.0,
                hang_extension=10.0,
                early_return_shorten=0.2,
            ),
            enable_predictive_patch=True,
        )


def apply_ttl_patch_to_simulator(
    simulator,
    prediction_accuracy: PredictionAccuracy = PredictionAccuracy.PERFECT,
    config: TTLAdjustmentConfig = None
) -> PredictiveTTLManager:
    """
    Apply TTL predictive patch to an existing simulator.

    This wraps the simulator's TTLManager with a PredictiveTTLManager.
    """
    if simulator.ttl_manager is None:
        raise ValueError("Simulator must have TTL enabled")

    # Store the original method BEFORE creating the patch
    original_calc = simulator.ttl_manager.calc_adaptive_ttl

    patch = PredictiveTTLManager(
        ttl_manager=simulator.ttl_manager,
        accuracy=prediction_accuracy,
        config=config,
        enable_predictive_patch=True,
    )

    # Use original_calc directly instead of going through the wrapped method
    def wrapped_calc(*args, **kwargs):
        # Get base TTL from the original method
        base_ttl, strategy, cdf_samples = original_calc(*args, **kwargs)

        # Get tool_name from kwargs or args
        tool_name = kwargs.get('tool_name')
        if tool_name is None and len(args) > 1:
            tool_name = args[1]
        
        # Get program_id and turn_index to construct request_id
        program_id = kwargs.get('program_id')
        if program_id is None and len(args) > 0:
            program_id = args[0]
        
        turn_index = kwargs.get('turn_index')
        if turn_index is None and len(args) > 4:
            turn_index = args[4]
        
        # Construct request_id: {program_id}_turn_{turn_index}
        # This matches the rid format used in generate_requests()
        if program_id is not None and turn_index is not None:
            request_id = f"{program_id}_turn_{turn_index}"
        else:
            request_id = program_id

        # Predict anomaly with request_id for more specific oracle lookup
        predicted_anomaly, prob = patch.predict_anomaly_for_tool(
            tool_name, request_id=request_id
        )

        # Calculate adjusted TTL
        adjusted_ttl, reason = patch.calculate_adjusted_ttl(
            base_ttl, tool_name, predicted_anomaly, prob
        )

        # Track prediction and adjustment
        patch.predictions.append(AnomalyKnowledge(
            tool_name=tool_name or "unknown",
            predicted_anomaly=predicted_anomaly,
            anomaly_probability=prob,
        ))
        patch.adjustments.append((tool_name, base_ttl, adjusted_ttl, reason))

        # Update stats
        if reason != "no_adjustment" and reason != "disabled":
            patch.stats.total_adjustments += 1
            if "extend" in reason:
                patch.stats.extensions += 1
            elif "shorten" in reason:
                patch.stats.shortenings += 1

        adjustment_factor = adjusted_ttl / max(base_ttl, 0.1)
        if patch.stats.total_adjustments > 0:
            patch.stats.avg_adjustment_factor = (
                (patch.stats.avg_adjustment_factor * (patch.stats.total_adjustments - 1) + adjustment_factor)
                / patch.stats.total_adjustments
            )
        else:
            patch.stats.avg_adjustment_factor = adjustment_factor

        return adjusted_ttl, f"{strategy}+{reason}", cdf_samples

    simulator.ttl_manager.calc_adaptive_ttl = wrapped_calc
    simulator.ttl_manager._predictive_patch = patch

    return patch
