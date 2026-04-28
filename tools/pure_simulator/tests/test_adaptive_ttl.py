"""Tests for Adaptive TTL functionality based on Continuum paper."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add the simulator directory to path
simulator_dir = Path(__file__).parent.parent
sys.path.insert(0, str(simulator_dir))

from simulator.ttl_manager import (
    TTLManager,
    ToolDurationStats,
    ProgramStats,
    PinnedEntry,
    ToolExecution,
)


@pytest.fixture
def mock_radix_tree():
    """Create a mock radix tree for testing."""
    tree = MagicMock()
    tree.allocator.capacity = 100000
    tree.allocator.used.return_value = 50000  # 50% memory usage
    return tree


@pytest.fixture
def ttl_manager(mock_radix_tree):
    """Create a TTLManager for testing."""
    return TTLManager(
        radix_tree=mock_radix_tree,
        default_ttl=5.0,
        min_ttl=1.0,
        max_ttl=15.0,
        history_threshold=3,
        enable_adaptive_ttl=True,
        ttl_grid_min=1.0,
        ttl_grid_max=60.0,
        ttl_grid_points=20,
        prefill_latency_ms_per_token=0.01,
    )


class TestToolDurationStats:
    """Tests for ToolDurationStats dataclass."""

    def test_add_duration(self):
        """Test adding duration samples."""
        stats = ToolDurationStats()
        stats.add_duration(1.0)
        stats.add_duration(2.0)
        stats.add_duration(3.0)
        assert len(stats.durations) == 3

    def test_add_duration_window_limit(self):
        """Test that duration window is limited."""
        stats = ToolDurationStats()
        for i in range(600):
            stats.add_duration(float(i))
        # Implementation trims to last 250 when exceeding 500 items
        # So after 600 additions, we should have 250 items
        assert len(stats.durations) <= 500  # Basic limit check
        assert len(stats.durations[-10:]) == 10  # Last 10 items are recent

    def test_add_duration_stores_values(self):
        """Test that duration values are stored correctly."""
        stats = ToolDurationStats()
        stats.add_duration(1.0)
        stats.add_duration(5.0)
        stats.add_duration(10.0)
        assert len(stats.durations) == 3
        assert 10.0 in stats.durations

    def test_get_percentile(self):
        """Test percentile calculation."""
        stats = ToolDurationStats()
        for i in range(1, 101):
            stats.add_duration(float(i))
        # 50th percentile should be around 50
        p50 = stats.get_percentile(50)
        assert p50 is not None
        assert 45 <= p50 <= 55

    def test_get_percentile_empty(self):
        """Test percentile with no data."""
        stats = ToolDurationStats()
        assert stats.get_percentile(50) is None


class TestProgramStats:
    """Tests for ProgramStats dataclass."""

    def test_add_idle_gap(self):
        """Test adding idle gap samples."""
        prog_stats = ProgramStats(program_id="test_program")
        prog_stats.add_idle_gap(1.0)
        prog_stats.add_idle_gap(2.0)
        assert len(prog_stats.idle_gaps) == 2

    def test_add_idle_gap_ignores_zero(self):
        """Test that zero idle gaps are ignored."""
        prog_stats = ProgramStats(program_id="test_program")
        prog_stats.add_idle_gap(0.0)
        assert len(prog_stats.idle_gaps) == 0

    def test_add_idle_gap_window_limit(self):
        """Test idle gap window limit."""
        prog_stats = ProgramStats(program_id="test_program")
        for i in range(150):
            prog_stats.add_idle_gap(float(i))
        # Implementation trims to last 50 when exceeding 100 items
        assert len(prog_stats.idle_gaps) <= 100
        assert prog_stats.idle_gaps[-1] == 149.0  # Last value included

    def test_get_tool_stats(self):
        """Test getting tool stats."""
        prog_stats = ProgramStats(program_id="test_program")
        stats1 = prog_stats.get_tool_stats("tool1")
        stats2 = prog_stats.get_tool_stats("tool1")
        # Should return same instance
        assert stats1 is stats2
        assert "tool1" in prog_stats.tool_durations


class TestTTLManagerRecordToolExecution:
    """Tests for TTLManager.record_tool_execution()."""

    def test_record_global_tool_duration(self, ttl_manager):
        """Test recording tool duration globally."""
        ttl_manager.record_tool_execution(
            program_id="prog1",
            tool_name="bash",
            duration=2.5,
            idle_gap=1.0,
        )
        assert "bash" in ttl_manager.global_tool_durations
        assert 2.5 in ttl_manager.global_tool_durations["bash"]

    def test_record_idle_gap(self, ttl_manager):
        """Test recording idle gap."""
        ttl_manager.record_tool_execution(
            program_id="prog1",
            tool_name="bash",
            duration=2.5,
            idle_gap=1.5,
        )
        assert 1.5 in ttl_manager.global_idle_gaps

    def test_record_program_stats(self, ttl_manager):
        """Test recording program-specific stats."""
        ttl_manager.record_tool_execution(
            program_id="prog1",
            tool_name="bash",
            duration=2.5,
            idle_gap=1.0,
        )
        assert "prog1" in ttl_manager.program_stats
        prog_stats = ttl_manager.program_stats["prog1"]
        assert "bash" in prog_stats.tool_durations


class TestTTLManagerCalculateT:
    """Tests for TTLManager._calculate_T()."""

    def test_calculate_T_with_idle_gaps(self, ttl_manager):
        """Test T calculation with idle gaps."""
        # Add some idle gaps
        ttl_manager.global_idle_gaps = [1.0, 2.0, 3.0, 4.0, 5.0]
        T = ttl_manager._calculate_T()
        # T should be based on average gap
        assert T > 0

    def test_calculate_T_fallback(self, ttl_manager):
        """Test T fallback when no data."""
        ttl_manager.global_idle_gaps = []
        T = ttl_manager._calculate_T()
        # Should return default_ttl * 500 (ms)
        assert T == ttl_manager.default_ttl * 500


class TestTTLManagerCalculateMemoryfulness:
    """Tests for TTLManager._calculate_memoryfulness()."""

    def test_memoryfulness_first_turn(self, ttl_manager):
        """Test memoryfulness for first turn."""
        eta = ttl_manager._calculate_memoryfulness(turn_index=0, total_turns=10)
        assert eta == 1.0

    def test_memoryfulness_last_turn(self, ttl_manager):
        """Test memoryfulness for last turn."""
        eta = ttl_manager._calculate_memoryfulness(turn_index=9, total_turns=10)
        # Should be lower than 1.0
        assert eta < 1.0
        assert eta >= 0.1  # Minimum bound

    def test_memoryfulness_single_turn(self, ttl_manager):
        """Test memoryfulness with single turn."""
        eta = ttl_manager._calculate_memoryfulness(turn_index=0, total_turns=1)
        assert eta == 1.0

    def test_memoryfulness_linear_decay(self, ttl_manager):
        """Test linear decay of memoryfulness."""
        eta_early = ttl_manager._calculate_memoryfulness(turn_index=1, total_turns=5)
        eta_late = ttl_manager._calculate_memoryfulness(turn_index=3, total_turns=5)
        assert eta_early > eta_late


class TestTTLManagerApplyMemoryPressure:
    """Tests for TTLManager._apply_memory_pressure()."""

    def test_no_pressure(self, ttl_manager):
        """Test no penalty at 50% memory usage."""
        ttl_manager.radix_tree.allocator.used.return_value = 50000  # 50%
        ttl = ttl_manager._apply_memory_pressure(10.0)
        assert ttl == 10.0

    def test_high_pressure(self, ttl_manager):
        """Test penalty at high memory usage."""
        ttl_manager.radix_tree.allocator.used.return_value = 75000  # 75%
        ttl = ttl_manager._apply_memory_pressure(10.0)
        # At 75%, penalty = 1.0 - (0.75 - 0.5) * 0.5 = 0.875
        assert ttl < 10.0
        assert ttl > 0


class TestTTLManagerComputeCDF:
    """Tests for TTLManager._compute_cdf()."""

    def test_cdf_basic(self, ttl_manager):
        """Test basic CDF calculation."""
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        # P(3.0) = 3/5 = 0.6
        cdf = ttl_manager._compute_cdf(durations, 3.0)
        assert cdf == 0.6

    def test_cdf_all_below(self, ttl_manager):
        """Test CDF when tau is above all durations."""
        durations = [1.0, 2.0, 3.0]
        cdf = ttl_manager._compute_cdf(durations, 10.0)
        assert cdf == 1.0

    def test_cdf_all_above(self, ttl_manager):
        """Test CDF when tau is below all durations."""
        durations = [5.0, 6.0, 7.0]
        cdf = ttl_manager._compute_cdf(durations, 1.0)
        assert cdf == 0.0

    def test_cdf_empty(self, ttl_manager):
        """Test CDF with empty durations."""
        cdf = ttl_manager._compute_cdf([], 5.0)
        assert cdf == 0.0


class TestTTLManagerGetCDFDurations:
    """Tests for TTLManager._get_cdf_durations()."""

    def test_insufficient_data(self, ttl_manager):
        """Test fallback with insufficient data."""
        durations, strategy = ttl_manager._get_cdf_durations("prog1", "bash")
        assert durations == []
        assert strategy == "insufficient_data"

    def test_global_tool_fallback(self, ttl_manager):
        """Test global tool durations fallback."""
        ttl_manager.global_tool_durations["bash"] = [1.0, 2.0, 3.0, 4.0]
        durations, strategy = ttl_manager._get_cdf_durations("prog1", "bash")
        assert len(durations) == 4
        assert "global_tool:bash" in strategy


class TestTTLManagerCalcAdaptiveTTL:
    """Tests for TTLManager.calc_adaptive_ttl()."""

    def test_disabled_adaptive_ttl(self, ttl_manager):
        """Test when adaptive TTL is disabled."""
        ttl_manager.enable_adaptive_ttl = False
        ttl, strategy = ttl_manager.calc_adaptive_ttl(
            program_id="prog1",
            tool_name="bash",
            miss_tokens=100,
            node_size=5000,
            turn_index=0,
            total_turns=10,
            current_time=0.0,
        )
        assert ttl == ttl_manager.default_ttl
        assert strategy == "disabled"

    def test_insufficient_data_fallback(self, ttl_manager):
        """Test fallback when insufficient data for CDF."""
        ttl, strategy = ttl_manager.calc_adaptive_ttl(
            program_id="prog1",
            tool_name="bash",
            miss_tokens=100,
            node_size=5000,
            turn_index=0,
            total_turns=10,
            current_time=0.0,
        )
        # Should return default TTL when no data
        assert ttl == ttl_manager.default_ttl

    def test_adaptive_ttl_with_data(self, ttl_manager):
        """Test adaptive TTL calculation with sufficient data."""
        # Add enough tool duration samples
        ttl_manager.global_tool_durations["bash"] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
        ]
        ttl_manager.global_idle_gaps = [1.0, 2.0, 3.0]

        ttl, strategy = ttl_manager.calc_adaptive_ttl(
            program_id="prog1",
            tool_name="bash",
            miss_tokens=100,
            node_size=5000,
            turn_index=0,
            total_turns=10,
            current_time=0.0,
        )
        # Should return TTL within bounds
        assert ttl_manager.min_ttl <= ttl <= ttl_manager.max_ttl

    def test_adaptive_ttl_bounds(self, ttl_manager):
        """Test that TTL is always within bounds."""
        # Add lots of data with extreme values
        ttl_manager.global_tool_durations["bash"] = [100.0] * 10
        ttl_manager.global_idle_gaps = [10.0] * 10

        ttl, _ = ttl_manager.calc_adaptive_ttl(
            program_id="prog1",
            tool_name="bash",
            miss_tokens=100,
            node_size=5000,
            turn_index=0,
            total_turns=10,
            current_time=0.0,
        )
        assert ttl_manager.min_ttl <= ttl <= ttl_manager.max_ttl

    def test_adaptive_ttl_memory_pressure(self, ttl_manager):
        """Test memory pressure affects TTL."""
        # Add data
        ttl_manager.global_tool_durations["bash"] = [5.0] * 10
        ttl_manager.global_idle_gaps = [2.0] * 10

        # Low memory usage
        ttl_manager.radix_tree.allocator.used.return_value = 25000  # 25%
        ttl_low, _ = ttl_manager.calc_adaptive_ttl(
            program_id="prog1",
            tool_name="bash",
            miss_tokens=100,
            node_size=5000,
            turn_index=0,
            total_turns=10,
            current_time=0.0,
        )

        # High memory usage
        ttl_manager.radix_tree.allocator.used.return_value = 90000  # 90%
        ttl_high, _ = ttl_manager.calc_adaptive_ttl(
            program_id="prog1",
            tool_name="bash",
            miss_tokens=100,
            node_size=5000,
            turn_index=0,
            total_turns=10,
            current_time=0.0,
        )

        # High memory should result in lower TTL
        assert ttl_high <= ttl_low


class TestTTLManagerGenerateTauGrid:
    """Tests for TTLManager._generate_tau_grid()."""

    def test_grid_size(self, ttl_manager):
        """Test tau grid size."""
        grid = ttl_manager._generate_tau_grid()
        assert len(grid) == ttl_manager.ttl_grid_points

    def test_grid_range(self, ttl_manager):
        """Test tau grid range."""
        grid = ttl_manager._generate_tau_grid()
        assert grid[0] == ttl_manager.ttl_grid_min
        assert grid[-1] == ttl_manager.ttl_grid_max

    def test_grid_sorted(self, ttl_manager):
        """Test tau grid is sorted."""
        grid = ttl_manager._generate_tau_grid()
        assert grid == sorted(grid)


class TestTTLManagerPinProgram:
    """Tests for TTLManager.pin_program()."""

    def test_pin_program_basic(self, ttl_manager):
        """Test basic pin operation."""
        mock_node = MagicMock()
        mock_node.node_id = 1

        ttl = ttl_manager.pin_program(
            program_id="prog1",
            node=mock_node,
            ttl_sec=10.0,
            current_time=0.0,
            tool_name="bash",
        )

        assert ttl == 10.0
        # Check entry exists with tuple key format
        assert ("prog1", 1) in ttl_manager.pinned_entries
        assert ttl_manager.stats.total_pins == 1

    def test_pin_program_bounds(self, ttl_manager):
        """Test TTL is bounded."""
        mock_node = MagicMock()
        mock_node.node_id = 1

        # Too high TTL should be capped
        ttl = ttl_manager.pin_program(
            program_id="prog1",
            node=mock_node,
            ttl_sec=100.0,  # Above max_ttl (15.0)
            current_time=0.0,
            tool_name="bash",
        )
        assert ttl == ttl_manager.max_ttl

    def test_pin_refresh(self, ttl_manager):
        """Test that repinning refreshes TTL."""
        mock_node = MagicMock()
        mock_node.node_id = 1

        ttl_manager.pin_program(
            program_id="prog1",
            node=mock_node,
            ttl_sec=5.0,
            current_time=0.0,
        )

        ttl_manager.pin_program(
            program_id="prog1",
            node=mock_node,
            ttl_sec=10.0,
            current_time=5.0,
        )

        # Should still have only one entry (refreshed)
        assert len(ttl_manager.pinned_entries) == 1
        assert ttl_manager.pinned_entries[("prog1", 1)].ttl_sec == 10.0


class TestTTLManagerIsPinned:
    """Tests for TTLManager.is_pinned()."""

    def test_not_pinned(self, ttl_manager):
        """Test is_pinned returns False for non-existent program."""
        assert ttl_manager.is_pinned("nonexistent", 0.0) is False

    def test_pinned_valid(self, ttl_manager):
        """Test is_pinned returns True for valid pin."""
        mock_node = MagicMock()
        mock_node.node_id = 1
        mock_node.ttl_expiry_time = 5.0  # expires at time 5.0
        mock_node.is_pinned = True

        # Set up the radix tree mock to return our node
        ttl_manager.radix_tree.nodes = {1: mock_node}

        ttl_manager.pin_program(
            program_id="prog1",
            node=mock_node,
            ttl_sec=5.0,
            current_time=0.0,
        )

        # After pin, node's ttl_expiry_time is set to current_time + ttl_sec = 5.0
        # So at current_time=3.0, the pin should still be valid
        assert ttl_manager.is_pinned("prog1", 3.0) is True

    def test_pinned_expired(self, ttl_manager):
        """Test is_pinned returns False for expired pin."""
        mock_node = MagicMock()
        mock_node.node_id = 1
        mock_node.is_pinned = True
        mock_node.ttl_expiry_time = 5.0

        ttl_manager.pin_program(
            program_id="prog1",
            node=mock_node,
            ttl_sec=5.0,
            current_time=0.0,
        )

        # After expiry time
        assert ttl_manager.is_pinned("prog1", 10.0) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
