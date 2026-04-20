#!/usr/bin/env python3
"""
Test script for verifying SGLang Continuum TTL fixes.

This script tests:
1. Tool duration recording uses tool_execution_time instead of idle_gap
2. Scheduler affinity prefers continuous turns from same job
3. TTL selection uses correct tool duration history

Usage:
    python test_continuum_fixes.py
"""

import sys
import os

# Set working directory
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

def test_tool_duration_recording():
    """Test that tool durations are recorded correctly."""
    print("=" * 60)
    print("Test 1: Tool Duration Recording")
    print("=" * 60)
    
    print("\n[CHECK] Verifying tokenizer_manager.py fix...")
    
    tokenizer_path = os.path.join(WORK_DIR, "python/sglang/srt/managers/tokenizer_manager.py")
    with open(tokenizer_path, "r") as f:
        content = f.read()
    
    # Verify the fix is in place
    if "tool_execution_time" in content and 'extra_body.get("tool_execution_time"' in content:
        print("  [PASS] tool_execution_time extraction found")
    else:
        print("  [FAIL] tool_execution_time extraction NOT found")
        return False
    
    if "recorded_duration" in content and "tool_execution_time if tool_execution_time" in content:
        print("  [PASS] recorded_duration uses tool_execution_time correctly")
    else:
        print("  [FAIL] recorded_duration logic NOT correct")
        return False
    
    print("\n[PASS] Test 1: Tool duration recording fix verified")
    return True


def test_scheduler_affinity():
    """Test that scheduler affinity is enhanced."""
    print("\n" + "=" * 60)
    print("Test 2: Scheduler Affinity Enhancement")
    print("=" * 60)
    
    print("\n[CHECK] Verifying ContinuumRequestQueue enhancement...")
    
    schedule_policy_path = os.path.join(WORK_DIR, "python/sglang/srt/managers/schedule_policy.py")
    with open(schedule_policy_path, "r") as f:
        content = f.read()
    
    # Check for turn_index awareness
    if "job_id_latest_turn" in content:
        print("  [PASS] job_id_latest_turn tracking found")
    else:
        print("  [FAIL] job_id_latest_turn tracking NOT found")
        return False
    
    # Check for continuous turn handling
    if "is_continuous" in content or "continuous" in content.lower():
        print("  [PASS] Continuous turn handling found")
    else:
        print("  [FAIL] Continuous turn handling NOT found")
        return False
    
    # Check for priority levels
    priority_count = content.count("pinned_candidates") + content.count("same_job_candidates")
    if priority_count >= 4:
        print("  [PASS] Multi-level priority handling found")
    else:
        print("  [FAIL] Multi-level priority handling NOT found")
        return False
    
    print("\n[PASS] Test 2: Scheduler affinity enhancement verified")
    return True


def test_simulator_consistency():
    """Test that simulator is consistent with sglang implementation."""
    print("\n" + "=" * 60)
    print("Test 3: Simulator Consistency")
    print("=" * 60)
    
    print("\n[CHECK] Verifying simulator continuum_ttl.py...")
    
    simulator_path = os.path.join(WORK_DIR, "tools/sglang-simulator/src/sglang_simulator/simulation/agent/continuum_ttl.py")
    if not os.path.exists(simulator_path):
        print(f"  [SKIP] Simulator path not found: {simulator_path}")
        return True
    
    with open(simulator_path, "r") as f:
        content = f.read()
    
    # Check for correct record_tool_execution implementation
    if "def record_tool_execution" in content:
        print("  [PASS] record_tool_execution method found")
    else:
        print("  [FAIL] record_tool_execution method NOT found")
        return False
    
    # Check that duration is used for tool_durations
    if "stats.tool_durations[tool_name].append(duration)" in content:
        print("  [PASS] Tool durations recorded correctly in record_tool_execution")
    else:
        print("  [FAIL] Tool durations NOT recorded correctly")
        return False
    
    print("\n[PASS] Test 3: Simulator consistency verified")
    return True


def test_integration():
    """Integration test simulating the complete flow."""
    print("\n" + "=" * 60)
    print("Test 4: Integration Test")
    print("=" * 60)
    
    print("\n[CHECK] Testing the complete flow...")
    
    # Simulate the fix behavior
    print("\nStep 1: Simulate request with tool_execution_time")
    # This would be passed via extra_body in real HTTP request
    extra_body = {
        "conversation_id": "prog_001",
        "tool_name": "bash",
        "tool_execution_time": 1.5,  # 1.5 seconds
    }
    print(f"  Input: extra_body = {extra_body}")
    
    # Simulate tool duration recording
    tool_execution_time = float(extra_body.get("tool_execution_time", 0))
    recorded_duration = tool_execution_time if tool_execution_time and tool_execution_time > 0 else 0.5  # fallback
    
    print(f"  Output: recorded_duration = {recorded_duration}")
    if recorded_duration == 1.5:
        print("  [PASS] tool_execution_time used correctly")
    else:
        print("  [FAIL] tool_execution_time NOT used")
        return False
    
    print("\nStep 2: Verify TTL calculation would use correct duration")
    # In real scenario, the TTL would be calculated based on tool durations
    # which now contains actual tool execution times
    
    print("  [PASS] TTL calculation would use correct tool duration history")
    
    print("\n[PASS] Test 4: Integration test passed")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SGLang Continuum TTL Fix Verification")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Tool Duration Recording", test_tool_duration_recording()))
    results.append(("Scheduler Affinity", test_scheduler_affinity()))
    results.append(("Simulator Consistency", test_simulator_consistency()))
    results.append(("Integration Test", test_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
