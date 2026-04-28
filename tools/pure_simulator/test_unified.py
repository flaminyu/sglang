#!/usr/bin/env python3
"""
Unified TTL Speedup Test Script

Configurable parameters:
- tokens_per_turn: prefill tokens per turn
- output_len: decode output length
- cache_capacity: KV cache capacity
- num_programs: number of concurrent programs
- num_turns: turns per program
- tool_time: tool execution time (creates interleaving)
- ttl_enabled: enable/disable TTL
- arrival_pattern: 'burst' (all at once) or 'staggered' (distributed)

Usage:
    python test_unified.py --prefill 1000 --decode 50 --cache 5000 --programs 10 --turns 5
"""

import sys
sys.path.insert(0, '.')

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig
import json
import tempfile
import random
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='TTL Speedup Test')
    
    # Timing parameters
    parser.add_argument('--prefill', type=int, default=400, help='Tokens per turn (prefill)')
    parser.add_argument('--decode', type=int, default=64, help='Decode output length')
    parser.add_argument('--prefill_ms', type=float, default=2.0, help='Prefill latency per token (ms)')
    parser.add_argument('--decode_ms', type=float, default=20.0, help='Decode latency per token (ms)')
    
    # Cache parameters
    parser.add_argument('--cache', type=int, default=3000, help='L1 cache capacity (tokens)')
    parser.add_argument('--l2_cache', type=int, default=0, help='L2 cache capacity (0=disabled)')
    parser.add_argument('--l2_reload_penalty', type=float, default=0.3, 
                       help='L2 reload penalty as fraction of prefill time')
    
    # Workload parameters
    parser.add_argument('--programs', type=int, default=10, help='Number of programs')
    parser.add_argument('--turns', type=int, default=5, help='Turns per program')
    parser.add_argument('--tool_time', type=float, default=0.5, help='Tool execution time (s)')
    
    # TTL parameters
    parser.add_argument('--ttl', type=float, default=5.0, help='TTL duration (s), 0 to disable')
    parser.add_argument('--adaptive_ttl', action='store_true', help='Enable adaptive TTL')
    
    # Arrival pattern
    parser.add_argument('--arrival', choices=['burst', 'staggered', 'poisson'], default='burst',
                       help='Arrival pattern: burst (all at once), staggered, or poisson')
    parser.add_argument('--jps', type=float, default=0.5,
                       help='Jobs per second arrival rate for poisson distribution')
    
    # Output options
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--compare', action='store_true', help='Compare TTL vs No TTL')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Preset scenarios
    parser.add_argument('--scenario', choices=['paper_swe', 'paper_bfcl', 'paper_small', 'stress'],
                      help='Use paper approximation scenario')
    
    return parser.parse_args()


def get_paper_scenario(scenario):
    """Get parameters approximating paper scenarios."""
    # Based on Table 1 in Continuum paper:
    # SWE-Bench: mean turns=10.9, tool_time=925ms, tokens/program=70k
    # BFCL v4: mean turns=6.3, tool_time=1923ms, tokens/program=93k
    
    scenarios = {
        'paper_swe': {
            'prefill': 6000,  # ~70k tokens / 10.9 turns / ~1.1x growth
            'decode': 64,
            'programs': 50,
            'turns': 10,
            'tool_time': 0.925,
            'cache': 100_000,  # ~100k tokens L1
            'l2_cache': 1_000_000,  # ~1M tokens L2 (200GB)
            'ttl': 5.0,
        },
        'paper_bfcl': {
            'prefill': 14000,  # ~93k tokens / 6.3 turns / ~1.5x growth
            'decode': 64,
            'programs': 30,
            'turns': 6,
            'tool_time': 1.923,
            'cache': 150_000,
            'l2_cache': 1_500_000,
            'ttl': 5.0,
        },
        'paper_small': {
            'prefill': 2000,
            'decode': 64,
            'programs': 20,
            'turns': 8,
            'tool_time': 0.5,
            'cache': 50_000,
            'l2_cache': 500_000,
            'ttl': 3.0,
        },
        'stress': {
            'prefill': 3000,
            'decode': 64,
            'programs': 100,
            'turns': 15,
            'tool_time': 0.5,
            'cache': 30_000,
            'l2_cache': 300_000,
            'ttl': 4.0,
        },
    }
    return scenarios.get(scenario, {})


def generate_requests(args):
    """Generate requests based on configuration."""
    rng = random.Random(args.seed)
    requests = []
    
    if args.arrival == 'poisson':
        # Poisson arrival: each program's first turn arrives with exponential inter-arrival time
        # Subsequent turns follow tool execution time but are interleaved
        rng = random.Random(args.seed)
        requests = []
        current_time = 0.0
        program_turns = {prog_idx: 0 for prog_idx in range(args.programs)}
        
        # Each program has turns to complete
        total_turns = args.programs * args.turns
        completed = 0
        
        while completed < total_turns:
            # Choose a program that hasn't completed all turns
            available = [p for p in range(args.programs) if program_turns[p] < args.turns]
            if not available:
                break
            
            prog_idx = rng.choice(available)
            turn = program_turns[prog_idx]
            
            if turn == 0:
                # First turn: poisson arrival
                inter_arrival = rng.expovariate(args.jps)
                current_time += inter_arrival
            else:
                # Subsequent turns: based on tool execution
                current_time += rng.expovariate(1.0 / args.tool_time)
            
            program_id = f"prog_{prog_idx:03d}"
            accumulated = list(range(
                prog_idx * 100000,
                prog_idx * 100000 + turn * args.prefill
            ))
            new_tokens = list(range(
                prog_idx * 100000 + turn * args.prefill,
                prog_idx * 100000 + (turn + 1) * args.prefill
            ))
            tokens = accumulated + new_tokens
            
            is_tool = turn < args.turns - 1
            
            requests.append({
                "rid": f"{program_id}_turn_{turn}",
                "program_id": program_id,
                "turn_index": turn,
                "token_ids": tokens,
                "input_len": len(tokens),
                "output_len": args.decode,
                "arrival_time": current_time,
                "is_tool_call": is_tool,
                "tool_name": f"tool_{turn % 3}",
                "extra_key": program_id,
                "ttl_sec": args.ttl if args.ttl > 0 else None,
                "actual_tool_duration": rng.expovariate(1.0 / args.tool_time) if is_tool else 0.0,
            })
            
            program_turns[prog_idx] += 1
            completed += 1
    else:
        # Burst or staggered arrival
        for prog_idx in range(args.programs):
            program_id = f"prog_{prog_idx:03d}"
            accumulated = []
            
            for turn in range(args.turns):
                new_tokens = list(range(
                    prog_idx * 100000 + turn * args.prefill,
                    prog_idx * 100000 + turn * args.prefill + args.prefill
                ))
                tokens = accumulated + new_tokens
                
                is_tool = turn < args.turns - 1
                
                # Arrival time based on pattern
                if args.arrival == 'burst':
                    arrival_time = 0.0
                else:  # staggered
                    arrival_time = prog_idx * 0.5 + turn * args.tool_time * 0.5
                
                requests.append({
                    "rid": f"{program_id}_turn_{turn}",
                    "program_id": program_id,
                    "turn_index": turn,
                    "token_ids": tokens,
                    "input_len": len(tokens),
                    "output_len": args.decode,
                    "arrival_time": arrival_time,
                    "is_tool_call": is_tool,
                    "tool_name": f"tool_{turn % 3}",
                    "extra_key": program_id,
                    "ttl_sec": args.ttl if args.ttl > 0 else None,
                    "actual_tool_duration": rng.expovariate(1.0 / args.tool_time) if is_tool else 0.0,
                })
                accumulated.extend(new_tokens)
    
    # Sort by arrival time
    requests.sort(key=lambda r: r['arrival_time'])
    
    return requests


def run_simulation(requests, args):
    """Run simulation and return results."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    
    try:
        hw = HardwareConfig(
            prefill_latency_per_token_ms=args.prefill_ms,
            decode_latency_per_token_ms=args.decode_ms,
            cache_hit_speedup=10.0,
            ttft_overhead_ms=100.0,
            queue_delay_per_request_ms=100.0,
        )
        
        l2_enabled = getattr(args, 'l2_cache', 0) > 0
        
        sim = KVCacheSimulator(
            cache_capacity=args.cache,
            l2_capacity=getattr(args, 'l2_cache', 0),
            l2_enabled=l2_enabled,
            default_ttl=args.ttl if args.ttl > 0 else 5.0,
            enable_ttl=args.ttl > 0,
            enable_adaptive_ttl=args.adaptive_ttl,
            simulation_mode='event_driven',
            tool_execution_time=args.tool_time,
            random_seed=args.seed,
            hardware_config=hw,
            l2_reload_penalty=getattr(args, 'l2_reload_penalty', 0.3),
        )
        
        sim.load_requests(temp_path)
        for pid in sim.scheduler.program_total_turns:
            sim.scheduler.program_total_turns[pid] = args.turns
        
        result = sim.run(verbose=args.verbose)
        
        return {
            'result': result,
            'scheduling_log': sim.scheduler.scheduling_log,
            'program_jcts': sim.collector.compute_program_jct(sim.scheduler.scheduling_log),
            'program_jct_stats': sim.collector.compute_program_jct_stats(sim.scheduler.scheduling_log),
            'program_first_arrival': dict(sim.scheduler.program_first_arrival),
            'program_last_finish': dict(sim.scheduler.program_last_finish),
        }
    finally:
        os.unlink(temp_path)


def print_summary(name, data, args):
    """Print summary of simulation results."""
    jct_stats = data['program_jct_stats']
    result = data['result']
    log = data['scheduling_log']
    
    # Time breakdown
    total_prefill = sum(e['prefill_ms'] for e in log)
    total_decode = sum(e['decode_ms'] for e in log)
    total_queue = sum(e['queue_wait_ms'] for e in log)
    tool_exec = len(log) * args.tool_time * 1000
    
    # Hit type counts
    ttl_hits = sum(1 for e in log if e['hit_type'] == 'TTL')
    l1_hits = sum(1 for e in log if e['hit_type'] == 'L1')
    l2_hits = sum(1 for e in log if e['hit_type'] == 'L2')
    misses = sum(1 for e in log if e['hit_type'] == 'L1+L2 Miss')
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    print(f"\nConfiguration:")
    print(f"  Prefill: {args.prefill} tokens/turn @ {args.prefill_ms}ms/token")
    print(f"  Decode: {args.decode} tokens @ {args.decode_ms}ms/token")
    print(f"  L1 Cache: {args.cache:,} tokens ({args.cache * 512 / 1e9:.2f} GB)")
    if getattr(args, 'l2_cache', 0) > 0:
        print(f"  L2 Cache: {args.l2_cache:,} tokens ({args.l2_cache * 512 / 1e9:.2f} GB)")
    print(f"  Workload: {args.programs} programs × {args.turns} turns")
    print(f"  Tool time: {args.tool_time}s")
    print(f"  Arrival: {args.arrival}")
    
    print(f"\nJCT Results:")
    print(f"  Avg JCT: {jct_stats.get('avg_jct', 0):.1f}ms")
    print(f"  P95 JCT: {jct_stats.get('p95_jct', 0):.1f}ms")
    print(f"  Programs: {jct_stats.get('num_programs', 0)}")
    
    print(f"\nCache Performance:")
    print(f"  TTL Hits: {ttl_hits}")
    print(f"  L1 Hits: {l1_hits}")
    print(f"  L2 Hits: {l2_hits}")
    print(f"  Misses: {misses}")
    print(f"  Evicted: {result.cache_stats.tokens_evicted:,}")
    
    print(f"\nTime Breakdown:")
    total = total_prefill + total_decode + tool_exec
    print(f"  Prefill: {total_prefill/1000:.2f}s ({total_prefill/total*100:.1f}%)")
    print(f"  Decode: {total_decode/1000:.2f}s ({total_decode/total*100:.1f}%)")
    print(f"  Tool Exec: {tool_exec/1000:.2f}s ({tool_exec/total*100:.1f}%)")
    
    return jct_stats


def compare_ttl_no_ttl(args):
    """Compare TTL vs No TTL."""
    print("\n" + "="*70)
    print("TTL vs No TTL Comparison")
    print("="*70)
    
    # Generate requests once
    requests = generate_requests(args)
    
    # Run with TTL
    data_ttl = run_simulation(requests, args)
    args_ttl = args
    
    # Run without TTL
    args_no_ttl = argparse.Namespace(
        prefill=args.prefill,
        decode=args.decode,
        prefill_ms=args.prefill_ms,
        decode_ms=args.decode_ms,
        cache=args.cache,
        l2_cache=getattr(args, 'l2_cache', 0),
        l2_reload_penalty=getattr(args, 'l2_reload_penalty', 0.3),
        programs=args.programs,
        turns=args.turns,
        tool_time=args.tool_time,
        ttl=0,  # Disable TTL
        adaptive_ttl=False,
        arrival=args.arrival,
        verbose=args.verbose,
        compare=False,
        seed=args.seed,
    )
    data_no_ttl = run_simulation(requests, args_no_ttl)
    
    # Print summaries
    jct_ttl = print_summary("With TTL", data_ttl, args_ttl)
    jct_no_ttl = print_summary("Without TTL", data_no_ttl, args_no_ttl)
    
    # Compare
    avg_speedup = jct_no_ttl.get('avg_jct', 0) / jct_ttl.get('avg_jct', 1)
    p95_speedup = jct_no_ttl.get('p95_jct', 0) / jct_ttl.get('p95_jct', 1)
    
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"\n{'Metric':>25} | {'No TTL':>15} | {'With TTL':>15} | {'Speedup':>10}")
    print("-" * 70)
    print(f"{'Avg JCT (ms)':>25} | {jct_no_ttl.get('avg_jct', 0):>15.1f} | {jct_ttl.get('avg_jct', 0):>15.1f} | {avg_speedup:>9.2f}x")
    print(f"{'P95 JCT (ms)':>25} | {jct_no_ttl.get('p95_jct', 0):>15.1f} | {jct_ttl.get('p95_jct', 0):>15.1f} | {p95_speedup:>9.2f}x")
    
    ttl_log = data_ttl['scheduling_log']
    no_ttl_log = data_no_ttl['scheduling_log']
    
    ttl_pinned = [e['queue_wait_ms'] for e in ttl_log if e.get('is_pinned')]
    ttl_unpinned = [e['queue_wait_ms'] for e in ttl_log if not e.get('is_pinned')]
    no_ttl_waits = [e['queue_wait_ms'] for e in no_ttl_log]
    
    import statistics
    print(f"\nQueue Wait:")
    print(f"  No TTL avg: {statistics.mean(no_ttl_waits):.0f}ms")
    if ttl_pinned:
        print(f"  TTL Pinned avg: {statistics.mean(ttl_pinned):.0f}ms")
    if ttl_unpinned:
        print(f"  TTL Unpinned avg: {statistics.mean(ttl_unpinned):.0f}ms")
    
    # Cache stats comparison
    ttl_result = data_ttl['result'].cache_stats
    no_ttl_result = data_no_ttl['result'].cache_stats
    
    print(f"\nCache Hits:")
    print(f"  TTL Hits: No TTL={no_ttl_result.ttl_hits}, TTL={ttl_result.ttl_hits}")
    print(f"  L1 Hits: No TTL={no_ttl_result.l1_hits}, TTL={ttl_result.l1_hits}")


def main():
    args = parse_args()
    
    # Apply scenario preset if specified
    if args.scenario:
        scenario_params = get_paper_scenario(args.scenario)
        for key, value in scenario_params.items():
            setattr(args, key, value)
        print(f"\nUsing scenario: {args.scenario}")
        print(f"Parameters: {scenario_params}")
    
    if args.compare:
        compare_ttl_no_ttl(args)
    else:
        requests = generate_requests(args)
        data = run_simulation(requests, args)
        
        name = f"TTL {'ON (' + str(args.ttl) + 's)' if args.ttl > 0 else 'OFF'}"
        print_summary(name, data, args)


if __name__ == "__main__":
    main()
