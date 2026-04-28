#!/usr/bin/env python3
"""
Pure Python KV Cache Simulator - Entry Point

This script provides a command-line interface for running simulations.

Usage:
    python run_simulation.py requests.jsonl
    python run_simulation.py requests.jsonl -c 100000 -t 5.0 -e lru
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simulator import KVCacheSimulator, run_simulation


def main():
    parser = argparse.ArgumentParser(
        description="Pure Python KV Cache Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python run_simulation.py requests.jsonl

    # With options
    python run_simulation.py requests.jsonl -c 50000 -t 10.0 -e continuum

    # With output
    python run_simulation.py requests.jsonl -o results.json

    # Verbose mode
    python run_simulation.py requests.jsonl -v
        """
    )
    
    parser.add_argument(
        "log_path",
        help="Path to request log file (JSON or JSONL format)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path for output JSON results (optional)"
    )
    
    parser.add_argument(
        "-c", "--capacity",
        type=int,
        default=100000,
        help="Cache capacity in tokens (default: 100000)"
    )
    
    parser.add_argument(
        "-t", "--ttl",
        type=float,
        default=5.0,
        help="Default TTL in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "-e", "--eviction",
        choices=["lru", "lfu", "fifo", "continuum"],
        default="lru",
        help="Eviction policy (default: lru)"
    )
    
    parser.add_argument(
        "--no-ttl",
        action="store_true",
        help="Disable TTL pinning"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress information"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.log_path).exists():
        print(f"Error: Request log file not found: {args.log_path}")
        sys.exit(1)
    
    # Run simulation
    # When --no-ttl is set, disable TTL completely via config
    enable_ttl = not args.no_ttl
    result = run_simulation(
        log_path=args.log_path,
        output_path=args.output,
        cache_capacity=args.capacity,
        default_ttl=args.ttl,
        eviction_policy=args.eviction,
        verbose=args.verbose,
        enable_ttl=enable_ttl,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Simulation Result Summary")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Cache Capacity: {args.capacity} tokens")
    print(f"  TTL: {'Disabled' if args.no_ttl else f'{args.ttl}s'}")
    print(f"  Eviction Policy: {args.eviction}")
    print()
    print("Cache Statistics:")
    print(f"  Total Requests: {result.cache_stats.total_requests}")
    print(f"  Cache Hits: {result.cache_stats.cache_hits}")
    print(f"  Cache Misses: {result.cache_stats.cache_misses}")
    print(f"  Hit Rate: {result.cache_stats.hit_rate:.1%}")
    print(f"  Tokens Cached: {result.cache_stats.tokens_cached}")
    print(f"  Tokens Evicted: {result.cache_stats.tokens_evicted}")
    print(f"  TTL Pins: {result.cache_stats.ttl_pins}")
    print()
    print(f"Duration: {result.duration:.2f}s")
    print("=" * 60)
    
    if args.output:
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
