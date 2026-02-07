"""Compare TTFT across different output lengths for streaming and normal requests.

Usage example:
  python3 test/manual/compare_ttft_by_output_length.py --prompt-file test/manual/prompts_10k.json --lengths 1,3,10,50 --chunk-delay 0.5 --no-launch

The script sends requests to /generate and extracts TTFT from returned meta_info (or falls back to timing derived from timestamps).
"""
import argparse
import json
import os
import sys
import time
from typing import List, Optional

import requests

from test_streaming_input_latency import load_prompt_chunks, run_streaming_input_test, run_normal_test, extract_meta_info, summarize_meta_info, wait_for_server, launch_server
from sglang.srt.utils import kill_process_tree


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-file", type=str, required=True)
    p.add_argument("--lengths", type=str, default="1,3,10", help="Comma-separated list of max_new_tokens values")
    p.add_argument("--chunk-delay", type=float, default=0.5)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--no-launch", action="store_true")
    p.add_argument("--model-path", type=str, default="Qwen/Qwen3-0.6B")
    return p.parse_args()


def get_ttft_from_meta(meta):
    if not meta:
        return None
    # meta may contain ttft directly or prefill_finished_ts/request_received_ts to compute
    ttft = meta.get("ttft")
    if ttft is not None:
        return float(ttft)
    # try derive
    start = meta.get("request_received_ts")
    prefill = meta.get("prefill_finished_ts")
    if start is not None and prefill is not None:
        return float(prefill) - float(start)
    return None


def run_for_length(base_url: str, prompt_chunks: List[str], length: int, chunk_delay: float):
    full_prompt = "".join(prompt_chunks)
    streaming_id = f"compare-{length}-{int(time.time())}"

    # streaming
    streaming_wall, streaming_post, streaming_resp = run_streaming_input_test(
        base_url, prompt_chunks, streaming_id, chunk_delay, length
    )
    streaming_meta = extract_meta_info(streaming_resp)
    streaming_summary = summarize_meta_info(streaming_meta)
    streaming_ttft = streaming_summary.get("timings", {}).get("ttft")

    # normal
    normal_post, normal_resp = run_normal_test(base_url, full_prompt, length)
    normal_meta = extract_meta_info(normal_resp)
    normal_summary = summarize_meta_info(normal_meta)
    normal_ttft = normal_summary.get("timings", {}).get("ttft")

    return {
        "length": length,
        "streaming_wall": streaming_wall,
        "streaming_post": streaming_post,
        "streaming_ttft": streaming_ttft,
        "normal_post": normal_post,
        "normal_ttft": normal_ttft,
    }


def main():
    args = parse_args()
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(key, None)
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/models"))

    base_url = f"http://localhost:{args.port}"
    prompt_chunks = load_prompt_chunks(args.prompt_file)
    full_prompt = "".join(prompt_chunks)

    # query model limits and tokenize prompt
    max_model_len = None
    try:
        models_resp = requests.get(f"{base_url}/v1/models", timeout=2).json()
        if isinstance(models_resp, dict) and "data" in models_resp and len(models_resp["data"]) > 0:
            max_model_len = models_resp["data"][0].get("max_model_len")
        else:
            first = models_resp[0] if isinstance(models_resp, list) and models_resp else None
            max_model_len = first.get("max_model_len") if first else None
    except Exception:
        max_model_len = None

    prompt_token_count = None
    try:
        tok_req = {"prompt": full_prompt}
        tok_resp = requests.post(f"{base_url}/tokenize", json=tok_req, timeout=5).json()
        if isinstance(tok_resp, dict):
            prompt_token_count = tok_resp.get("count")
            if isinstance(prompt_token_count, list):
                prompt_token_count = prompt_token_count[0]
            if max_model_len is None:
                max_model_len = tok_resp.get("max_model_len")
    except Exception:
        prompt_token_count = None

    server_process = None
    if not args.no_launch:
        try:
            requests.get(f"{base_url}/health")
            print(f"Server already running at {base_url}")
        except requests.exceptions.ConnectionError:
            print("Starting server...")
            server_process = launch_server(args.port, args.host, args.model_path)

    try:
        wait_for_server(base_url)
        lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
        # Clip lengths according to model capacity if available
        if prompt_token_count is not None and max_model_len is not None:
            allowed_new = max(0, int(max_model_len) - int(prompt_token_count) - 1)
            lengths = [min(l, allowed_new) for l in lengths]
            print(f"Adjusted lengths according to model capacity: {lengths} (allowed_new={allowed_new})")
        results = []
        for L in lengths:
            print(f"Running length={L} ...")
            res = run_for_length(base_url, prompt_chunks, L, args.chunk_delay)
            results.append(res)
            # small cooldown
            time.sleep(0.2)

        # print table
        print("\nResults:")
        print("len | streaming_post(s) | streaming_ttft(s) | normal_post(s) | normal_ttft(s)")
        for r in results:
            print(f"{r['length']:>3} | {r['streaming_post']!s:>16} | {r['streaming_ttft']!s:>17} | {r['normal_post']!s:>14} | {r['normal_ttft']!s:>14}")

    finally:
        if server_process:
            print("Stopping server...")
            kill_process_tree(server_process.pid)


if __name__ == '__main__':
    main()
