"""Sweep TTFT across multiple prompt files (varying input length) at fixed chunk count."""
import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

from test_streaming_input_latency import (
    extract_meta_info,
    flush_server_cache,
    launch_server,
    load_prompt_chunks,
    run_normal_test,
    run_streaming_input_test,
    summarize_meta_info,
    wait_for_server,
)
from sglang.srt.utils import kill_process_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure TTFT/post latencies for multiple prompt files while keeping chunk count and completion length fixed."
    )
    parser.add_argument(
        "prompt_files",
        nargs="+",
        help="List of JSON prompt files (each containing chunks or text).",
    )
    parser.add_argument("--chunk-count", type=int, default=10, help="Number of chunks from each prompt to use (truncate if longer).")
    parser.add_argument("--max-new-tokens", type=int, default=3, help="Completion tokens requested for all runs (auto-clipped to context).")
    parser.add_argument("--chunk-delay", type=float, default=0.5, help="Delay between streaming chunks.")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--no-launch", action="store_true", help="Assume server already running; skip auto-launch.")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-0.6B")
    return parser.parse_args()


def _fetch_model_limit(base_url: str) -> Optional[int]:
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=2)
        data = resp.json()
    except Exception:
        return None
    if isinstance(data, dict) and "data" in data:
        models = data.get("data") or []
        if models:
            return models[0].get("max_model_len")
    if isinstance(data, list) and data:
        return data[0].get("max_model_len")
    return None


def _tokenize_prompt(base_url: str, text: str) -> Optional[int]:
    try:
        resp = requests.post(f"{base_url}/tokenize", json={"prompt": text}, timeout=5)
        payload = resp.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    count = payload.get("count")
    if isinstance(count, list):
        count = count[0]
    return count


def _effective_new_tokens(max_model_len: Optional[int], prompt_tokens: Optional[int], desired_new: int) -> int:
    if max_model_len is None or prompt_tokens is None:
        return desired_new
    allowed = max(0, int(max_model_len) - int(prompt_tokens) - 1)
    return max(0, min(desired_new, allowed))


def _run_for_prompt(
    base_url: str,
    prompt_file: str,
    chunk_count: int,
    chunk_delay: float,
    max_new_tokens: int,
    max_model_len: Optional[int],
) -> Dict[str, Optional[float]]:
    chunks = load_prompt_chunks(prompt_file)
    if chunk_count > len(chunks):
        print(f"{prompt_file}: requested {chunk_count} chunks but only {len(chunks)} available; using all.")
        chunk_count = len(chunks)
    subset = list(chunks[:chunk_count])
    full_prompt = "".join(subset)
    prompt_tokens = _tokenize_prompt(base_url, full_prompt)
    effective_new = _effective_new_tokens(max_model_len, prompt_tokens, max_new_tokens)
    if effective_new < max_new_tokens:
        print(
            f"{prompt_file}: clipping max_new_tokens from {max_new_tokens} to {effective_new} (prompt tokens={prompt_tokens})."
        )
    streaming_id = f"prompt-{Path(prompt_file).stem}-{int(time.time()*1000)}"

    flush_server_cache(base_url)
    streaming_start = time.time()
    streaming_wall, streaming_post, streaming_resp = run_streaming_input_test(
        base_url,
        subset,
        streaming_id,
        chunk_delay,
        effective_new,
    )
    streaming_e2e = time.time() - streaming_start
    streaming_meta = extract_meta_info(streaming_resp)
    streaming_ttft = summarize_meta_info(streaming_meta)["timings"].get("ttft") if streaming_meta else None

    flush_server_cache(base_url)
    # 在普通模式下也引入与流式模式相同的分段等待，以模拟相同的网络分段延迟。
    normal_start = time.time()
    if chunk_count > 0 and chunk_delay > 0:
        for _ in range(chunk_count - 1):
            time.sleep(chunk_delay)
    normal_post, normal_resp = run_normal_test(base_url, full_prompt, effective_new)
    normal_e2e = time.time() - normal_start
    normal_meta = extract_meta_info(normal_resp)
    normal_ttft = summarize_meta_info(normal_meta)["timings"].get("ttft") if normal_meta else None

    return {
        "prompt": prompt_file,
        "chunk_count": chunk_count,
        "prompt_tokens": prompt_tokens,
        "streaming_wall": streaming_wall,
        "streaming_post": streaming_post,
        "streaming_ttft": streaming_ttft,
        "streaming_e2e": streaming_e2e,
        "normal_post": normal_post,
        "normal_ttft": normal_ttft,
        "normal_e2e": normal_e2e,
    }


def main():
    args = parse_args()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/models"))

    if args.chunk_count <= 0:
        raise ValueError("--chunk-count must be positive")

    base_url = f"http://localhost:{args.port}"
    max_model_len = _fetch_model_limit(base_url)

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
        results = []
        for prompt_file in args.prompt_files:
            print(f"\nRunning prompt={prompt_file} ...")
            res = _run_for_prompt(
                base_url,
                prompt_file,
                args.chunk_count,
                args.chunk_delay,
                args.max_new_tokens,
                max_model_len,
            )
            results.append(res)
            time.sleep(0.2)

        print("\nResults:")
        header = (
            "prompt | chunks | prompt_tokens | streaming_post(s) | streaming_ttft(s) | streaming_e2e(s) | "
            "normal_post(s) | normal_ttft(s) | normal_e2e(s)"
        )
        print(header)
        for r in results:
            tokens = r["prompt_tokens"] if r["prompt_tokens"] is not None else "?"
            short_name = Path(r["prompt"]).name
            print(
                f"{short_name:>20} | {r['chunk_count']:6} | {tokens!s:>13} | "
                f"{r['streaming_post']!s:>16} | {r['streaming_ttft']!s:>17} | {r['streaming_e2e']!s:>16} | "
                f"{r['normal_post']!s:>14} | {r['normal_ttft']!s:>14} | {r['normal_e2e']!s:>14}"
            )

    finally:
        if server_process:
            print("Stopping server...")
            kill_process_tree(server_process.pid)


if __name__ == "__main__":
    main()
