"""Compare TTFT while varying input length (prompt size) for streaming vs normal requests."""
import argparse
import os
import time
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
        description="Compare TTFT as the prompt length scales (input-length sweep)."
    )
    parser.add_argument("--prompt-file", type=str, required=True, help="JSON file containing prompt chunks.")
    parser.add_argument(
        "--chunk-counts",
        type=str,
        default="5,10,20,40",
        help="Comma-separated chunk counts to test (each count uses the first N chunks).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Fixed completion length to request for all runs (will be clipped to context).",
    )
    parser.add_argument("--chunk-delay", type=float, default=0.5, help="Delay (s) between streaming chunks.")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--no-launch", action="store_true", help="Skip auto-launch; assume server already running.")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-0.6B")
    return parser.parse_args()


def _parse_counts(raw: str, max_chunks: int) -> List[int]:
    counts = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:  # pragma: no cover - user input error path
            raise ValueError(f"chunk count '{item}' is not an integer") from exc
        if value <= 0:
            raise ValueError("chunk counts must be positive")
        counts.append(min(value, max_chunks))
    if not counts:
        raise ValueError("No valid chunk counts provided")
    # keep order but drop duplicates by tracking seen combos
    seen = set()
    ordered: List[int] = []
    for count in counts:
        if count not in seen:
            ordered.append(count)
            seen.add(count)
    return ordered


def _fetch_model_limit(base_url: str) -> Optional[int]:
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=2)
        data = resp.json()
    except Exception:
        return None
    # handle openai-style list or raw list of dict
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


def run_for_chunk_count(
    base_url: str,
    prompt_chunks: Sequence[str],
    chunk_count: int,
    chunk_delay: float,
    max_model_len: Optional[int],
    desired_new_tokens: int,
) -> Dict[str, Optional[float]]:
    subset = list(prompt_chunks[:chunk_count])
    full_prompt = "".join(subset)
    prompt_tokens = _tokenize_prompt(base_url, full_prompt)
    effective_new = _effective_new_tokens(max_model_len, prompt_tokens, desired_new_tokens)
    if effective_new < desired_new_tokens:
        print(
            f"Chunk count {chunk_count}: clipping max_new_tokens from {desired_new_tokens} to {effective_new} (prompt tokens={prompt_tokens})."
        )
    streaming_id = f"input-len-{chunk_count}-{int(time.time()*1000)}"

    flush_server_cache(base_url)
    streaming_wall, streaming_post, streaming_resp = run_streaming_input_test(
        base_url,
        subset,
        streaming_id,
        chunk_delay,
        effective_new,
    )
    streaming_meta = extract_meta_info(streaming_resp)
    streaming_ttft = summarize_meta_info(streaming_meta)["timings"].get("ttft") if streaming_meta else None

    flush_server_cache(base_url)
    normal_post, normal_resp = run_normal_test(base_url, full_prompt, effective_new)
    normal_meta = extract_meta_info(normal_resp)
    normal_ttft = summarize_meta_info(normal_meta)["timings"].get("ttft") if normal_meta else None

    return {
        "chunk_count": chunk_count,
        "prompt_tokens": prompt_tokens,
        "streaming_wall": streaming_wall,
        "streaming_post": streaming_post,
        "streaming_ttft": streaming_ttft,
        "normal_post": normal_post,
        "normal_ttft": normal_ttft,
    }


def main():
    args = parse_args()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/models"))

    base_url = f"http://localhost:{args.port}"
    prompt_chunks = load_prompt_chunks(args.prompt_file)
    chunk_counts = _parse_counts(args.chunk_counts, len(prompt_chunks))

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
        for count in chunk_counts:
            print(f"Running chunk_count={count} ...")
            res = run_for_chunk_count(
                base_url,
                prompt_chunks,
                count,
                args.chunk_delay,
                max_model_len,
                args.max_new_tokens,
            )
            results.append(res)
            time.sleep(0.2)

        print("\nResults:")
        header = (
            "chunks | prompt_tokens | streaming_post(s) | streaming_ttft(s) | "
            "normal_post(s) | normal_ttft(s)"
        )
        print(header)
        for r in results:
            tokens = r["prompt_tokens"] if r["prompt_tokens"] is not None else "?"
            print(
                f"{r['chunk_count']:>6} | {tokens!s:>13} | "
                f"{r['streaming_post']!s:>16} | {r['streaming_ttft']!s:>17} | "
                f"{r['normal_post']!s:>14} | {r['normal_ttft']!s:>14}"
            )

    finally:
        if server_process:
            print("Stopping server...")
            kill_process_tree(server_process.pid)


if __name__ == "__main__":
    main()
