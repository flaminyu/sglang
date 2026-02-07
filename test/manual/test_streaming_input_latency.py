import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import requests

from sglang.srt.utils import kill_process_tree

DEFAULT_PROMPT_CHUNKS = [
    "From peak load studies to clean-firm build-outs, this default prompt exists only as a placeholder.\n",
    "Provide your own chunks via --prompt-file to stress token counts of any size.\n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latency of streaming-input vs normal prompts using SGLang native API."
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Path to JSON file describing prompt chunks. Accepts either a list of strings or an object with a 'chunks' list.",
    )
    parser.add_argument("--chunk-delay", type=float, default=2.0, help="Delay (seconds) between streaming chunks.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3,
        help="Completion tokens requested for the final streaming chunk and the normal request.",
    )
    parser.add_argument("--port", type=int, default=30000, help="Server port.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host binding.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model path used when auto-launching the server.",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Assume the server is already running; skip auto-launch.",
    )
    return parser.parse_args()


def load_prompt_chunks(path: Optional[str]) -> List[str]:
    """Load prompt chunks from JSON. Expected formats:

    - List[str]: treated as the chunk sequence.
    - {"chunks": [...] }: read the list under the 'chunks' key.
    - {"text": "..."}: single chunk derived from the provided text.
    """

    if path is None:
        print("Using built-in placeholder prompt (override with --prompt-file).")
        return DEFAULT_PROMPT_CHUNKS

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(chunk) for chunk in data]

    if isinstance(data, dict):
        chunks = data.get("chunks")
        if isinstance(chunks, list):
            return [str(chunk) for chunk in chunks]
        if isinstance(data.get("text"), str):
            return [data["text"]]

    raise ValueError("Prompt file must contain a list of strings or an object with 'chunks'.")


def launch_server(port: int, host: str, model_path: str):
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--port",
        str(port),
        "--host",
        host,
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process


def flush_server_cache(base_url: str, max_attempts: int = 3, backoff: float = 0.5) -> bool:
    """Best-effort clear of the KV/radix cache via /flush_cache."""
    url = f"{base_url}/flush_cache"
    last_error: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, timeout=5)
            if resp.status_code == 200:
                return True
            # Server responds with 400 when cache cannot be flushed (e.g., active reqs)
            print(
                f"WARNING: flush_cache attempt {attempt + 1} failed: status={resp.status_code}, body={resp.text.strip()}"
            )
        except requests.RequestException as exc:  # network error
            last_error = exc
            print(f"WARNING: flush_cache attempt {attempt + 1} raised: {exc}")
        time.sleep(backoff)
    if last_error:
        print(f"WARNING: flush_cache ultimately failed after {max_attempts} attempts: {last_error}")
    else:
        print(f"WARNING: flush_cache ultimately failed after {max_attempts} attempts.")
    return False

def wait_for_server(url):
    while True:
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)

def run_streaming_input_test(
    base_url: str,
    prompt_chunks: Sequence[str],
    streaming_input_id: str,
    inter_chunk_delay: float,
    max_new_tokens: int,
):
    url = f"{base_url}/generate"

    start_time = time.time()
    final_response = None
    final_post_latency = None

    for i, chunk in enumerate(prompt_chunks):
        if i > 0:
            time.sleep(inter_chunk_delay)

        is_last = i == len(prompt_chunks) - 1

        data = {
            "text": chunk,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens if is_last else 0,
            },
            "stream": False,
            "is_streaming_input": True,
            "is_last_chunk": is_last,
            "streaming_input_id": streaming_input_id,
            "streaming_payload_is_delta": True,
        }

        req_start = time.time()
        response = requests.post(url, json=data)
        req_end = time.time()
        result = response.json()

        if not is_last:
            print(f"Chunk {i+1}/{len(prompt_chunks)} ack: {json.dumps(result, ensure_ascii=False)}")
        else:
            final_response = result
            final_post_latency = req_end - req_start
            print(f"Final streaming response: {json.dumps(result, ensure_ascii=False)}")

    end_time = time.time()
    return end_time - start_time, final_post_latency, final_response


def run_normal_test(base_url: str, full_prompt: str, max_new_tokens: int):
    url = f"{base_url}/generate"

    data = {
        "text": full_prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
        "stream": False,
    }

    req_start = time.time()
    response = requests.post(url, json=data)
    req_end = time.time()
    result = response.json()
    print(f"Normal response: {json.dumps(result, ensure_ascii=False)}")
    return req_end - req_start, result


def _response_text(response_json):
    if not isinstance(response_json, dict):
        return None

    text = response_json.get("text")
    if isinstance(text, str):
        return text

    choices = response_json.get("choices") or []
    if choices:
        message = (choices[0].get("message") or {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content

    return None


def _prompt_tokens_from_response(response_json, meta_info):
    if isinstance(response_json, dict):
        usage = response_json.get("usage")
        if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
            return usage.get("prompt_tokens")

    if isinstance(meta_info, dict):
        return meta_info.get("prompt_tokens")

    return None


def compare_prompt_equivalence(streaming_resp, normal_resp, streaming_meta=None, normal_meta=None):
    if not streaming_resp or not normal_resp:
        print("Skipping equivalence check due to missing responses.")
        return

    streaming_tokens = _prompt_tokens_from_response(streaming_resp, streaming_meta)
    normal_tokens = _prompt_tokens_from_response(normal_resp, normal_meta)
    if streaming_tokens is not None and normal_tokens is not None:
        if streaming_tokens == normal_tokens:
            print(f"Prompt tokens match exactly: {streaming_tokens}")
        else:
            print(
                f"WARNING: prompt tokens differ (streaming={streaming_tokens}, normal={normal_tokens})"
            )
    else:
        print("Prompt token usage missing; skipping token comparison.")

    streaming_text = _response_text(streaming_resp)
    normal_text = _response_text(normal_resp)
    if streaming_text == normal_text:
        print("Final completions match exactly.")
    else:
        print("WARNING: final completions differ between streaming and normal modes.")


META_INFO_KEY_HINTS = {
    "e2e_latency",
    "prefill_finished_ts",
    "decode_finished_ts",
    "request_received_ts",
    "request_sent_to_scheduler_ts",
    "response_sent_to_client_ts",
    "queue_time",
    "prefill_launch_delay",
    "prefill_launch_latency",
    "decode_throughput",
}


def _maybe_meta_dict(candidate):
    if not isinstance(candidate, dict):
        return None
    nested = candidate.get("sglang_meta")
    if isinstance(nested, dict):
        return nested
    if any(key in candidate for key in META_INFO_KEY_HINTS):
        return candidate
    return None


def extract_meta_info(response_json):
    """Best-effort extraction of meta_info from different response layouts."""
    if not isinstance(response_json, dict):
        return None

    direct = response_json.get("meta_info")
    if isinstance(direct, dict):
        return direct

    metadata = _maybe_meta_dict(response_json.get("metadata"))
    if metadata:
        return metadata

    choices = response_json.get("choices") or []
    for choice in choices:
        choice_meta = _maybe_meta_dict(choice.get("metadata"))
        if choice_meta:
            return choice_meta

        message = choice.get("message") or {}
        msg_meta = _maybe_meta_dict(message.get("metadata") or message.get("meta_info"))
        if msg_meta:
            return msg_meta

    return None


def _as_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize_meta_info(meta_info):
    """Return timing and counter summaries derived from meta_info."""
    if not isinstance(meta_info, dict):
        return {"timings": {}, "counters": {}, "raw": None}

    timings = {}
    counters = {}

    def ts(name):
        return _as_float(meta_info.get(name))

    start = ts("request_received_ts")
    scheduler = ts("request_sent_to_scheduler_ts")
    prefill = ts("prefill_finished_ts")
    decode = ts("decode_finished_ts")
    response_sent = ts("response_sent_to_client_ts")

    e2e = _as_float(meta_info.get("e2e_latency"))
    if e2e is None and start is not None and decode is not None:
        e2e = max(0.0, decode - start)
    if e2e is not None:
        timings["e2e_latency"] = e2e

    if scheduler is not None and start is not None:
        timings["tokenization_time"] = max(0.0, scheduler - start)

    if prefill is not None and start is not None:
        timings["ttft"] = max(0.0, prefill - start)

    if prefill is not None and scheduler is not None:
        timings["prefill_duration"] = max(0.0, prefill - scheduler)
    elif prefill is not None and start is not None:
        timings["prefill_duration"] = max(0.0, prefill - start)

    if decode is not None and prefill is not None:
        timings["decode_duration"] = max(0.0, decode - prefill)

    if response_sent is not None and decode is not None:
        timings["postprocess_duration"] = max(0.0, response_sent - decode)

    for key in (
        "queue_time",
        "prefill_launch_delay",
        "prefill_launch_latency",
        "inference_time",
    ):
        value = _as_float(meta_info.get(key))
        if value is not None:
            timings[key] = value

    decode_throughput = _as_float(meta_info.get("decode_throughput"))
    if decode_throughput is not None:
        timings["decode_throughput"] = decode_throughput

    for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "total_retractions"):
        if key in meta_info and meta_info[key] is not None:
            counters[key] = meta_info[key]

    return {"timings": timings, "counters": counters, "raw": meta_info}


TIMING_PRINT_ORDER = [
    "e2e_latency",
    "queue_time",
    "prefill_launch_delay",
    "prefill_launch_latency",
    "tokenization_time",
    "ttft",
    "prefill_duration",
    "decode_duration",
    "postprocess_duration",
    "inference_time",
    "decode_throughput",
]


def print_latency_summary(label, summary):
    meta_info = summary.get("raw")
    if not meta_info:
        print(f"{label} meta_info is missing. Enable server metrics to populate timing data.")
        return

    print(f"{label} timing summary:")
    timings = summary.get("timings", {})
    if timings:
        for field in TIMING_PRINT_ORDER:
            if field not in timings:
                continue
            value = timings[field]
            if field == "decode_throughput":
                print(f"  - {field}: {value:.2f} tok/s")
            else:
                print(f"  - {field}: {value:.4f}s")
        other_fields = [
            key for key in timings.keys() if key not in TIMING_PRINT_ORDER
        ]
        for field in sorted(other_fields):
            value = timings[field]
            print(f"  - {field}: {value:.4f}s")
    else:
        print("  - No timing fields available.")

    counters = summary.get("counters", {})
    if counters:
        print("  Token stats: " + ", ".join(
            f"{name}={counters[name]}" for name in ("prompt_tokens", "completion_tokens", "cached_tokens", "total_retractions") if name in counters
        ))


def print_latency_comparison(streaming_summary, normal_summary):
    stream_timings = streaming_summary.get("timings", {})
    normal_timings = normal_summary.get("timings", {})

    comparable_fields = [
        "e2e_latency",
        "ttft",
        "prefill_duration",
        "decode_duration",
        "queue_time",
        "prefill_launch_delay",
        "prefill_launch_latency",
        "decode_throughput",
    ]
    overlaps = [f for f in comparable_fields if f in stream_timings and f in normal_timings]

    if not overlaps:
        print("No overlapping timing metrics to compare between modes.")
        return

    print("Timing delta (streaming - normal):")
    for field in overlaps:
        s_val = stream_timings[field]
        n_val = normal_timings[field]
        delta = s_val - n_val
        if n_val:
            percent = delta / n_val * 100.0
        else:
            percent = None
        unit = "tok/s" if field == "decode_throughput" else "s"
        percent_str = f" ({percent:+.2f}%)" if percent is not None else ""
        print(f"  - {field}: {delta:+.4f}{unit}{percent_str}")


def main():
    args = parse_args()

    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(key, None)
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/models"))

    base_url = f"http://localhost:{args.port}"
    prompt_chunks = load_prompt_chunks(args.prompt_file)
    full_prompt = "".join(prompt_chunks)
    streaming_id = f"stream-{int(time.time())}"

    # Query server for model max length (if available) and tokenize prompt to avoid requesting too many new tokens
    try:
        models_resp = requests.get(f"{base_url}/v1/models", timeout=2).json()
        if isinstance(models_resp, dict) and "data" in models_resp and len(models_resp["data"]) > 0:
            max_model_len = models_resp["data"][0].get("max_model_len")
        else:
            # /v1/models may return a Pydantic model list directly
            first = models_resp[0] if isinstance(models_resp, list) and models_resp else None
            max_model_len = first.get("max_model_len") if first else None
    except Exception:
        max_model_len = None

    # Tokenize full prompt to compute prompt token usage (use /tokenize if available)
    prompt_token_count = None
    try:
        tok_req = {"prompt": full_prompt}
        tok_resp = requests.post(f"{base_url}/tokenize", json=tok_req, timeout=5).json()
        # TokenizeResponse: {'tokens': [...], 'count': n, 'max_model_len': m}
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
            print(f"Server is already running at {base_url}")
        except requests.exceptions.ConnectionError:
            print("Starting server...")
            server_process = launch_server(args.port, args.host, args.model_path)

    try:
        wait_for_server(base_url)
        print("Server ready.")

        print("Flushing cache before streaming test...")
        requests.post(f"{base_url}/flush_cache")

        # Respect server-side max model length when requesting new tokens
        if prompt_token_count is not None and max_model_len is not None:
            allowed_new = max(0, int(max_model_len) - int(prompt_token_count) - 1)
            if args.max_new_tokens > allowed_new:
                print(f"Warning: requested --max-new-tokens={args.max_new_tokens} exceeds model capacity; capping to {allowed_new}.")
                args.max_new_tokens = allowed_new

        print("Running streaming input test...")
        streaming_wall, streaming_post, streaming_resp = run_streaming_input_test(
            base_url,
            prompt_chunks,
            streaming_id,
            args.chunk_delay,
            args.max_new_tokens,
        )
        print(f"Streaming wall-clock duration: {streaming_wall:.4f}s")
        if streaming_post is not None:
            print(f"Final chunk POST latency: {streaming_post:.4f}s")

        print("Flushing cache before normal test...")
        requests.post(f"{base_url}/flush_cache")

        print("Running normal test...")
        normal_post, normal_resp = run_normal_test(
            base_url, full_prompt, args.max_new_tokens
        )
        print(f"Normal POST latency: {normal_post:.4f}s")

        streaming_meta = extract_meta_info(streaming_resp)
        normal_meta = extract_meta_info(normal_resp)
        compare_prompt_equivalence(streaming_resp, normal_resp, streaming_meta, normal_meta)

        streaming_summary = summarize_meta_info(streaming_meta)
        normal_summary = summarize_meta_info(normal_meta)
        print_latency_summary("Streaming input", streaming_summary)
        print_latency_summary("Normal input", normal_summary)
        if streaming_summary.get("raw") and normal_summary.get("raw"):
            print_latency_comparison(streaming_summary, normal_summary)

        if streaming_post is not None:
            # Compare final POST latencies: streaming final-chunk POST vs normal full POST
            print(f"Streaming final POST latency: {streaming_post:.4f}s")
            print(f"Normal full POST latency: {normal_post:.4f}s")
            absolute_saving = normal_post - streaming_post
            print(f"Absolute saving (normal - streaming final POST): {absolute_saving:+.4f}s")
            if normal_post:
                pct = absolute_saving / normal_post * 100.0
                print(f"Percentage saving: {pct:+.2f}%")
        else:
            # Fallback: compare normal POST to streaming total wall-clock duration
            print("Streaming final POST latency unavailable; falling back to streaming wall-clock duration.")
            print(f"Streaming wall-clock duration: {streaming_wall:.4f}s")
            print(f"Normal full POST latency: {normal_post:.4f}s")
            absolute_saving = normal_post - streaming_wall
            print(f"Absolute saving (normal - streaming wall): {absolute_saving:+.4f}s")
            if normal_post:
                pct = absolute_saving / normal_post * 100.0
                print(f"Percentage saving: {pct:+.2f}%")
    finally:
        if server_process:
            print("Stopping server...")
            kill_process_tree(server_process.pid)

if __name__ == "__main__":
    main()