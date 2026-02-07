# Streaming Input Equivalence

SGLang now keeps the prompt tokens generated via streaming chunks bit-identical with
prompts that are submitted in a single request. This section describes the
contract that enables the new incremental prompt plumbing.

## Request contract

1. Every streaming chunk **must** set `is_streaming_input=true`, provide the
  same `streaming_input_id`, and specify whether it is the final chunk through
  `is_last_chunk`.
2. Chunks are **append-only**: the user payload contained in chunk *N* must be
  a strict extension of chunk *N − 1*. Re-draining or reordering chunks is
  rejected.
3. Native `/generate` requests can now send only the incremental delta per
  chunk by setting `streaming_payload_is_delta=true`. When this flag is set the
  HTTP front-end rebuilds the full prompt for the tokenizer manager, so clients
  no longer need to resend the entire prompt every time.
4. The OpenAI-compatible entrypoints still accept fully templated prompts, but
  internally we compute the delta tokens between two consecutive chunks and
  remove the previously cached template suffix before appending the new delta.

## Runtime behavior

- The tokenizer manager tracks the last tokenized prompt for every
  `streaming_input_id` and emits two additional metadata fields:
  - `streaming_trim_len`: how many tokens must be removed from the current
    request before appending the new delta tokens.
  - `streaming_total_input_len`: the total prompt length after the chunk is
    applied (used for validation/metrics).
- While a request is waiting for the next chunk we keep its KV pages pinned in
  the request-specific pool instead of inserting them into the Radix cache. The
  Radix cache is only updated after the final chunk, once the prompt is
  complete, so downstream decode stages see the exact same cache state as a
  single-shot request.
- If you are running with multiple tokenizer workers, make sure that all chunks
  for the same `streaming_input_id` are routed to the same worker (the default
  HTTP front-end does this when `tokenizer_worker_num=1`). Mixed-worker routing
  will fall back to full prompt replays and defeats the incremental trim logic.

## Validation tools

`test/manual/test_streaming_input_latency.py` now reports both latency and prompt
parity between streaming and non-streaming modes. After running the script you
should see:

```
Prompt tokens match exactly: XXXX
Final completions match exactly.
```

A mismatch indicates that chunks were not append-only, were routed to a different
worker, or the client reused a `streaming_input_id` across incompatible models.