"""Generate a JSON file with many prompt chunks approximating a target token count.

Usage example:
  python3 test/manual/generate_prompts_10k.py --out test/manual/prompts_10k.json --target-tokens 10000 --chunks 40

The script uses a rough tokens-per-word estimate, so resulting token count is approximate.
"""
import argparse
import json
import math
import random
from textwrap import fill

BASE_PARAGRAPH = (
    "This is a synthetic test paragraph intended to simulate realistic technical prose. "
    "It contains multiple sentences, diverse punctuation, and variable clause lengths to better emulate tokenizer behavior. "
    "Use these chunks to stress the request path and to measure latency characteristics under large prompts."
)


def build_chunks(target_tokens: int, chunks: int) -> list:
    # Rough heuristic: assume ~0.75 words per token (i.e., 1 token ~= 0.75 words) => words_needed = tokens * 0.75
    words_per_token = 0.75
    total_words = int(target_tokens * words_per_token)
    words_per_chunk = max(50, total_words // max(1, chunks))

    out = []
    for i in range(chunks):
        # Create a chunk by repeating base paragraph and adding a small variation.
        paragraph = (BASE_PARAGRAPH + " ") * ((words_per_chunk // 40) + 1)
        # insert a short unique header and shuffle some phrases for variety
        header = f"Section {i+1}: Summary of synthetic content for chunk {i+1}. "
        body = header + paragraph
        # truncate to approximately words_per_chunk words
        words = body.split()
        truncated = " ".join(words[:words_per_chunk])
        # wrap lines a bit for readability
        out.append(fill(truncated, width=100))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="test/manual/prompts_10k.json")
    parser.add_argument("--target-tokens", type=int, default=10000)
    parser.add_argument("--chunks", type=int, default=40)
    args = parser.parse_args()

    chunks = build_chunks(args.target_tokens, args.chunks)
    payload = {"chunks": chunks}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Print summary
    total_words = sum(len(c.split()) for c in chunks)
    approx_tokens = int(total_words / words_per_token) if (words_per_token := 0.75) else None
    print(f"Wrote {len(chunks)} chunks to {args.out}")
    print(f"Approx words: {total_words}; approximate tokens: {approx_tokens}")


if __name__ == "__main__":
    main()
