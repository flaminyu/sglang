"""Generate multiple prompt JSON files spanning different target token lengths."""
import argparse
import json
import os
from textwrap import fill
from typing import Iterable, List, Sequence

BASE_PARAGRAPH = (
    "This is a synthetic test paragraph intended to simulate realistic technical prose. "
    "It contains multiple sentences, diverse punctuation, and variable clause lengths to better emulate tokenizer behavior. "
    "Use these chunks to stress the request path and to measure latency characteristics under large prompts."
)

WORDS_PER_TOKEN = 0.75  # heuristic used across generators


def parse_int_list(raw: str, label: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(','):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            value = int(stripped)
        except ValueError as exc:  # pragma: no cover - user-controlled input
            raise ValueError(f"{label}: '{stripped}' is not an integer") from exc
        if value <= 0:
            raise ValueError(f"{label}: values must be positive (got {value})")
        values.append(value)
    if not values:
        raise ValueError(f"{label}: no valid integers provided")
    return values


def build_chunks(target_tokens: int, chunk_count: int) -> List[str]:
    total_words = int(target_tokens * WORDS_PER_TOKEN)
    words_per_chunk = max(50, total_words // max(1, chunk_count))
    chunks: List[str] = []
    for idx in range(chunk_count):
        paragraph = (BASE_PARAGRAPH + " ") * ((words_per_chunk // 40) + 1)
        header = f"Section {idx + 1}: Synthetic content for chunk {idx + 1}. "
        words = (header + paragraph).split()
        truncated = " ".join(words[:words_per_chunk])
        chunks.append(fill(truncated, width=100))
    return chunks


def approx_tokens_from_chunks(chunks: Sequence[str]) -> int:
    total_words = sum(len(chunk.split()) for chunk in chunks)
    return int(total_words / WORDS_PER_TOKEN)


def write_prompt(out_path: str, chunks: Sequence[str]) -> None:
    payload = {"chunks": list(chunks)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate prompt JSONs for multiple target token lengths.")
    parser.add_argument(
        "--targets",
        type=str,
        default="2000,5000,10000,20000",
        help="Comma-separated target token counts (approximate).",
    )
    parser.add_argument(
        "--chunk-counts",
        type=str,
        default=None,
        help="Optional comma-separated chunk counts matching --targets. If omitted, --chunks is used for all.",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=10,
        help="Chunk count applied uniformly when --chunk-counts is not provided.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="test/manual",
        help="Directory where prompt files are written.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="prompts",
        help="Filename prefix (final files look like prefix_<tokens>tok_<chunks>chunks.json).",
    )
    args = parser.parse_args()

    targets = parse_int_list(args.targets, "--targets")
    if args.chunk_counts:
        chunk_counts = parse_int_list(args.chunk_counts, "--chunk-counts")
        if len(chunk_counts) != len(targets):
            raise ValueError("--chunk-counts must have the same number of entries as --targets")
    else:
        if args.chunks <= 0:
            raise ValueError("--chunks must be a positive integer")
        chunk_counts = [args.chunks] * len(targets)

    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating prompt variants:")
    print("target_tokens | chunks | approx_tokens | path")

    for target, chunk_count in zip(targets, chunk_counts):
        chunks = build_chunks(target, chunk_count)
        approx_tokens = approx_tokens_from_chunks(chunks)
        filename = f"{args.prefix}_{target}tok_{chunk_count}chunks.json"
        out_path = os.path.join(args.out_dir, filename)
        write_prompt(out_path, chunks)
        print(f"{target:13} | {chunk_count:6} | {approx_tokens:13} | {out_path}")


if __name__ == "__main__":
    main()
