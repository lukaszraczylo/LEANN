#!/usr/bin/env python3
"""Measure generation latency of a HuggingFace/OpenAI-compatible model over prompt files."""

import argparse
import contextlib
import io
import json
import logging
import time
from pathlib import Path

from leann.chat import get_llm

PROMPT_PREFIX = "PROMPT #"
logging.getLogger("leann.chat").setLevel(logging.ERROR)


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    buffer: list[str] = []
    collecting = False

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(PROMPT_PREFIX):
                if buffer:
                    prompts.append("".join(buffer).strip())
                    buffer.clear()
                collecting = True
                continue

            if collecting:
                buffer.append(line)

    if buffer:
        prompts.append("".join(buffer).strip())

    return prompts


def measure_generation_times(
    prompts: list[str],
    llm,
    generation_kwargs: dict[str, object],
    allow_truncation: bool,
    enable_qwen_thinking: bool,
):
    timings: list[float] = []
    tokenizer = getattr(llm, "tokenizer", None)
    max_positions = None
    if hasattr(llm, "model") and hasattr(llm.model, "config"):
        max_positions = getattr(llm.model.config, "max_position_embeddings", None)

    requested_new_tokens = None
    if max_positions is not None:
        if "max_new_tokens" in generation_kwargs:
            requested_new_tokens = generation_kwargs["max_new_tokens"]
        elif "max_tokens" in generation_kwargs:
            requested_new_tokens = generation_kwargs["max_tokens"]

    context_max_length = max_positions
    if max_positions is not None and requested_new_tokens is not None:
        if requested_new_tokens >= max_positions:
            requested_new_tokens = max_positions - 1
        context_max_length = max(max_positions - requested_new_tokens, 1)

    suppress_buffer = io.StringIO()
    for prompt in prompts:
        prompt_for_llm = prompt
        if (
            enable_qwen_thinking
            and "/think" not in prompt_for_llm
            and "/no_think" not in prompt_for_llm
        ):
            prompt_for_llm = f"{prompt_for_llm}\n/think"

        if allow_truncation and tokenizer is not None and max_positions is not None:
            tokenized = tokenizer(
                prompt_for_llm,
                truncation=True,
                max_length=context_max_length,
                return_tensors="pt",
            )
            prompt_for_llm = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)

        per_call_kwargs = dict(generation_kwargs)
        if requested_new_tokens is not None:
            per_call_kwargs["max_new_tokens"] = requested_new_tokens

        start = time.perf_counter()
        with contextlib.redirect_stdout(suppress_buffer):
            llm.ask(prompt_for_llm, **per_call_kwargs)
        end = time.perf_counter()
        timings.append(end - start)
        suppress_buffer.seek(0)
        suppress_buffer.truncate(0)

    return timings


def parse_args():
    parser = argparse.ArgumentParser(description="Measure generation timing for prompt files")
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional limit on number of prompts to evaluate per file",
    )
    parser.add_argument(
        "--allow-truncation",
        action="store_true",
        help="Allow truncating prompt context to respect model's max context",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sshleifer/tiny-gpt2",
        help="LLM model identifier (default: sshleifer/tiny-gpt2)",
    )
    parser.add_argument(
        "--llm-type",
        type=str,
        default="hf",
        choices=["hf", "openai", "ollama", "gemini", "simulated"],
        help="LLM backend type (default: hf)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "auto"],
        help="Device override for HF models (default: cpu)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Max new tokens per generation (default: 16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling top-p (default: 0.8)",
    )
    parser.add_argument(
        "--qwen-thinking",
        action="store_true",
        help="Append /think to prompts for Qwen models",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_files = [
        Path("prompt_all_nq_bm25.txt"),
        Path("prompt_all_nq_diskann_full.txt"),
        Path("prompt_all_nq_diskann_pq5.txt"),
        Path("prompt_all_nq_hnsw.txt"),
    ]

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    results: dict[str, dict[str, float | int]] = {}

    llm_config = {"type": args.llm_type, "model": args.model}
    try:
        llm = get_llm(llm_config)
    except Exception as exc:
        print(f"Failed to initialize LLM: {exc}")
        raise SystemExit(1) from exc

    if args.llm_type == "hf" and hasattr(llm, "model") and args.device == "cpu":
        llm.model = llm.model.to("cpu")
        if hasattr(llm, "device"):
            llm.device = "cpu"

    for dataset_path in dataset_files:
        print(f"Processing {dataset_path.name}...")
        prompts = load_prompts(dataset_path)
        if args.max_prompts is not None:
            prompts = prompts[: args.max_prompts]
        timings = measure_generation_times(
            prompts,
            llm,
            generation_kwargs,
            args.allow_truncation,
            args.qwen_thinking,
        )
        total_time = sum(timings)
        count = len(timings)
        average_time = total_time / count if count else 0.0
        results[str(dataset_path.name)] = {
            "total_prompts": count,
            "total_time_seconds": total_time,
            "average_time_seconds": average_time,
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
