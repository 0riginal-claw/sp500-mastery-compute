#!/usr/bin/env python3
"""LLMLingua prompt compressor CLI wrapper.

Canonical invocation:
  /Users/orginal/.venvs/sp500-mastery/bin/python scripts/llmlingua_compress.py [options]

Reads from --text or stdin. Writes compressed text to stdout.
Info/timing/errors go to stderr so the output stays pipeable.
"""

import argparse
import sys
import time

LLMLINGUA2_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
LLMLINGUA2_SMALL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
LLMLINGUA1_MODEL = "gpt2-medium"


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress a prompt with LLMLingua and print the result to stdout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  echo "your long prompt" | \\
      /Users/orginal/.venvs/sp500-mastery/bin/python scripts/llmlingua_compress.py --target-ratio 0.5

  /Users/orginal/.venvs/sp500-mastery/bin/python scripts/llmlingua_compress.py \\
      --text "your long prompt" --target-ratio 0.4 --info

  cat big_context.txt | \\
      /Users/orginal/.venvs/sp500-mastery/bin/python scripts/llmlingua_compress.py \\
      --question "What is the main conclusion?" --info
""",
    )
    parser.add_argument("--text", type=str, default=None,
                        help="Input text to compress (reads stdin if omitted)")
    parser.add_argument("--target-ratio", type=float, default=0.5,
                        help="Target compression ratio — fraction of original to keep (default: 0.5)")
    parser.add_argument("--model", type=str, default=None,
                        help=f"HuggingFace model ID (default: {LLMLINGUA2_SMALL})")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (default: cpu)")
    parser.add_argument("--instruction", type=str, default="",
                        help="System/instruction text to preserve during compression")
    parser.add_argument("--question", type=str, default="",
                        help="Question/task text to preserve during compression")
    parser.add_argument("--use-v1", action="store_true",
                        help="Force LLMLingua v1 (default is v2)")
    parser.add_argument("--info", action="store_true",
                        help="Print token counts and timing to stderr")
    args = parser.parse_args()

    # Read input
    if args.text is not None:
        raw_text = args.text
    else:
        if sys.stdin.isatty():
            print("llmlingua_compress: reading from stdin (pipe text or use --text)", file=sys.stderr)
        raw_text = sys.stdin.read()

    if not raw_text.strip():
        print("llmlingua_compress: empty input — nothing to compress", file=sys.stderr)
        sys.exit(1)

    # Choose model and init compressor
    use_v2 = not args.use_v1
    model_id = args.model or (LLMLINGUA2_SMALL if use_v2 else LLMLINGUA1_MODEL)

    print(f"llmlingua_compress: loading model '{model_id}' on {args.device} …", file=sys.stderr)
    print("  (first run downloads ~500 MB to ~/.cache/huggingface — subsequent runs are fast)", file=sys.stderr)

    t0 = time.perf_counter()
    from llmlingua import PromptCompressor  # noqa: PLC0415

    compressor = PromptCompressor(
        model_name=model_id,
        use_llmlingua2=use_v2,
        device_map=args.device,
    )
    t_load = time.perf_counter() - t0
    print(f"llmlingua_compress: model loaded in {t_load:.1f}s", file=sys.stderr)

    # Compress
    t1 = time.perf_counter()
    compress_kwargs: dict = dict(
        rate=args.target_ratio,
    )
    if args.instruction:
        compress_kwargs["instruction"] = args.instruction
    if args.question:
        compress_kwargs["question"] = args.question

    result = compressor.compress_prompt([raw_text], **compress_kwargs)
    t_compress = time.perf_counter() - t1

    compressed = result.get("compressed_prompt", "")

    # Info output to stderr
    if args.info:
        orig_tokens = count_tokens(raw_text)
        comp_tokens = count_tokens(compressed)
        achieved = comp_tokens / orig_tokens if orig_tokens else 0.0
        print(
            f"llmlingua_compress: original={orig_tokens} tokens  "
            f"compressed={comp_tokens} tokens  "
            f"ratio={achieved:.2%}  "
            f"compress_time={t_compress:.2f}s",
            file=sys.stderr,
        )

    # Compressed text → stdout only
    print(compressed, end="")


if __name__ == "__main__":
    main()
