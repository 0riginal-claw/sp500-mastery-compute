#!/usr/bin/env python3
"""Selective Context prompt compressor CLI wrapper (JSON output).

Uses perplexity-based token pruning (GPT-2) to drop low-information phrases.
No GPU required; runs CPU-only.

Canonical invocation:
  /Users/orginal/.venvs/sp500-mastery/bin/python scripts/selective_context_helper.py \
      --text "your long prompt" --reduce_ratio 0.5

  /Users/orginal/.venvs/sp500-mastery/bin/python scripts/selective_context_helper.py \
      --file /path/to/text.txt --reduce_ratio 0.35 --level sent
"""

import argparse
import json
import sys
import time


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def main() -> None:
    t_start = time.perf_counter()

    try:
        parser = argparse.ArgumentParser(
            description="Compress a prompt with Selective Context and emit JSON.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--text", type=str, default=None,
                            help="Input text to compress")
        group.add_argument("--file", type=str, default=None,
                            help="Path to file containing text to compress")
        parser.add_argument("--reduce_ratio", type=float, default=0.35,
                            help="Fraction of low-info tokens to remove (default: 0.35 = keep 65%%)")
        parser.add_argument("--level", type=str, default="phrase",
                            choices=["token", "phrase", "sent"],
                            help="Granularity of removal: token/phrase/sent (default: phrase)")
        parser.add_argument("--model", type=str, default="gpt2",
                            help="Language model for perplexity scoring (default: gpt2)")
        parser.add_argument("--lang", type=str, default="en",
                            help="Language code (default: en)")
        args = parser.parse_args()

        # Read input
        if args.text is not None:
            raw_text = args.text
        else:
            with open(args.file, "r", encoding="utf-8") as f:
                raw_text = f.read()

        if not raw_text.strip():
            json.dump({
                "success": False,
                "tool_name": "selective_context",
                "error": "empty input",
                "latency_s": round(time.perf_counter() - t_start, 3),
            }, sys.stdout)
            sys.exit(1)

        print(f"selective_context_helper: loading SelectiveContext (model={args.model}) …",
              file=sys.stderr)
        t0 = time.perf_counter()
        from selective_context import SelectiveContext  # noqa: PLC0415
        sc = SelectiveContext(model_type=args.model, lang=args.lang)
        t_load = time.perf_counter() - t0
        print(f"selective_context_helper: model loaded in {t_load:.1f}s", file=sys.stderr)

        # Time only the prune step
        t1 = time.perf_counter()
        compressed, _masked = sc(raw_text, reduce_ratio=args.reduce_ratio, reduce_level=args.level)
        t_compress = time.perf_counter() - t1

        orig_tokens = count_tokens(raw_text)
        comp_tokens = count_tokens(compressed)
        achieved_ratio = comp_tokens / orig_tokens if orig_tokens else 0.0

        print(
            f"selective_context_helper: original={orig_tokens} tokens  "
            f"compressed={comp_tokens} tokens  "
            f"ratio={achieved_ratio:.2%}  "
            f"prune_time={t_compress:.2f}s",
            file=sys.stderr,
        )

        # JSON output
        json.dump({
            "success": True,
            "tool_name": "selective_context",
            "pruned_text": compressed,
            "original_tokens": orig_tokens,
            "pruned_tokens": comp_tokens,
            "achieved_ratio": round(achieved_ratio, 3),
            "latency_s": round(time.perf_counter() - t_start, 3),
        }, sys.stdout)

    except Exception as e:
        json.dump({
            "success": False,
            "tool_name": "selective_context",
            "error": str(e),
            "latency_s": round(time.perf_counter() - t_start, 3),
        }, sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
