#!/usr/bin/env python3
"""CausalRepack helper — auto-fallback wrapper.

INSTALLATION STATUS (re-verified 2026-05-20):
  The repository "causallm/CausalRepack" cited in the original task brief does
  not exist on GitHub. Verification:
    - HTTP 404 on github.com/causallm/CausalRepack
    - pip install --dry-run: "No matching distribution found for causalrepack"

Behavior (changed 2026-05-20 §G of sigterm_token_savers_spec):
  Instead of exiting 1 with an error, this wrapper now SILENTLY FALLS BACK
  to LLMLingua compression. This makes the helper drop-in usable from hooks
  (Stop / SubagentStop / PreToolUse on large Bash outputs) without breaking
  the calling hook.

  STDIN: text to compress (any size).
  STDOUT: compressed text.
  STDERR: one-line provenance ('causalrepack -> llmlingua fallback') unless
          --quiet is passed.

Usage in hooks (example — append to Stop hook for compression on long replies):
  /bin/bash -c 'tail -c 50000 | /Users/orginal/.venvs/sp500-mastery/bin/python \\
      /path/to/causalrepack_helper.py --target-ratio 0.5 > /tmp/last_compressed.txt'

When/if a real CausalRepack package is published, replace this file with a
real wrapper following the pattern in selective_context_helper.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DRIVE = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools"
)
VENV_PY = "/Users/orginal/.venvs/sp500-mastery/bin/python"
LLMLINGUA = DRIVE / "scripts" / "llmlingua_compress.py"


def fallback_compress(text: str, target_ratio: float = 0.5, quiet: bool = False) -> str:
    """Compress via LLMLingua. On failure, return the original text untouched."""
    if not LLMLINGUA.exists():
        if not quiet:
            print(
                "causalrepack_helper: WARNING — LLMLingua not found; "
                "returning input unchanged.",
                file=sys.stderr,
            )
        return text
    try:
        proc = subprocess.run(
            [
                VENV_PY,
                str(LLMLINGUA),
                "--target-ratio",
                str(target_ratio),
            ],
            input=text,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            if not quiet:
                print(
                    f"causalrepack_helper: LLMLingua exit={proc.returncode}; "
                    f"returning input unchanged. stderr={proc.stderr[:200]}",
                    file=sys.stderr,
                )
            return text
        if not quiet:
            print(
                "causalrepack_helper: causalrepack -> llmlingua fallback "
                f"(target_ratio={target_ratio})",
                file=sys.stderr,
            )
        return proc.stdout
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print(
                f"causalrepack_helper: LLMLingua subprocess failed ({exc}); "
                "returning input unchanged.",
                file=sys.stderr,
            )
        return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CausalRepack helper (auto-falls back to LLMLingua).",
    )
    parser.add_argument("--target-ratio", type=float, default=0.5,
                        help="Compression target (default 0.5 = 50%% retention)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress fallback provenance line on stderr")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Read input from file instead of stdin")
    parser.add_argument("--min-bytes", type=int, default=500,
                        help="Skip compression if input is smaller than this many bytes "
                             "(default 500 — avoids latency on small payloads)")
    args = parser.parse_args()

    if args.input_file:
        text = Path(args.input_file).read_text()
    else:
        text = sys.stdin.read()

    if len(text) < args.min_bytes:
        # Tiny input — pass through unchanged.
        sys.stdout.write(text)
        return

    compressed = fallback_compress(
        text, target_ratio=args.target_ratio, quiet=args.quiet,
    )
    sys.stdout.write(compressed)


if __name__ == "__main__":
    main()
