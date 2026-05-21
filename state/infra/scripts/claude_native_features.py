"""
Wraps native Claude Code 2026 slash commands so sub-agents (which can't type slash
commands directly) can invoke them via subprocess shell-outs to the `claude` CLI.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Optional


AI_ROOT = (
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
)


def _claude_bin() -> str:
    """Return path to claude CLI, raise if not found."""
    path = shutil.which("claude")
    if not path:
        raise FileNotFoundError(
            "claude binary not found in PATH. "
            "Launch via AI-Tools/bin/claude-gdrive or add claude to PATH."
        )
    return path


def _run(args: list[str], timeout: int = 120) -> dict:
    """Run a subprocess, return {success, stdout, stderr, returncode}."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        raw = result.stdout.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return {
            "success": result.returncode == 0,
            "data": parsed,
            "stdout": raw,
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "data": {}, "stdout": "", "stderr": "timeout", "returncode": -1}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "data": {}, "stdout": "", "stderr": str(exc), "returncode": -1}


def invoke_ultraplan(task: str, files: Optional[list[str]] = None) -> dict:
    """
    Invoke /ultraplan for multi-file architecture planning.

    Use when task touches >3 files OR involves architecture changes.
    Maps to: claude --print --output-format stream-json -p "/ultraplan <task>"
    """
    claude = _claude_bin()
    file_hint = f" Files: {', '.join(files)}" if files else ""
    prompt = f"/ultraplan {task}{file_hint}"
    return _run(
        [claude, "--print", "--output-format", "stream-json", "-p", prompt],
        timeout=300,
    )


def invoke_batch(file_list: list[str], transformation: str) -> dict:
    """
    Invoke /batch for parallel same-transformation across 3+ files.

    Use when applying identical refactor/edit to multiple files simultaneously.
    """
    claude = _claude_bin()
    files_str = " ".join(file_list)
    prompt = f"/batch {files_str} -- {transformation}"
    return _run(
        [claude, "--print", "--output-format", "stream-json", "-p", prompt],
        timeout=300,
    )


def invoke_loop(prompt: str, max_iter: int = 10) -> dict:
    """
    Invoke /loop for iterative optimization until eval improves.

    Replaces ad-hoc while-True Python loops in autoresearch workflows.
    max_iter: safety cap passed to /loop's --max flag.
    """
    claude = _claude_bin()
    loop_prompt = f"/loop --max {max_iter} {prompt}"
    return _run(
        [claude, "--print", "--output-format", "stream-json", "-p", loop_prompt],
        timeout=600,
    )


def invoke_ctx_viz() -> dict:
    """
    Invoke /ctx-viz and parse context-usage report.

    MUST be called at any 5-min check-in if context concern surfaces.
    """
    claude = _claude_bin()
    return _run(
        [claude, "--print", "--output-format", "stream-json", "-p", "/ctx-viz"],
        timeout=30,
    )


def recommend_feature(task_description: str) -> str:
    """
    Heuristic: map task description → recommended Claude Code native feature.

    Returns one of: /ultraplan, /batch, /loop, /ctx-viz, or None.
    """
    desc = task_description.lower()

    # /batch: same transform across multiple files
    batch_signals = [
        "refactor",
        "rename",
        "replace",
        "update all",
        "across files",
        "across all",
        "multiple files",
        "same change",
        "transform",
    ]
    # count file mentions
    file_count_signals = [" files", " scripts", " modules"]
    file_count = sum(1 for s in file_count_signals if s in desc)
    digit_mentions = [int(t) for t in desc.split() if t.isdigit() and int(t) >= 3]
    has_multi_file_count = bool(digit_mentions) or file_count >= 1

    if any(s in desc for s in batch_signals) and has_multi_file_count:
        return "/batch"

    # /ultraplan: architecture, planning, cross-file dependencies, sweeps
    ultraplan_signals = [
        "plan",
        "architect",
        "design",
        "sweep",
        "cross-ticker",
        "cross-file",
        "dependency",
        "dependencies",
        "multi-step",
        "multi-file",
        "restructure",
        "overhaul",
        "migrate",
    ]
    if any(s in desc for s in ultraplan_signals):
        return "/ultraplan"

    # /loop: iterative optimization
    loop_signals = [
        "iterate",
        "optimize",
        "improve until",
        "keep improving",
        "autoresearch",
        "tune",
        "hyperparameter",
        "loop",
    ]
    if any(s in desc for s in loop_signals):
        return "/loop"

    # /ctx-viz: context/token concerns
    ctx_signals = ["context", "token", "ctx", "context window", "memory"]
    if any(s in desc for s in ctx_signals):
        return "/ctx-viz"

    return "none (task fits inline work — no slash command needed)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invoke Claude Code 2026 native slash commands from sub-agents."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("ultraplan", help="Run /ultraplan on a task")
    p_up.add_argument("task", help="Task description")
    p_up.add_argument("--files", nargs="*", default=None, help="Relevant file paths")

    p_batch = sub.add_parser("batch", help="Run /batch across files")
    p_batch.add_argument("files", nargs="+", help="Files to transform")
    p_batch.add_argument("--transformation", required=True, help="Transformation description")

    p_loop = sub.add_parser("loop", help="Run /loop iterative optimization")
    p_loop.add_argument("prompt", help="Loop prompt")
    p_loop.add_argument("--max-iter", type=int, default=10)

    sub.add_parser("ctx-viz", help="Run /ctx-viz context usage report")

    p_rec = sub.add_parser("recommend", help="Recommend which feature to use")
    p_rec.add_argument("task", help="Task description")

    args = parser.parse_args()

    if args.cmd == "ultraplan":
        result = invoke_ultraplan(args.task, args.files)
        print(json.dumps(result, indent=2))
    elif args.cmd == "batch":
        result = invoke_batch(args.files, args.transformation)
        print(json.dumps(result, indent=2))
    elif args.cmd == "loop":
        result = invoke_loop(args.prompt, args.max_iter)
        print(json.dumps(result, indent=2))
    elif args.cmd == "ctx-viz":
        result = invoke_ctx_viz()
        print(json.dumps(result, indent=2))
    elif args.cmd == "recommend":
        print(recommend_feature(args.task))


if __name__ == "__main__":
    main()
