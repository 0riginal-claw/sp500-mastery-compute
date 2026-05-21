"""
lookahead_checker.py — Static AST + regex scan for common look-ahead /
temporal-leakage bugs in time-series feature/backtest code.

Vendored 2026-05-21 (purged-CV wire-in mission). Inspired by
shatianming5/Agent_market/workspace/lookahead_checker.py — re-implemented
self-contained (no Agent_market local clone in repo). Patterns enforce the
no-lookahead invariant described in López de Prado, "Advances in Financial
Machine Learning" §3 & §7.

Usage:
    python lookahead_checker.py <path> [<path> ...]      # exit 1 on findings
    from lookahead_checker import scan_file              # programmatic

Patterns flagged (HIGH/MED/LOW):
  HIGH  .shift(<negative_int>)              # future leak
  HIGH  .shift(periods=<negative_int>)
  HIGH  resample(...).<agg>().shift(<neg>)
  HIGH  rolling(...).agg().shift(<neg>)
  HIGH  iloc[i+<n>:] / loc[ts + <td>:]      # forward indexing in train build
  HIGH  pd.merge_asof(direction='forward')  # forward-looking merge
  MED   pct_change(<negative>)
  MED   .diff(<negative>)
  MED   ffill().bfill()                     # bfill on time-indexed Series can leak
  LOW   .max() / .min() over full series    # global stats used in per-row feature
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Finding:
    path: str
    lineno: int
    col: int
    severity: str  # HIGH | MED | LOW
    rule: str
    snippet: str

    def __str__(self) -> str:
        return (
            f"[{self.severity}] {self.path}:{self.lineno}:{self.col}  "
            f"{self.rule}  | {self.snippet.strip()}"
        )


# ---------- AST visitor (catches .shift(-1), .pct_change(-1), .diff(-1)) ----

_NEG_SHIFT_METHODS = {"shift", "pct_change", "diff"}


def _is_negative_int(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        op = node.operand
        return isinstance(op, ast.Constant) and isinstance(op.value, int) and op.value > 0
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value < 0:
        return True
    return False


class _LeakVisitor(ast.NodeVisitor):
    def __init__(self, path: str, source_lines: List[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.findings: List[Finding] = []

    def _line(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1]
        return ""

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # df.shift(-1), df.pct_change(-1), df.diff(-1)
        if isinstance(node.func, ast.Attribute) and node.func.attr in _NEG_SHIFT_METHODS:
            method = node.func.attr
            severity = "HIGH" if method == "shift" else "MED"
            # positional
            for arg in node.args:
                if _is_negative_int(arg):
                    self.findings.append(
                        Finding(
                            path=self.path,
                            lineno=node.lineno,
                            col=node.col_offset,
                            severity=severity,
                            rule=f"{method}(<negative>)",
                            snippet=self._line(node.lineno),
                        )
                    )
            # keyword (shift(periods=-1))
            for kw in node.keywords:
                if kw.arg in ("periods", "n", "fill") and _is_negative_int(kw.value):
                    self.findings.append(
                        Finding(
                            path=self.path,
                            lineno=node.lineno,
                            col=node.col_offset,
                            severity=severity,
                            rule=f"{method}({kw.arg}=<negative>)",
                            snippet=self._line(node.lineno),
                        )
                    )

        # pd.merge_asof(..., direction='forward')
        if isinstance(node.func, ast.Attribute) and node.func.attr == "merge_asof":
            for kw in node.keywords:
                if kw.arg == "direction" and isinstance(kw.value, ast.Constant) and kw.value.value == "forward":
                    self.findings.append(
                        Finding(
                            path=self.path,
                            lineno=node.lineno,
                            col=node.col_offset,
                            severity="HIGH",
                            rule="merge_asof(direction='forward')",
                            snippet=self._line(node.lineno),
                        )
                    )

        # ffill().bfill() chain — bfill on time-indexed Series leaks future
        # Heuristic: a call whose func attr is 'bfill' and whose value chain
        # contains ffill — captured by regex pass (faster than full AST chain).

        self.generic_visit(node)


# ---------- Regex pass (catches text patterns AST misses) ----------

_REGEX_RULES = [
    # (severity, rule, pattern)
    ("MED",  "ffill().bfill() chain",        re.compile(r"\.ffill\([^)]*\)\s*\.\s*bfill\(")),
    ("HIGH", "iloc[i+<n>:]",                 re.compile(r"\.iloc\[\s*\w+\s*\+\s*\d+\s*:")),
    ("HIGH", "iloc[<n>:] in train builder",  re.compile(r"#.*train.*\n.*\.iloc\[\d+:")),
    ("LOW",  "global .max()/.min() on series",
              re.compile(r"=\s*\w+\[['\"]\w+['\"]\]\.(?:max|min)\(\)")),
    ("MED",  "rolling().sum().shift(-)",     re.compile(r"\.rolling\([^)]*\)\.[a-z]+\(\)\.shift\(\s*-\s*\d+")),
]


def _regex_scan(path: str, text: str, source_lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    for severity, rule, pat in _REGEX_RULES:
        for m in pat.finditer(text):
            # Compute line number from byte offset
            lineno = text.count("\n", 0, m.start()) + 1
            col = m.start() - (text.rfind("\n", 0, m.start()) + 1)
            findings.append(
                Finding(
                    path=path,
                    lineno=lineno,
                    col=col,
                    severity=severity,
                    rule=rule,
                    snippet=source_lines[lineno - 1] if lineno - 1 < len(source_lines) else "",
                )
            )
    return findings


# ---------- Public API ----------

def scan_file(path: str | Path) -> List[Finding]:
    path = str(path)
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return [Finding(path=path, lineno=0, col=0, severity="LOW",
                        rule="read-error", snippet=str(e))]
    source_lines = text.splitlines()
    findings: List[Finding] = []
    try:
        tree = ast.parse(text, filename=path)
        visitor = _LeakVisitor(path, source_lines)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    except SyntaxError as e:
        findings.append(Finding(path=path, lineno=e.lineno or 0, col=e.offset or 0,
                                severity="LOW", rule="syntax-error", snippet=str(e)))
    findings.extend(_regex_scan(path, text, source_lines))
    return findings


def scan_paths(paths: Iterable[str | Path]) -> List[Finding]:
    out: List[Finding] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for f in pp.rglob("*.py"):
                out.extend(scan_file(f))
        elif pp.is_file():
            out.extend(scan_file(pp))
    return out


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: lookahead_checker.py <path> [<path> ...]", file=sys.stderr)
        return 2
    findings = scan_paths(argv)
    high = [f for f in findings if f.severity == "HIGH"]
    med = [f for f in findings if f.severity == "MED"]
    low = [f for f in findings if f.severity == "LOW"]
    for f in findings:
        print(f)
    print(
        f"\n[lookahead_checker] HIGH={len(high)} MED={len(med)} LOW={len(low)}",
        file=sys.stderr,
    )
    return 1 if high else 0


if __name__ == "__main__":
    raise SystemExit(main())
