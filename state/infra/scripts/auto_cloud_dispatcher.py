"""
auto_cloud_dispatcher.py — Transparent subprocess → cloud_dispatch rerouter.

Monkey-patches subprocess.run / Popen / call / check_call / check_output so
that "heavy compute" Python script invocations are transparently rerouted to
cloud_dispatch.enqueue_job without touching each caller.

Public API:
    install()                   — activate patches (idempotent)
    uninstall()                 — restore originals
    register_pattern(regex)     — add a heavy-compute pattern at runtime
    disabled()                  — context manager: temporarily disable routing

Environment vars:
    AUTO_CLOUD_DISPATCH=0           — disable rerouting entirely (passthrough)
    AUTO_CLOUD_DISPATCH_FORCE_LOCAL=1 — same as disabled() for the process
    AUTO_CLOUD_DISPATCH_DRY_RUN=1   — log reroute decisions but don't enqueue

Log location:
    …/AI-Tools/logs/auto_cloud_dispatch/YYYY-MM-DD.log
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import sys
import threading
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DRIVE_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
)
_AI_TOOLS = _DRIVE_ROOT / "AI-Tools"
_LOG_DIR   = _AI_TOOLS / "logs" / "auto_cloud_dispatch"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Primary cloud_dispatch location (s&p500 project); trading project is a symlink variant
_CLOUD_DISPATCH_DIRS: list[Path] = [
    _AI_TOOLS / "s&p500-ticker-mastery" / "scripts",
    _AI_TOOLS / "trading-ticker-mastery" / "scripts",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log = logging.getLogger("auto_cloud_dispatcher")
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] auto_cloud_dispatcher — %(message)s",
        stream=sys.stderr,
    )


def _dispatch_log(record: dict) -> None:
    """Append one JSON-like line to the daily dispatch log."""
    try:
        log_path = _LOG_DIR / f"{date.today().isoformat()}.log"
        ts = datetime.now(timezone.utc).isoformat()
        parts = [f"ts={ts!r}"]
        for k, v in record.items():
            parts.append(f"{k}={v!r}")
        line = " ".join(parts) + "\n"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as exc:  # noqa: BLE001
        _log.debug("dispatch log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Heavy-compute patterns (compiled regexes matched against the script path)
# These are script basenames that represent per-ticker backtest / strategy runs.
# ---------------------------------------------------------------------------
_HEAVY_PATTERNS: list[re.Pattern] = [
    # Any orb_*.py script not caught by EXCLUDE_PATTERNS (coordinator/batch/etc.)
    re.compile(r"\borb", re.IGNORECASE),
    # Generic backtest scripts
    re.compile(r"\bbacktest", re.IGNORECASE),
    # VWAP single-ticker workers
    re.compile(r"\bvwap_(single|worker|run|ticker)", re.IGNORECASE),
    # Momentum single-ticker workers
    re.compile(r"\bmomentum_(single|worker|run|ticker)", re.IGNORECASE),
    # Catalyst single-ticker workers
    re.compile(r"\bcatalyst_(single|worker|run|ticker)", re.IGNORECASE),
    # Fade single-ticker workers
    re.compile(r"\bfade_(single|worker|run|ticker)", re.IGNORECASE),
    # Generic worker pattern (two-strategy, etc.)
    re.compile(r"run_two_strategy_worker", re.IGNORECASE),
]

# Scripts whose names match these patterns are NEVER rerouted even if they
# match a heavy pattern — orchestrators, daemons, monitors, data tools.
_EXCLUDE_PATTERNS: list[re.Pattern] = [
    re.compile(r"coordinator|dispatcher|daemon|orchestrator|launcher|monitor", re.IGNORECASE),
    re.compile(r"batch|sequential|parallel", re.IGNORECASE),
    re.compile(r"download|fetch|ingest|import", re.IGNORECASE),
    re.compile(r"report|summary|dashboard|sweep(?!.*worker)", re.IGNORECASE),
    re.compile(r"cloud_dispatch|auto_cloud", re.IGNORECASE),
    re.compile(r"multi_cloud", re.IGNORECASE),
]

_PATTERNS_LOCK = threading.Lock()

# Python executable names we recognize as "python invocation"
_PYTHON_EXES: set[str] = {"python", "python3", "python3.11", "python3.10", "python3.9"}

# Strategy guessed from script basename keywords.
# Use left-anchored \b only — right side often hits _ (word char), killing the boundary.
_STRATEGY_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bvwap",     re.IGNORECASE), "vwap"),
    (re.compile(r"\bcatalyst", re.IGNORECASE), "catalyst"),
    (re.compile(r"\bfade",     re.IGNORECASE), "fade"),
    (re.compile(r"\bmomentum", re.IGNORECASE), "momentum"),
    (re.compile(r"\borb",      re.IGNORECASE), "orb"),
    (re.compile(r"\bxgb",      re.IGNORECASE), "orb"),
    (re.compile(r"\bbacktest", re.IGNORECASE), "orb"),
]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_installed   = False
_disable_tls = threading.local()  # thread-local disable flag
_originals: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Pattern registration
# ---------------------------------------------------------------------------

def register_pattern(regex: str | re.Pattern) -> None:
    """Add a heavy-compute pattern at runtime.

    Args:
        regex: A regex string or compiled pattern matched against the script
               basename. Case-insensitive if provided as a string.
    """
    with _PATTERNS_LOCK:
        if isinstance(regex, str):
            regex = re.compile(regex, re.IGNORECASE)
        _HEAVY_PATTERNS.append(regex)
    _log.info("registered custom pattern: %s", regex.pattern)


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _is_heavy(script_path: str) -> bool:
    """Return True if script_path matches a heavy-compute pattern."""
    basename = Path(script_path).name
    with _PATTERNS_LOCK:
        if any(p.search(basename) for p in _EXCLUDE_PATTERNS):
            return False
        return any(p.search(basename) for p in _HEAVY_PATTERNS)


def _infer_strategy(script_path: str) -> str:
    basename = Path(script_path).name
    for pat, strat in _STRATEGY_MAP:
        if pat.search(basename):
            return strat
    return "orb"  # default fallback


def _looks_like_ticker(s: str) -> bool:
    """Rough heuristic: 1–5 uppercase letters, optionally with dot (e.g. BRK.B)."""
    return bool(re.match(r"^[A-Z]{1,5}(\.[A-Z])?$", s))


def _parse_command(args: list[str]) -> Optional[dict]:
    """Try to extract routing fields from a subprocess args list.

    Returns a dict with keys: exe, script, ticker, strategy, extra_args
    or None if the call shouldn't be rerouted.
    """
    if not args or len(args) < 2:
        return None

    exe = Path(args[0]).name
    if exe not in _PYTHON_EXES:
        return None

    script = args[1]
    # Skip -m / -c / other flags
    if script.startswith("-"):
        return None

    if not _is_heavy(script):
        return None

    strategy = _infer_strategy(script)
    ticker = None
    extra_args = args[2:]

    # Try to find ticker in remaining args
    for arg in extra_args:
        if _looks_like_ticker(arg):
            ticker = arg
            break

    if ticker is None:
        # Can't extract ticker — can't enqueue; passthrough
        return None

    return {
        "exe":        args[0],
        "script":     script,
        "ticker":     ticker,
        "strategy":   strategy,
        "extra_args": extra_args,
    }


def _get_cloud_dispatch():
    """Import cloud_dispatch, trying each registered scripts dir."""
    for scripts_dir in _CLOUD_DISPATCH_DIRS:
        if scripts_dir.exists() and str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
    try:
        import cloud_dispatch  # type: ignore[import]
        return cloud_dispatch
    except ImportError:
        return None


def _should_route() -> bool:
    """Return True if routing is currently enabled."""
    if os.environ.get("AUTO_CLOUD_DISPATCH", "1") == "0":
        return False
    if os.environ.get("AUTO_CLOUD_DISPATCH_FORCE_LOCAL", "0") == "1":
        return False
    if getattr(_disable_tls, "disabled", False):
        return False
    return True


def _dry_run() -> bool:
    return os.environ.get("AUTO_CLOUD_DISPATCH_DRY_RUN", "0") == "1"


def _do_reroute(parsed: dict, caller: str) -> Optional[str]:
    """Enqueue the job via cloud_dispatch. Returns job_id or None on dry-run."""
    dry = _dry_run()
    job_id = None

    if not dry:
        cd = _get_cloud_dispatch()
        if cd is None:
            _log.warning("cloud_dispatch unavailable — passthrough for %s", parsed["script"])
            _dispatch_log({
                "decision": "passthrough",
                "reason": "cloud_dispatch_unavailable",
                "cmd": " ".join([parsed["exe"], parsed["script"]] + list(parsed["extra_args"])),
                "caller": caller,
            })
            return None
        try:
            job_id = cd.enqueue_job(
                ticker=parsed["ticker"],
                strategy=parsed["strategy"],
                script=parsed["script"],
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("enqueue_job failed (%s) — passthrough for %s", exc, parsed["script"])
            _dispatch_log({
                "decision": "passthrough",
                "reason": f"enqueue_job_error:{exc}",
                "cmd": " ".join([parsed["exe"], parsed["script"]] + list(parsed["extra_args"])),
            })
            return None

    _log.info(
        "REROUTED%s %s %s/%s → cloud job_id=%s",
        " (dry-run)" if dry else "",
        parsed["script"], parsed["ticker"], parsed["strategy"], job_id,
    )
    _dispatch_log({
        "decision": "reroute" if not dry else "reroute_dry",
        "script":   parsed["script"],
        "ticker":   parsed["ticker"],
        "strategy": parsed["strategy"],
        "job_id":   job_id,
        "caller":   caller,
    })
    return job_id or "dry-run"


# ---------------------------------------------------------------------------
# Mock return objects for rerouted calls
# ---------------------------------------------------------------------------

class _MockCompletedProcess:
    """Stands in for subprocess.CompletedProcess when a job is rerouted."""
    def __init__(self, args):
        self.args       = args
        self.returncode = 0
        self.stdout     = b""
        self.stderr     = b""


class _MockPopen:
    """Stands in for subprocess.Popen when a job is rerouted."""
    def __init__(self, args):
        self.args       = args
        self.returncode = 0
        self.pid        = -1
        self.stdin      = None
        self.stdout     = None
        self.stderr     = None

    def poll(self):         return 0
    def wait(self, timeout=None): return 0
    def communicate(self, input=None, timeout=None): return b"", b""
    def kill(self):         pass
    def terminate(self):    pass
    def __enter__(self):    return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Patched versions
# ---------------------------------------------------------------------------

def _patched_run(args=None, *posargs, **kwargs):
    if _should_route() and isinstance(args, (list, tuple)):
        parsed = _parse_command(list(args))
        if parsed:
            job_id = _do_reroute(parsed, caller="subprocess.run")
            if job_id is not None:
                return _MockCompletedProcess(args)
    return _originals["run"](args, *posargs, **kwargs)


def _patched_popen_init(self, args=None, *posargs, **kwargs):
    if _should_route() and isinstance(args, (list, tuple)):
        parsed = _parse_command(list(args))
        if parsed:
            job_id = _do_reroute(parsed, caller="subprocess.Popen")
            if job_id is not None:
                mock = _MockPopen(args)
                self.__dict__.update(mock.__dict__)
                self.__class__ = _MockPopen
                return
    _originals["Popen.__init__"](self, args, *posargs, **kwargs)


def _patched_call(args=None, *posargs, **kwargs):
    if _should_route() and isinstance(args, (list, tuple)):
        parsed = _parse_command(list(args))
        if parsed:
            job_id = _do_reroute(parsed, caller="subprocess.call")
            if job_id is not None:
                return 0
    return _originals["call"](args, *posargs, **kwargs)


def _patched_check_call(args=None, *posargs, **kwargs):
    if _should_route() and isinstance(args, (list, tuple)):
        parsed = _parse_command(list(args))
        if parsed:
            job_id = _do_reroute(parsed, caller="subprocess.check_call")
            if job_id is not None:
                return 0
    return _originals["check_call"](args, *posargs, **kwargs)


def _patched_check_output(args=None, *posargs, **kwargs):
    if _should_route() and isinstance(args, (list, tuple)):
        parsed = _parse_command(list(args))
        if parsed:
            job_id = _do_reroute(parsed, caller="subprocess.check_output")
            if job_id is not None:
                return b""
    return _originals["check_output"](args, *posargs, **kwargs)


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------

def install() -> None:
    """Activate monkey-patches. Idempotent — safe to call multiple times."""
    global _installed
    if _installed:
        return
    _originals["run"]            = subprocess.run
    _originals["Popen.__init__"] = subprocess.Popen.__init__
    _originals["call"]           = subprocess.call
    _originals["check_call"]     = subprocess.check_call
    _originals["check_output"]   = subprocess.check_output

    subprocess.run          = _patched_run  # type: ignore[assignment]
    subprocess.Popen.__init__ = _patched_popen_init  # type: ignore[method-assign]
    subprocess.call         = _patched_call  # type: ignore[assignment]
    subprocess.check_call   = _patched_check_call  # type: ignore[assignment]
    subprocess.check_output = _patched_check_output  # type: ignore[assignment]

    _installed = True
    _log.info("auto_cloud_dispatcher installed")


def uninstall() -> None:
    """Restore original subprocess functions."""
    global _installed
    if not _installed:
        return
    subprocess.run          = _originals.pop("run")  # type: ignore[assignment]
    subprocess.Popen.__init__ = _originals.pop("Popen.__init__")  # type: ignore[method-assign]
    subprocess.call         = _originals.pop("call")  # type: ignore[assignment]
    subprocess.check_call   = _originals.pop("check_call")  # type: ignore[assignment]
    subprocess.check_output = _originals.pop("check_output")  # type: ignore[assignment]
    _installed = False
    _log.info("auto_cloud_dispatcher uninstalled")


@contextmanager
def disabled():
    """Context manager: temporarily disable routing for this thread."""
    _disable_tls.disabled = True
    try:
        yield
    finally:
        _disable_tls.disabled = False


# ---------------------------------------------------------------------------
# Self-tests (run with: python auto_cloud_dispatcher.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import tempfile
    import unittest

    # --- mock cloud_dispatch ---
    class _MockCloudDispatch:
        calls: list = []

        @staticmethod
        def enqueue_job(ticker, strategy, script, **_):
            job_id = f"mock{len(_MockCloudDispatch.calls):04d}"
            _MockCloudDispatch.calls.append({
                "ticker": ticker, "strategy": strategy,
                "script": script, "job_id": job_id,
            })
            return job_id

    # Inject mock into sys.modules so _get_cloud_dispatch() picks it up
    sys.modules["cloud_dispatch"] = _MockCloudDispatch  # type: ignore[assignment]

    class TestAutoRouter(unittest.TestCase):

        def setUp(self):
            _MockCloudDispatch.calls.clear()
            install()

        def tearDown(self):
            uninstall()

        def test_passthrough_non_python(self):
            """Non-python command passes through untouched."""
            result = subprocess.run(["echo", "hello"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(_MockCloudDispatch.calls, [])

        def test_passthrough_no_ticker(self):
            """Python heavy script but no ticker arg → passthrough (can't enqueue)."""
            try:
                subprocess.run(
                    ["python", "scripts/orb_single_ticker.py"],
                    capture_output=True, text=True,
                )
            except FileNotFoundError:
                pass  # expected: no ticker → passthrough; "python" may not be in PATH
            self.assertEqual(_MockCloudDispatch.calls, [])

        def test_reroute_orb_single(self):
            """orb_single_ticker.py with ticker arg → rerouted."""
            result = subprocess.run(
                ["python", "scripts/orb_single_ticker.py", "AAPL"],
            )
            self.assertIsInstance(result, _MockCompletedProcess)
            self.assertEqual(len(_MockCloudDispatch.calls), 1)
            call = _MockCloudDispatch.calls[0]
            self.assertEqual(call["ticker"], "AAPL")
            self.assertEqual(call["strategy"], "orb")
            self.assertIn("orb_single_ticker", call["script"])

        def test_reroute_backtest_xgb(self):
            """backtest_xgb_v8.py with ticker → rerouted with strategy=orb."""
            subprocess.run(["python3", "scripts/backtest_xgb_v8.py", "MSFT"])
            self.assertEqual(len(_MockCloudDispatch.calls), 1)
            self.assertEqual(_MockCloudDispatch.calls[0]["strategy"], "orb")

        def test_reroute_vwap_worker(self):
            """vwap_worker.py → strategy=vwap."""
            subprocess.run(["/usr/bin/python3", "scripts/vwap_worker.py", "TSLA"])
            self.assertEqual(_MockCloudDispatch.calls[0]["strategy"], "vwap")

        def test_no_reroute_coordinator(self):
            """Coordinator scripts are excluded."""
            self.assertFalse(_is_heavy("sweep_coordinator.py"))
            self.assertFalse(_is_heavy("multi_cloud_dispatcher.py"))

        def test_no_reroute_batch(self):
            """Batch orchestrators are excluded."""
            self.assertFalse(_is_heavy("orb_sequential_batch.py"))

        def test_disabled_context_manager(self):
            """disabled() context manager prevents rerouting."""
            with disabled():
                try:
                    subprocess.run(["python", "scripts/orb_single_ticker.py", "GOOG"])
                except FileNotFoundError:
                    pass  # passthrough → "python" not in PATH; expected
            self.assertEqual(_MockCloudDispatch.calls, [])

        def test_env_disable(self):
            """AUTO_CLOUD_DISPATCH=0 disables routing."""
            os.environ["AUTO_CLOUD_DISPATCH"] = "0"
            try:
                try:
                    subprocess.run(["python", "scripts/orb_single_ticker.py", "NVDA"])
                except FileNotFoundError:
                    pass  # passthrough → "python" not in PATH; expected
                self.assertEqual(_MockCloudDispatch.calls, [])
            finally:
                os.environ.pop("AUTO_CLOUD_DISPATCH")

        def test_dry_run(self):
            """DRY_RUN=1 logs but doesn't actually enqueue."""
            os.environ["AUTO_CLOUD_DISPATCH_DRY_RUN"] = "1"
            try:
                result = subprocess.run(
                    ["python", "scripts/orb_single_ticker.py", "AMZN"]
                )
                # dry-run returns mock immediately without calling enqueue_job
                self.assertIsInstance(result, _MockCompletedProcess)
                self.assertEqual(_MockCloudDispatch.calls, [])
            finally:
                os.environ.pop("AUTO_CLOUD_DISPATCH_DRY_RUN")

        def test_register_pattern(self):
            """register_pattern adds a custom heavy pattern."""
            register_pattern(r"my_custom_heavy_script")
            self.assertTrue(_is_heavy("my_custom_heavy_script.py"))

        def test_popen_reroute(self):
            """Popen with heavy script returns MockPopen."""
            proc = subprocess.Popen(["python", "scripts/orb_single_ticker.py", "SPY"])
            self.assertIsInstance(proc, _MockPopen)
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(len(_MockCloudDispatch.calls), 1)

        def test_call_reroute(self):
            """subprocess.call returns 0 for rerouted job."""
            rc = subprocess.call(["python", "scripts/orb_single_ticker.py", "QQQ"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(_MockCloudDispatch.calls), 1)

        def test_check_output_reroute(self):
            """subprocess.check_output returns b'' for rerouted job."""
            out = subprocess.check_output(["python", "scripts/orb_single_ticker.py", "IWM"])
            self.assertEqual(out, b"")

        def test_log_file_created(self):
            """Log file written for reroute decision."""
            subprocess.run(["python", "scripts/orb_single_ticker.py", "META"])
            log_path = _LOG_DIR / f"{date.today().isoformat()}.log"
            self.assertTrue(log_path.exists())
            content = log_path.read_text()
            self.assertIn("reroute", content)
            self.assertIn("META", content)

    print("Running auto_cloud_dispatcher self-tests…")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestAutoRouter)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
