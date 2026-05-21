"""main.py — Replit entrypoint that hands off to start.sh.

Replit shows this file by default in the editor; the actual runtime is
driven by `.replit`'s `run = "bash start.sh"`.
"""
import os
import subprocess
import sys

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    rc = subprocess.call(["bash", os.path.join(here, "start.sh")])
    sys.exit(rc)
