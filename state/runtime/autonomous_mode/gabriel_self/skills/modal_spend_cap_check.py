"""Gabriel skill: modal_spend_cap_check.

Auto-registered by autonomous_mode_daemon._register_skill().
Pattern: Modal monthly spend approaching cap
"""
# SKILL_META: {"pattern": "Modal monthly spend approaching cap", "invocations": 1, "success_rate": 1.0, "updated_ts": "2026-05-20T20:46:40.135174+00:00", "promoted": false}

def modal_spend_cap_check():
    # Returns (current_usd, cap_usd, fraction_used). Drain queued jobs if >0.95.
    # Real impl reads dashboard/modal_spend.json populated by modal_spend_daemon.
    import json, os
    p = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com'
                            '/My Drive/AI-Tools/dashboard/modal_spend.json')
    try:
        d = json.loads(open(p).read())
        return d.get('current_usd', 0.0), d.get('cap_usd', 1.0), d.get('fraction_used', 0.0)
    except Exception:
        return 0.0, 1.0, 0.0

