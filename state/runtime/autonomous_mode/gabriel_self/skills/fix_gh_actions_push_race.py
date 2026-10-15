"""Gabriel skill: fix_gh_actions_push_race.

Auto-registered by autonomous_mode_daemon._register_skill().
Pattern: gh_actions push race / refs/heads/main update conflict
"""
# SKILL_META: {"pattern": "gh_actions push race / refs/heads/main update conflict", "invocations": 1, "success_rate": 1.0, "updated_ts": "2026-05-20T20:46:40.134208+00:00", "promoted": false}

def fix_gh_actions_push_race():
    # Pull --rebase before push; if still racing, queue via cloud_dispatch.
    import subprocess as _sp
    _sp.run(['git', 'pull', '--rebase', '--autostash'], check=False)
    return _sp.run(['git', 'push'], check=False).returncode == 0

