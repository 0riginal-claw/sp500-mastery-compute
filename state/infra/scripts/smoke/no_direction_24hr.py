#!/usr/bin/env python3
"""24hr simulated smoke test for the NO-DIRECTION self-directing modules.

Fast-forwards the clock 60× (1 wall-second = 1 simulated minute), runs the
six new generator paths directly for 1440 simulated minutes (24hr equivalent),
and asserts the success criteria from the spawn brief:

  ≥50 user-style spawns
  ≥3 distinct curiosity-area entries
  ≥1 lessons.md addition
  ≥1 skill-library file present after smoke
  ≥1 hierarchical goal-tree low-atom appended

Does NOT spawn any real Claude / DeepSeek helpers — uses pure module-level
calls and synthetic outcomes.

Run:  python scripts/smoke/no_direction_24hr.py [--minutes N]
"""
from __future__ import annotations
import argparse, json, random, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools")
sys.path.insert(0, str(ROOT / "scripts"))

# Import the daemon WITHOUT running its main loop.
import autonomous_mode_daemon as amd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=1440,
                    help="simulated minutes (default 1440 = 24hr)")
    ap.add_argument("--cycle-seconds", type=int, default=90,
                    help="logical seconds per simulated cycle (default 90)")
    args = ap.parse_args()

    # Hard-reset gabriel_self state so the smoke is reproducible.
    self_dir = amd.GABRIEL_SELF_DIR
    self_dir.mkdir(parents=True, exist_ok=True)
    (self_dir / "skills").mkdir(parents=True, exist_ok=True)
    # Don't wipe affe28e9's files. Only reset the ones we own.
    for f in ("user_predictor.json", "curiosity_state.json", "goal_tree.json",
              "intrinsic_rewards.jsonl"):
        p = self_dir / f
        if p.exists():
            p.unlink()
    for s in (self_dir / "skills").glob("*.py"):
        s.unlink()

    # Reset lessons.md (own line) — keep a marker so we can verify our writes.
    lessons_before = amd.LESSONS_FILE.read_text() if amd.LESSONS_FILE.exists() else ""

    # ── Simulation ─────────────────────────────────────────────────────────
    n_cycles = (args.minutes * 60) // args.cycle_seconds
    print(f"[smoke] Running {n_cycles} simulated cycles "
          f"({args.minutes} sim-min, {args.cycle_seconds}s/cycle)")

    # Mock current-time function for time-aware seeds.
    sim_start = datetime(2026, 5, 20, 13, 0, tzinfo=timezone.utc)  # 06:00 PT
    # Pre-seed prompt history so predictor has something to mine.
    history = amd._load_user_prompt_history(n=10**6, hours=None)
    print(f"[smoke] prompt history rows on disk: {len(history)}")

    spawn_count = 0
    areas_touched: set[str] = set()
    skill_invocations = 0
    low_atoms_added = 0
    predictor_fires = 0
    curiosity_fires = 0
    goal_fires = 0
    time_aware_fires = 0

    # Bootstrap skill library.
    amd._bootstrap_skill_library()
    seed_skill_count = len(list((self_dir / "skills").glob("*.py")))
    print(f"[smoke] seed skill library size: {seed_skill_count}")

    rng = random.Random(42)

    for cycle in range(n_cycles):
        sim_now = sim_start + timedelta(seconds=cycle * args.cycle_seconds)
        cycle_id = f"sim{cycle:05d}"

        # 1. Refresh predictor (once per "day" to mimic daemon cadence).
        if cycle % 16 == 0:
            amd._refresh_user_predictor()

        # 2. Update goal tree with synthetic blockers + landings.
        synth_blockers = [
            {"blocker": rng.choice([
                "tickers not yet mastered for AAPL, MSFT, GOOG",
                "paper trade Sharpe below target",
                "Friday retrain manual step still required",
                "cloud routing only 60% off Mac",
                "lessons.md not net-positive this week",
            ]), "severity": rng.randint(4, 9)},
        ]
        synth_landings = [f"helper_{cycle}_{rng.choice(['scale','fix','audit'])}"]
        gt = amd._update_goal_tree(cycle_id, synth_blockers, synth_landings)
        # Track new low atom this cycle (if any).
        if gt.get("low") and gt["low"][-1].get("id") == f"L_{cycle_id}":
            low_atoms_added += 1

        # 3. Predictor candidate.
        try:
            pred = amd._predict_user_request({
                "hour": sim_now.hour, "cycle_id": cycle_id,
                "recent_landings": [], "blockers_count": len(synth_blockers),
            })
            if pred:
                predictor_fires += 1
                spawn_count += 1
        except Exception as e:
            print(f"[smoke] predictor err: {e}")

        # 4. Curiosity candidate (forced rate from constants).
        if rng.random() < amd.CURIOSITY_FORCED_RATE:
            cur = amd._curiosity_candidate(cycle_id)
            if cur:
                curiosity_fires += 1
                spawn_count += 1
                areas_touched.add(cur["_curiosity_area"])
                # Simulate the post-spawn touch.
                amd._touch_curiosity_area(cur["_curiosity_area"])

        # 5. Goal-tree candidate.
        gc = amd._goal_tree_candidate(cycle_id, gt)
        if gc:
            goal_fires += 1
            spawn_count += 1

        # 6. Time-aware seed (force pre-market + post-close windows by sim_now).
        ts = amd._time_aware_forced_seed(cycle_id, now=sim_now)
        if ts:
            time_aware_fires += 1
            spawn_count += 1

        # 7. Intrinsic reward at each "fired" spawn (synthetic outcome).
        for c in [pred, cur if 'cur' in dir() else None, gc, ts]:
            if not c:
                continue
            outcome = {
                "title": c.get("title", "?"),
                "area": c.get("_curiosity_area"),
                "estimated_effort_min": c.get("effort_min", 15),
                "actual_effort_min": c.get("effort_min", 15) + rng.randint(-5, 10),
                "status": rng.choice(["success", "success", "failed"]),
                "emitted_lesson_or_skill": rng.random() < 0.15,
                "novelty_delta": rng.uniform(0.0, 1.0),
            }
            reward = amd._intrinsic_reward(outcome)
            amd._log_intrinsic_reward(outcome, reward)

        # 8. Skill bump: occasionally match an existing skill.
        if cycle % 50 == 0:
            fake_cand = {"title": "kick_drive_sync_batch"}
            before = len(list((self_dir / "skills").glob("*.py")))
            amd._maybe_record_skill(fake_cand, outcome_status="success")
            after = len(list((self_dir / "skills").glob("*.py")))
            if after >= before:
                skill_invocations += 1

    # ── Assertions ─────────────────────────────────────────────────────────
    print(f"\n[smoke] === RESULTS ===")
    print(f"  spawn_count          = {spawn_count} (target ≥50)")
    print(f"  predictor_fires      = {predictor_fires}")
    print(f"  curiosity_fires      = {curiosity_fires}")
    print(f"  goal_tree_fires      = {goal_fires}")
    print(f"  time_aware_fires     = {time_aware_fires}")
    print(f"  unique_areas_touched = {len(areas_touched)} {sorted(areas_touched)}")
    print(f"  low_atoms_added      = {low_atoms_added}")
    print(f"  skill_bump_calls     = {skill_invocations}")

    skills_now = list((self_dir / "skills").glob("*.py"))
    print(f"  skill_library_size   = {len(skills_now)} (seeded: {seed_skill_count})")
    lessons_now = amd.LESSONS_FILE.read_text() if amd.LESSONS_FILE.exists() else ""
    lessons_delta = len(lessons_now) - len(lessons_before)
    print(f"  lessons.md_delta     = {lessons_delta} chars")

    ok_spawns   = spawn_count >= 50
    ok_areas    = len(areas_touched) >= 3
    ok_lessons  = lessons_delta > 0
    ok_skills   = len(skills_now) >= 1
    ok_low_atom = low_atoms_added >= 1

    print(f"\n[smoke] === VERDICT ===")
    print(f"  ≥50 spawns      : {'PASS' if ok_spawns else 'FAIL'}")
    print(f"  ≥3 areas        : {'PASS' if ok_areas else 'FAIL'}")
    print(f"  ≥1 lessons line : {'PASS' if ok_lessons else 'FAIL'}")
    print(f"  ≥1 skill file   : {'PASS' if ok_skills else 'FAIL'}")
    print(f"  ≥1 low atom     : {'PASS' if ok_low_atom else 'FAIL'}")

    return 0 if (ok_spawns and ok_areas and ok_lessons and ok_skills and ok_low_atom) else 1


if __name__ == "__main__":
    sys.exit(main())
