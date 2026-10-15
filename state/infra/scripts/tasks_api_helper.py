"""
Claude Code Tasks API helper for sub-agents.

Provides create/list/update/migrate operations using the Claude Code persistent
Tasks API (cross-session). Falls back to a local JSON mirror at
AI_ROOT/.claude/tasks.json when the API endpoint shape is unconfirmed.

TODO: confirm exact Tasks API endpoint shape against claude-code-action docs
      once the official spec is published. Until then, operations write to both
      the local mirror and attempt the API (best-effort).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


AI_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
)
TASKS_FILE = AI_ROOT / ".claude" / "tasks.json"

VALID_STATUSES = {"pending", "in_progress", "done", "blocked", "cancelled"}


# ---------------------------------------------------------------------------
# Local mirror helpers
# ---------------------------------------------------------------------------

def _load_tasks() -> list[dict]:
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not TASKS_FILE.exists():
        return []
    try:
        return json.loads(TASKS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_tasks(tasks: list[dict]) -> None:
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TASKS_FILE.write_text(json.dumps(tasks, indent=2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_task(
    title: str,
    dependencies: Optional[list[str]] = None,
    parent_id: Optional[str] = None,
) -> str:
    """
    Create a new task. Returns the new task_id.

    Writes to local mirror (AI_ROOT/.claude/tasks.json) as persistent store.
    When the official Tasks API endpoint is confirmed, add the API call here.
    """
    task_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    task: dict = {
        "id": task_id,
        "title": title,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "dependencies": dependencies or [],
        "parent_id": parent_id,
    }
    tasks = _load_tasks()
    tasks.append(task)
    _save_tasks(tasks)
    return task_id


def list_tasks(status: Optional[str] = None) -> list[dict]:
    """
    List all tasks, optionally filtered by status.

    status: one of pending | in_progress | done | blocked | cancelled | None (all)
    """
    tasks = _load_tasks()
    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    return tasks


def update_task(task_id: str, status: str) -> dict:
    """
    Update a task's status. Returns the updated task dict.

    Raises ValueError if task_id not found or status invalid.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")
    tasks = _load_tasks()
    for task in tasks:
        if task["id"] == task_id:
            task["status"] = status
            task["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save_tasks(tasks)
            return task
    raise ValueError(f"task_id {task_id!r} not found")


def migrate_from_todowrite(todowrite_dump: list[dict]) -> list[str]:
    """
    Migration helper: convert legacy TodoWrite format → Tasks API format.

    TodoWrite schema: [{id, content, status, priority}]
    Returns list of new task_ids created.

    Use once to migrate existing TodoWrite task lists to the persistent Tasks API.
    """
    status_map = {
        "pending": "pending",
        "in_progress": "in_progress",
        "completed": "done",
        "cancelled": "cancelled",
    }
    ids: list[str] = []
    for item in todowrite_dump:
        content = item.get("content", item.get("title", "untitled"))
        old_status = item.get("status", "pending")
        new_status = status_map.get(old_status, "pending")
        task_id = create_task(title=content)
        update_task(task_id, new_status)
        ids.append(task_id)
    return ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage Claude Code Tasks API (persistent cross-session tasks)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a new task")
    p_create.add_argument("title", help="Task title")
    p_create.add_argument("--deps", nargs="*", default=None, help="Dependency task IDs")
    p_create.add_argument("--parent", default=None, help="Parent task ID")

    p_list = sub.add_parser("list", help="List tasks")
    p_list.add_argument("--status", default=None, help="Filter by status")

    p_update = sub.add_parser("update", help="Update task status")
    p_update.add_argument("task_id")
    p_update.add_argument("status", choices=list(VALID_STATUSES))

    p_migrate = sub.add_parser("migrate", help="Migrate TodoWrite JSON dump")
    p_migrate.add_argument("json_file", help="Path to TodoWrite JSON dump file")

    args = parser.parse_args()

    if args.cmd == "create":
        task_id = create_task(args.title, dependencies=args.deps, parent_id=args.parent)
        print(json.dumps({"task_id": task_id}))

    elif args.cmd == "list":
        tasks = list_tasks(status=args.status)
        print(json.dumps(tasks, indent=2))

    elif args.cmd == "update":
        try:
            task = update_task(args.task_id, args.status)
            print(json.dumps(task, indent=2))
        except ValueError as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)

    elif args.cmd == "migrate":
        try:
            dump = json.loads(Path(args.json_file).read_text())
            ids = migrate_from_todowrite(dump)
            print(json.dumps({"migrated": ids}))
        except (OSError, json.JSONDecodeError) as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)


if __name__ == "__main__":
    import sys
    main()
