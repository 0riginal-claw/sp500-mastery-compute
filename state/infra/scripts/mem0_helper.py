#!/usr/bin/env python3
"""Mem0 persistent memory CLI wrapper.

Supports add/search/clear operations. Uses local Qdrant (in-process) for vector
storage and SQLite for history — no external services required when using a
local LLM embedder. With the default config, mem0 calls OpenAI for embeddings;
set OPENAI_API_KEY or pass --provider anthropic to change.

Canonical invocation:
  /Users/orginal/.venvs/sp500-mastery/bin/python scripts/mem0_helper.py \
      --add "text" --user_id <id> [--provider openai|anthropic] [--dry-run]
  /Users/orginal/.venvs/sp500-mastery/bin/python scripts/mem0_helper.py \
      --search "query" --user_id <id> [--limit 5] [--provider ...] [--dry-run]
  /Users/orginal/.venvs/sp500-mastery/bin/python scripts/mem0_helper.py \
      --clear --user_id <id> [--dry-run]

NOTE ON KEYS:
  - Default provider (openai): requires OPENAI_API_KEY env var.
  - --dry-run skips API calls and returns what would happen (still success=true).

Examples:
  # Add a memory
  /Users/orginal/.venvs/sp500-mastery/bin/python scripts/mem0_helper.py \
      --add "User prefers terse code with no docstrings." --user_id zach

  # Search memories
  /Users/orginal/.venvs/sp500-mastery/bin/python scripts/mem0_helper.py \
      --search "code style preferences" --user_id zach --limit 3

  # Clear all memories for a user
  /Users/orginal/.venvs/sp500-mastery/bin/python scripts/mem0_helper.py \
      --clear --user_id zach

  # Dry-run (no API key needed)
  /Users/orginal/.venvs/sp500-mastery/bin/python scripts/mem0_helper.py \
      --add "test" --user_id zach --dry-run
"""

import argparse
import json
import sys
import time


QDRANT_PATH = "/tmp/mem0_qdrant"
HISTORY_DB = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/data/mem0_history.db"


def build_config(provider: str) -> dict:
    base = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_aitools",
                "path": QDRANT_PATH,
            },
        },
        "history_db_path": HISTORY_DB,
        "version": "v1.1",
    }
    if provider == "openai":
        # OpenAI is the default; key must be in OPENAI_API_KEY env var
        pass
    elif provider == "anthropic":
        base["embedder"] = {
            "provider": "openai",
            "config": {},
        }
        base["llm"] = {
            "provider": "anthropic",
            "config": {},
        }
    elif provider == "ollama":
        # Local Ollama backend — no external API key required. Requires
        # `ollama serve` running on localhost:11434 with the listed models
        # pulled (`ollama pull nomic-embed-text`, `ollama pull llama3.2:3b`).
        # Use separate collection + 768-dim vectors (nomic-embed-text default)
        # to avoid dimensional mismatch with OpenAI's 1536-dim collection.
        # Audit-gap-closure 2026-05-20 Fix 6.
        base["vector_store"]["config"]["collection_name"] = "mem0_aitools_ollama"
        base["vector_store"]["config"]["embedding_model_dims"] = 768
        base["embedder"] = {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "ollama_base_url": "http://localhost:11434",
                "embedding_dims": 768,
            },
        }
        base["llm"] = {
            "provider": "ollama",
            "config": {
                "model": "llama3.2:3b",
                "ollama_base_url": "http://localhost:11434",
                "temperature": 0.1,
            },
        }
    return base


def main() -> None:
    t0 = time.perf_counter()

    try:
        parser = argparse.ArgumentParser(
            description="Mem0 persistent memory — add, search, or clear memories.",
        )
        parser.add_argument("--provider", type=str, default="openai",
                            choices=["openai", "anthropic", "ollama"],
                            help="LLM/embedder provider (default: openai — needs OPENAI_API_KEY; "
                                 "ollama uses localhost:11434 with nomic-embed-text + llama3.2:3b — "
                                 "requires `ollama serve` + pulled models)")

        # Mutually exclusive group for operation mode
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--add", type=str, default=None,
                          help="Add a memory (provide text content)")
        mode.add_argument("--search", type=str, default=None,
                          help="Search memories (provide query string)")
        mode.add_argument("--clear", action="store_true",
                          help="Clear all memories for the user")

        parser.add_argument("--user_id", type=str, required=True,
                            help="User/session identifier")
        parser.add_argument("--limit", type=int, default=5,
                            help="Max results to return for search (default: 5)")
        parser.add_argument("--dry-run", action="store_true",
                            help="Skip API calls, return what would happen")

        args = parser.parse_args()

        if args.add is not None:
            # Add mode
            text = args.add
            if not text.strip():
                resp = {
                    "success": False,
                    "tool_name": "mem0",
                    "error": "empty input for --add",
                    "latency_s": time.perf_counter() - t0,
                }
                print(json.dumps(resp))
                sys.exit(1)

            if args.dry_run:
                resp = {
                    "success": True,
                    "tool_name": "mem0",
                    "memory_id": "<dry-run>",
                    "latency_s": time.perf_counter() - t0,
                }
                print(json.dumps(resp))
                return

            from mem0 import Memory  # noqa: PLC0415

            cfg_dict = build_config(args.provider)
            mem = Memory.from_config(cfg_dict)
            result = mem.add(text, user_id=args.user_id)

            # Extract memory_id from result (structure depends on mem0 version)
            memory_id = result.get("id") or result.get("memory_id") or str(result.get("data", {}).get("id", "unknown"))

            resp = {
                "success": True,
                "tool_name": "mem0",
                "memory_id": memory_id,
                "latency_s": time.perf_counter() - t0,
            }
            print(json.dumps(resp))

        elif args.search is not None:
            # Search mode
            query = args.search

            if args.dry_run:
                resp = {
                    "success": True,
                    "tool_name": "mem0",
                    "memories": [],
                    "latency_s": time.perf_counter() - t0,
                }
                print(json.dumps(resp))
                return

            from mem0 import Memory  # noqa: PLC0415

            cfg_dict = build_config(args.provider)
            mem = Memory.from_config(cfg_dict)
            results = mem.search(query, user_id=args.user_id, limit=args.limit)

            resp = {
                "success": True,
                "tool_name": "mem0",
                "memories": results,
                "latency_s": time.perf_counter() - t0,
            }
            print(json.dumps(resp, default=str))

        elif args.clear:
            # Clear mode
            if args.dry_run:
                resp = {
                    "success": True,
                    "tool_name": "mem0",
                    "cleared": 0,
                    "latency_s": time.perf_counter() - t0,
                }
                print(json.dumps(resp))
                return

            from mem0 import Memory  # noqa: PLC0415

            cfg_dict = build_config(args.provider)
            mem = Memory.from_config(cfg_dict)

            # Try delete_all first; if not available, iterate and delete
            try:
                count = mem.delete_all(user_id=args.user_id)
            except (AttributeError, TypeError):
                # Fallback: get_all then delete each
                all_memories = mem.get_all(user_id=args.user_id)
                count = 0
                for memory in all_memories:
                    mem.delete(memory.get("id"), user_id=args.user_id)
                    count += 1

            resp = {
                "success": True,
                "tool_name": "mem0",
                "cleared": count,
                "latency_s": time.perf_counter() - t0,
            }
            print(json.dumps(resp))

    except Exception as e:
        latency = time.perf_counter() - t0
        resp = {
            "success": False,
            "tool_name": "mem0",
            "error": str(e),
            "latency_s": latency,
        }
        print(json.dumps(resp))
        sys.exit(1)


if __name__ == "__main__":
    main()
