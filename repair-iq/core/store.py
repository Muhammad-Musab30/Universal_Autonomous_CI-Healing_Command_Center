"""Run History Store and Repair Memory Persistence Layer."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger("RepairIQ.Store")

BASE_DIR = Path(__file__).parent.parent.resolve()
RUNS_FILE = BASE_DIR / "runs_history.json"
MEMORY_FILE = BASE_DIR / "memory.json"

_RUNS_CACHE: Dict[str, Dict[str, Any]] = {}


def _init_runs_cache():
    global _RUNS_CACHE
    if RUNS_FILE.exists():
        try:
            with open(RUNS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
                if isinstance(records, list):
                    _RUNS_CACHE = {r["run_id"]: r for r in records if "run_id" in r}
                elif isinstance(records, dict):
                    _RUNS_CACHE = records
        except Exception as e:
            logger.warning(f"Could not load runs_history.json ({e}). Starting with empty cache.")
            _RUNS_CACHE = {}
    else:
        _RUNS_CACHE = {}


_init_runs_cache()


def _save_runs_to_disk():
    try:
        with open(RUNS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(_RUNS_CACHE.values()), f, indent=2)
    except Exception as e:
        logger.error(f"Error persisting runs to {RUNS_FILE}: {e}")


def save_run(
    run_id: str,
    scenario_id: str,
    seed: int,
    model_routing_policy: str,
    status: str,
    full_result: Dict[str, Any],
    dossier_markdown: str,
    reviewer_copilot: Dict[str, Any],
    github_pr_body: str
) -> Dict[str, Any]:
    """Saves a completed repair run to in-memory cache and persists to runs_history.json."""
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario_id": scenario_id,
        "seed": seed,
        "model_routing_policy": model_routing_policy,
        "status": status,
        "full_result": full_result,
        "dossier_markdown": dossier_markdown,
        "reviewer_copilot": reviewer_copilot,
        "github_pr_body": github_pr_body,
        "decision": None,
        "reviewer_feedback": None,
        "pr_url": None
    }
    _RUNS_CACHE[run_id] = record
    _save_runs_to_disk()
    return record


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a past run by run_id."""
    return _RUNS_CACHE.get(run_id)


def list_runs() -> List[Dict[str, Any]]:
    """Returns all past runs sorted by timestamp descending."""
    runs = list(_RUNS_CACHE.values())
    runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return runs


def update_run(run_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Updates fields in a run (such as human decision, pr_url, or status)."""
    if run_id not in _RUNS_CACHE:
        return None
    _RUNS_CACHE[run_id].update(updates)
    _save_runs_to_disk()
    return _RUNS_CACHE[run_id]


def get_memory_bank() -> List[Dict[str, Any]]:
    """Returns all records from memory.json."""
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read memory.json: {e}")
            return []
    return []


def append_memory_record(
    failure_signature: str,
    root_cause: str,
    accepted_patch_pattern: str,
    reviewer_feedback: str,
    project_convention: str,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Appends a new verified decision record into memory.json, enabling the system
    to learn continuously from human reviewer feedback.
    """
    records = get_memory_bank()
    new_id = f"MEM-{datetime.now().year}-{len(records) + 1:03d}"
    record = {
        "id": new_id,
        "failure_signature": failure_signature,
        "root_cause": root_cause,
        "accepted_patch_pattern": accepted_patch_pattern,
        "reviewer_feedback": reviewer_feedback,
        "project_convention": project_convention,
        "tags": tags or ["human_reviewed", "production_verified"],
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    records.append(record)
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to append to memory.json: {e}")

    return record
