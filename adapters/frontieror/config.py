"""Paths and resource budget for the FrontierOR -> OptiMUS input adapter.

Every value here is a deliberate part of the evaluation setup and is released
alongside the frozen artifacts, so changing one changes the reported numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- repository layout -------------------------------------------------------
# run.py resolves data paths relative to the current working directory
# (run.py:111 and run.py:138), so the repo root is also the required cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "frontieror"

# --- FrontierOR sources (bhz host) -------------------------------------------
INDEX_JSON = Path(
    os.environ.get(
        "FRONTIEROR_INDEX",
        "/home/bhz/Decision Brain/benchmarks/frontieror-large-all/index.json",
    )
)
INSTANCE_ROOT = Path(os.environ.get("FRONTIEROR_INSTANCE_ROOT", "/home/bhz/FrontierOR_all"))
PROBLEM_ROOT = Path(
    os.environ.get(
        "FRONTIEROR_PROBLEM_ROOT",
        "/home/bhz/Decision Brain/benchmarks/frontieror",
    )
)

# --- resource budget ---------------------------------------------------------
# Measured on bhz: 128 GB total, ~123 GB available, shared with other users.
# One knob (TOTAL_BUDGET) divided by parallelism gives the per-task cgroup cap.
TOTAL_BUDGET_GB = int(os.environ.get("ADAPTER_TOTAL_BUDGET_GB", "100"))
JOBS = int(os.environ.get("ADAPTER_JOBS", "4"))
PER_TASK_MEM_GB = TOTAL_BUDGET_GB // JOBS

# Matches DecisionBrain's --task-timeout-seconds 7200. No inner solver limit is
# imposed: OptiMUS gets the full wall clock for solving, which favours the
# baseline relative to DecisionBrain's SOLVER_TIMEOUT=600.
TASK_TIMEOUT_SECONDS = int(os.environ.get("ADAPTER_TASK_TIMEOUT", "7200"))

# Generation loop budget.
MAX_GENERATION_ROUNDS = int(os.environ.get("ADAPTER_MAX_ROUNDS", "8"))
# Fixed converter; deliberately not one of the models under evaluation, so
# every evaluated model receives byte-identical input.
CONVERTER_MODEL = os.environ.get("ADAPTER_CONVERTER_MODEL", "deepseek-v4-flash")


def instance_path(paper_id: str, instance_index: int) -> Path:
    return INSTANCE_ROOT / paper_id / "instance" / f"large_instance_{instance_index}.json"


def problem_md_path(paper_id: str) -> Path:
    return PROBLEM_ROOT / paper_id / "input" / "problem.md"


def task_dir(paper_id: str) -> Path:
    """OptiMUS expects one flat directory per problem; paper_id is already unique
    across the 65 selected cases, so no compound id is needed."""
    return DATA_ROOT / paper_id


def load_cases() -> dict:
    import json

    with INDEX_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)["cases"]
