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

# Everything a run produces lives under one directory inside the baseline repo,
# so every baseline's results sit together under one parent and nothing a run
# produces is mixed into the checkout. Frozen conversions stay in DATA_ROOT:
# they outlive any single run.
RUNS_ROOT = Path(
    os.environ.get("ADAPTER_RUNS_ROOT", "/home/bhz/baselines/optimus-v2-runs")
)


def new_run_dir(tag: str = "frontieror") -> Path:
    """RUNS_ROOT/<tag>-<UTC timestamp>/ with report.jsonl and logs/ inside it."""
    import time

    run_dir = RUNS_ROOT / f"{tag}-{time.strftime('%Y%m%d-%H%M%SZ', time.gmtime())}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


# --- FrontierOR sources (bhz host) -------------------------------------------
INDEX_JSON = Path(
    os.environ.get(
        "FRONTIEROR_INDEX",
        "/home/bhz/Decision Brain/benchmarks/frontieror-large-all/index.json",
    )
)
INSTANCE_ROOT = Path(
    os.environ.get("FRONTIEROR_INSTANCE_ROOT", "/home/bhz/FrontierOR_all")
)
PROBLEM_ROOT = Path(
    os.environ.get(
        "FRONTIEROR_PROBLEM_ROOT",
        "/home/bhz/Decision Brain/benchmarks/frontieror",
    )
)

# --- resource budget ---------------------------------------------------------
TOTAL_MEM_GB = 100
LOGICAL_CPUS = 32
PHYSICAL_CORES = 24
JOBS = int(os.environ.get("ADAPTER_JOBS", "1"))
CGROUP_SLICE = os.environ.get("ADAPTER_CGROUP_SLICE", "")
BASELINE_PYTHON_ENV = Path(
    os.environ.get(
        "ADAPTER_PYTHON_ENV", "/home/bhz/miniforge3/envs/decisionbrain_baseline"
    )
)
GUROBI_HOME = Path(os.environ.get("GUROBI_HOME", "/home/bhz/gurobi1302/linux64"))


def solver_threads(jobs: int) -> int:
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    return max(1, LOGICAL_CPUS // jobs)


def process_cpu_cores(jobs: int) -> int:
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if jobs > PHYSICAL_CORES:
        raise ValueError(f"jobs cannot exceed {PHYSICAL_CORES} physical cores")
    return PHYSICAL_CORES // jobs


# Matches DecisionBrain's --task-timeout-seconds 7200. The Evaluator separately
# gives Gurobi a 600-second TimeLimit so an incumbent can be formatted before
# this outer task deadline.
TASK_TIMEOUT_SECONDS = int(os.environ.get("ADAPTER_TASK_TIMEOUT", "7200"))

# Generation loop budget.
MAX_GENERATION_ROUNDS = int(os.environ.get("ADAPTER_MAX_ROUNDS", "8"))
# Fixed converter; deliberately not one of the models under evaluation, so
# every evaluated model receives byte-identical input.
CONVERTER_MODEL = os.environ.get("ADAPTER_CONVERTER_MODEL", "deepseek-v4-flash")


def instance_path(paper_id: str, instance_index: int) -> Path:
    return (
        INSTANCE_ROOT / paper_id / "instance" / f"large_instance_{instance_index}.json"
    )


def problem_md_path(paper_id: str) -> Path:
    return PROBLEM_ROOT / paper_id / "input" / "problem.md"


def solution_schema_path(paper_id: str) -> Path:
    return PROBLEM_ROOT / paper_id / "hidden" / "solution_schema.json"


def task_dir(paper_id: str) -> Path:
    """OptiMUS expects one flat directory per problem; paper_id is already unique
    across the 65 selected cases, so no compound id is needed."""
    return DATA_ROOT / paper_id


def load_cases() -> dict:
    import json

    with INDEX_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)["cases"]
