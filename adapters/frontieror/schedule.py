"""Run OptiMUS over the 65 FrontierOR cases.

One code path: a task whose frozen conversion is missing or stale is converted
first, then run. Conversion happens once; afterwards the manifest hashes match
and it is skipped.

Conversion failures are recorded as ``adapter_failed`` and kept out of the
baseline's own success statistics -- they are ours, not OptiMUS's.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config, manifest


def _run_wrapper(
    command: list[str],
    mem_gb: int,
    timeout_s: int,
    log_path: Path,
    sandbox: dict | None = None,
) -> dict:
    """Spawn wrapper.py so this task's resource usage is measured on its own."""
    argv = [
        sys.executable,
        "-m",
        "adapters.frontieror.wrapper",
        "--mem-gb",
        str(mem_gb),
        "--timeout",
        str(timeout_s),
        "--cwd",
        str(config.REPO_ROOT),
        "--log",
        str(log_path),
    ]
    if sandbox is not None:
        argv.extend(
            [
                "--sandbox",
                "--sandbox-python-env",
                str(sandbox["python_env"]),
                "--sandbox-output",
                str(sandbox["output_dir"]),
                "--sandbox-task-data",
                str(sandbox["task_data"]),
                "--sandbox-problem",
                str(sandbox["problem"]),
                "--sandbox-solution-schema",
                str(sandbox["solution_schema"]),
                "--sandbox-instance",
                str(sandbox["instance"]),
                "--sandbox-gurobi-home",
                str(sandbox["gurobi_home"]),
            ]
        )
    argv.extend(["--", *command])
    proc = subprocess.run(
        argv, cwd=str(config.REPO_ROOT), capture_output=True, text=True, check=False
    )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {
            "outcome": "wrapper_failed",
            "returncode": proc.returncode,
            "stderr": proc.stderr[-2000:],
        }


def process_case(
    paper_id: str,
    case: dict,
    mem_gb: int,
    solver_threads: int,
    log_dir: Path,
) -> dict:
    instance_path = config.instance_path(paper_id, case["instance_index"])
    task_dir = config.task_dir(paper_id)
    record: dict = {
        "paper_id": paper_id,
        "instance_bytes": case["instance_bytes"],
        "formulation_type": case["formulation_type"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if not manifest.is_fresh(task_dir, instance_path):
        from .generate import generate  # imported late: only this path needs an LLM

        try:
            record["conversion"] = generate(paper_id, case)
        except Exception as exc:  # noqa: BLE001
            record["outcome"] = "adapter_failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            return record
    else:
        record["conversion"] = "cached"

    artifact_dir = log_dir / f"{paper_id}-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result = _run_wrapper(
        [
            "python",
            "run.py",
            "--dataset",
            "frontieror",
            "--problem",
            paper_id,
            "--log_dir",
            str(artifact_dir),
            "--problem_path",
            "/input/problem.md",
            "--solution_schema",
            "/input/solution_schema.json",
            "--instance_path",
            "/input/instance.json",
            "--solver_threads",
            str(solver_threads),
        ],
        mem_gb=mem_gb,
        timeout_s=config.TASK_TIMEOUT_SECONDS,
        log_path=log_dir / f"{paper_id}.log",
        sandbox={
            "python_env": config.BASELINE_PYTHON_ENV,
            "output_dir": artifact_dir,
            "task_data": task_dir,
            "problem": config.problem_md_path(paper_id),
            "solution_schema": config.solution_schema_path(paper_id),
            "instance": instance_path,
            "gurobi_home": config.GUROBI_HOME,
        },
    )
    record.update(result)
    raw_solution = artifact_dir / "raw_solution.json"
    solution = artifact_dir / "solution.json"
    record["raw_candidate_available"] = raw_solution.is_file()
    record["solution_available"] = solution.is_file()
    if raw_solution.is_file():
        record["raw_solution_path"] = str(raw_solution)
    if solution.is_file():
        record["solution_path"] = str(solution)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=config.JOBS)
    parser.add_argument("--only", nargs="*", help="restrict to these paper_ids")
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="defaults to a new runs/ folder"
    )
    args = parser.parse_args()

    mem_gb = config.TASK_MEM_GB
    solver_threads = config.solver_threads(args.jobs)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(solver_threads)

    cases = config.load_cases()
    if args.only:
        cases = {k: v for k, v in cases.items() if k in set(args.only)}
    # Largest last: the two instances above 100 MB are the ones most likely to
    # hit the cap, and finishing the cheap cases first makes partial runs useful.
    ordered = sorted(cases.items(), key=lambda kv: kv[1]["instance_bytes"])

    print(
        f"{len(ordered)} cases | jobs={args.jobs} | {mem_gb} GB per task "
        f"| {solver_threads} solver threads per task "
        f"| {config.TASK_TIMEOUT_SECONDS}s wall",
        flush=True,
    )

    run_dir = args.run_dir or config.new_run_dir()
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.jsonl"
    print(f"run dir: {run_dir}", flush=True)

    with report_path.open("w", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    process_case, pid, case, mem_gb, solver_threads, log_dir
                ): pid
                for pid, case in ordered
            }
            for future in concurrent.futures.as_completed(futures):
                paper_id = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001
                    record = {
                        "paper_id": paper_id,
                        "outcome": "scheduler_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"  {paper_id:<20} {record.get('outcome', '?')}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
