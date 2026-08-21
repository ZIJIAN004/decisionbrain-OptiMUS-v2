"""Run one command under a memory cap and a wall clock, then report what it cost.

Invoked as a subprocess, one per task. Running it in its own process is what
makes the accounting per-task: getrusage(RUSAGE_CHILDREN) reports a high-water
mark across *all* children of the calling process, so a single scheduler process
supervising several tasks at once could only report their maximum, not each
task's own peak.

Prints one JSON object on stdout. Everything the supervised command writes goes
to the log file, not here.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from .sandbox import build_command, configured_license_file

# Verified on bhz (systemd 255, cgroup v2 with the memory controller): a process
# that exceeds MemoryMax is SIGKILLed inside its own scope and the host is
# unaffected. gurobipy is an in-process library, so Gurobi's allocations count
# towards this cap, and the scope covers every descendant.
SCOPE_CMD = ["systemd-run", "--user", "--scope", "-q"]


def run_capped(
    command: list[str],
    mem_gb: int,
    timeout_s: int,
    cwd: Path,
    log_path: Path,
    sandbox: dict | None = None,
) -> dict:
    supervised = (
        build_command(
            command,
            repo=cwd,
            license_file=configured_license_file(),
            **sandbox,
        )
        if sandbox is not None
        else command
    )
    argv = SCOPE_CMD + [
        "-p",
        f"MemoryMax={mem_gb}G",
        "-p",
        "MemorySwapMax=0",
        *supervised,
    ]
    started = time.time()
    timed_out = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            argv, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT
        )
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            returncode = proc.wait()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)

    return {
        "returncode": returncode,
        "outcome": classify(returncode, timed_out),
        "peak_rss_gb": round(usage.ru_maxrss / 1e6, 3),
        "cpu_user_s": round(usage.ru_utime, 1),
        "cpu_sys_s": round(usage.ru_stime, 1),
        "wall_s": round(time.time() - started, 1),
        "mem_cap_gb": mem_gb,
        "timeout_s": timeout_s,
        "log": str(log_path),
    }


def classify(returncode: int, timed_out: bool) -> str:
    """Separate resource limits from the baseline's own failures, so that a task
    killed by our budget is never scored as OptiMUS getting the answer wrong."""
    if timed_out:
        return "task_timeout"
    if returncode == -9:
        return "memory_exceeded"
    if returncode == 0:
        return "completed"
    return "optimus_failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mem-gb", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--sandbox-python-env", type=Path)
    parser.add_argument("--sandbox-output", type=Path)
    parser.add_argument("--sandbox-task-data", type=Path)
    parser.add_argument("--sandbox-problem", type=Path)
    parser.add_argument("--sandbox-solution-schema", type=Path)
    parser.add_argument("--sandbox-instance", type=Path)
    parser.add_argument("--sandbox-gurobi-home", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")

    sandbox = None
    if args.sandbox:
        required = {
            "python_env": args.sandbox_python_env,
            "output_dir": args.sandbox_output,
            "task_data": args.sandbox_task_data,
            "problem": args.sandbox_problem,
            "solution_schema": args.sandbox_solution_schema,
            "instance": args.sandbox_instance,
            "gurobi_home": args.sandbox_gurobi_home,
        }
        missing = [
            name
            for name, value in required.items()
            if value is None and name != "gurobi_home"
        ]
        if missing:
            parser.error(f"sandbox requires: {', '.join(missing)}")
        sandbox = required
    record = run_capped(
        command, args.mem_gb, args.timeout, args.cwd, args.log, sandbox=sandbox
    )
    json.dump(record, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
