"""Structural checks for the per-case bubblewrap boundary."""

import os
import tempfile
from pathlib import Path

from adapters.frontieror import config
from adapters.frontieror.sandbox import build_command, build_scope_command


def test_sandbox_exposes_only_current_case_inputs():
    assert config.TASK_MEM_GB == 100
    assert config.solver_threads(1) == 32
    assert config.solver_threads(4) == 8
    assert config.solver_threads(5) == 6
    assert config.process_cpu_cores(1) == 24
    assert config.process_cpu_cores(4) == 6
    assert config.process_cpu_cores(5) == 4
    scope = build_scope_command(["python", "run.py"], mem_gb=100, cpu_cores=6)
    assert "MemoryMax=100G" in scope
    assert "CPUQuota=600%" in scope
    root = Path(tempfile.mkdtemp())
    repo = root / "repo"
    python_env = root / "python-env"
    output = root / "output"
    task_data = repo / "data" / "frontieror" / "case-a"
    gurobi_home = root / "gurobi"
    for directory in (repo, python_env, output, task_data, gurobi_home):
        directory.mkdir(parents=True, exist_ok=True)
    (repo / ".git").mkdir()
    (repo / "logs").mkdir()
    (repo / "config.json").write_text("secret", encoding="utf-8")
    problem = root / "problem.md"
    schema = root / "schema.json"
    instance = root / "instance.json"
    license_file = root / "gurobi.lic"
    for path in (problem, schema, instance, license_file):
        path.write_text("{}", encoding="utf-8")

    command = build_command(
        ["python", "run.py"],
        repo=repo,
        python_env=python_env,
        output_dir=output,
        task_data=task_data,
        problem=problem,
        solution_schema=schema,
        instance=instance,
        license_file=license_file,
        gurobi_home=gurobi_home,
    )
    rendered = "\n".join(command)

    assert "--unshare-pid" in command
    assert "--unshare-net" not in command
    assert f"{os.sep}Decision Brain" not in rendered
    assert "feasibility_check.py" not in rendered
    assert "reference_solution" not in rendered
    assert str(repo / ".git") in rendered
    assert str(repo / "config.json") in rendered
    assert "/input/problem.md" in command
    assert "/input/solution_schema.json" in command
    assert "/input/instance.json" in command
    assert not any(
        command[index : index + 3] == ["--ro-bind", "/", "/"]
        for index in range(len(command) - 2)
    )


if __name__ == "__main__":
    test_sandbox_exposes_only_current_case_inputs()
    print("1/1 as expected")
