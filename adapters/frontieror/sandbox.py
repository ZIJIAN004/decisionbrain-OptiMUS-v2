"""Build the bubblewrap boundary for one evaluated OptiMUS case."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


class SandboxConfigurationError(ValueError):
    pass


def _required(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise SandboxConfigurationError(f"missing {label}: {resolved}")
    return resolved


def _home_parent_dirs(paths: list[Path]) -> list[str]:
    directories: set[str] = {"/home", "/home/bhz"}
    for path in paths:
        current = PurePosixPath(path.as_posix()).parent
        while str(current).startswith("/home/bhz/"):
            directories.add(str(current))
            current = current.parent
    return sorted(directories, key=lambda value: (value.count("/"), value))


def build_command(
    command: list[str],
    *,
    repo: str | Path,
    python_env: str | Path,
    output_dir: str | Path,
    task_data: str | Path,
    problem: str | Path,
    solution_schema: str | Path,
    instance: str | Path,
    license_file: str | Path,
    gurobi_home: str | Path | None = None,
) -> list[str]:
    repo_path = _required(repo, "repository")
    python_path = _required(python_env, "Python environment")
    output_path = _required(output_dir, "output directory")
    task_data_path = _required(task_data, "current task data")
    problem_path = _required(problem, "problem statement")
    schema_path = _required(solution_schema, "solution schema")
    instance_path = _required(instance, "instance")
    license_path = _required(license_file, "Gurobi license")
    gurobi_path = _required(gurobi_home, "Gurobi home") if gurobi_home else None

    home_paths = [repo_path, python_path, output_path, license_path]
    if gurobi_path is not None:
        home_paths.append(gurobi_path)

    argv = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
    ]
    for directory in _home_parent_dirs(home_paths):
        argv.extend(["--dir", directory])
    argv.extend(["--dir", "/etc"])

    for source in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(source).exists():
            argv.extend(["--ro-bind", source, source])
    for source in (
        "/etc/hosts",
        "/etc/resolv.conf",
        "/etc/nsswitch.conf",
        "/etc/localtime",
        "/etc/machine-id",
        "/etc/ssl",
        "/etc/ca-certificates",
    ):
        if Path(source).exists():
            argv.extend(["--ro-bind", source, source])

    argv.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--ro-bind",
            "/sys",
            "/sys",
            "--tmpfs",
            "/tmp",
            "--ro-bind",
            str(repo_path),
            str(repo_path),
            "--ro-bind",
            str(python_path),
            str(python_path),
            "--bind",
            str(output_path),
            str(output_path),
            "--ro-bind",
            str(license_path),
            str(license_path),
        ]
    )
    if gurobi_path is not None:
        argv.extend(["--ro-bind", str(gurobi_path), str(gurobi_path)])

    for relative in ("data", "logs", "runs", ".git"):
        target = repo_path / relative
        if target.exists():
            argv.extend(["--tmpfs", str(target)])
    for relative in (".env", "config.json"):
        target = repo_path / relative
        if target.exists():
            argv.extend(["--ro-bind", "/dev/null", str(target)])

    task_target = repo_path / "data" / "frontieror" / task_data_path.name
    argv.extend(
        [
            "--dir",
            str(repo_path / "data" / "frontieror"),
            "--ro-bind",
            str(task_data_path),
            str(task_target),
            "--dir",
            "/input",
            "--ro-bind",
            str(problem_path),
            "/input/problem.md",
            "--ro-bind",
            str(schema_path),
            "/input/solution_schema.json",
            "--ro-bind",
            str(instance_path),
            "/input/instance.json",
            "--setenv",
            "HOME",
            str(output_path),
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PATH",
            f"{python_path / 'bin'}:/usr/bin:/bin",
            "--chdir",
            str(repo_path),
            "--",
            *command,
        ]
    )
    return argv


def configured_license_file() -> Path:
    value = os.environ.get("GRB_LICENSE_FILE")
    if not value:
        raise SandboxConfigurationError(
            "GRB_LICENSE_FILE must be set before sandbox launch"
        )
    return Path(value)
