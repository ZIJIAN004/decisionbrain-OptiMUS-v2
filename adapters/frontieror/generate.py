"""Generate one task's frozen conversion.

The converter never receives the instance data. It receives problem.md and a
read-only Python interface it can use to inspect the instance itself, and it
answers with code:

    transform.py       instance.json -> data.json  (renamed keys, real shapes,
                                                    padded arrays plus masks)
    parameters.json    [{symbol, shape, definition, source_key, role?}]
    targets.json       {background, constraints[], objective}

Everything is then checked mechanically. Rejections are fed back verbatim and
the converter tries again, up to MAX_GENERATION_ROUNDS.

The converter model is fixed and is not one of the models under evaluation, and
the artifacts are frozen and released, so every evaluated model sees the same
input.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import config, converter, manifest, validate
from .validate import ValidationError

EXPLORE_TIMEOUT_S = 900
TRANSFORM_TIMEOUT_S = 3600


class ConversionFailed(Exception):
    """The converter did not produce an accepted conversion within its budget."""


def explore(snippet: str, instance_path: Path) -> str:
    """Execute converter-written inspection code against the instance.

    The snippet is handed the parsed instance as ``data`` and may print whatever
    it wants; only its stdout comes back. Reading the instance this way, rather
    than through a schema summary we wrote, keeps the description of the data in
    the converter's own terms.
    """
    driver = (
        "import json, sys\n"
        f"data = json.load(open({str(instance_path)!r}, encoding='utf-8'))\n" + snippet
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(driver)
        script = Path(handle.name)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "adapters.frontieror.wrapper",
                "--cpu-cores",
                str(config.process_cpu_cores(config.JOBS)),
                "--cgroup-slice",
                config.CGROUP_SLICE,
                "--timeout",
                str(EXPLORE_TIMEOUT_S),
                "--cwd",
                str(config.REPO_ROOT),
                "--log",
                str(script.with_suffix(".log")),
                "--",
                sys.executable,
                str(script),
            ],
            cwd=str(config.REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        log = script.with_suffix(".log")
        output = (
            log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        )
        record = json.loads(proc.stdout.strip().splitlines()[-1])
        if record["outcome"] != "completed":
            output += f"\n[interrupted: {record['outcome']}]"
        return output[-20000:]
    finally:
        script.unlink(missing_ok=True)
        script.with_suffix(".log").unlink(missing_ok=True)


def run_transform(staging: Path, instance_path: Path) -> None:
    """Execute the converter's transform.py to produce data.json."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "adapters.frontieror.wrapper",
            "--cpu-cores",
            str(config.process_cpu_cores(config.JOBS)),
            "--cgroup-slice",
            config.CGROUP_SLICE,
            "--timeout",
            str(TRANSFORM_TIMEOUT_S),
            "--cwd",
            str(staging),
            "--log",
            str(staging / "transform.log"),
            "--",
            sys.executable,
            str(staging / "transform.py"),
            "--instance",
            str(instance_path),
            "--out",
            str(staging / "data.json"),
        ],
        cwd=str(config.REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        record = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise ValidationError(
            f"Could not run transform.py: {proc.stderr[-2000:]}"
        ) from None
    if record["outcome"] != "completed":
        log = (staging / "transform.log").read_text(encoding="utf-8", errors="replace")
        raise ValidationError(
            f"transform.py did not finish ({record['outcome']}):\n{log[-4000:]}"
        )


def build_input_targets(staging: Path) -> None:
    """Assemble what run.py:111 reads, from the converter's two answers.

    run.py expects `objective` as a single string and `constraints` as a list of
    strings; it wraps them into work items itself.
    """
    targets = json.loads((staging / "targets.json").read_text(encoding="utf-8"))
    params = json.loads((staging / "parameters.json").read_text(encoding="utf-8"))
    for required in ("background", "constraints", "objective"):
        if required not in targets:
            raise ValidationError(f"targets.json is missing '{required}'.")
    payload = {
        "background": targets["background"],
        "constraints": targets["constraints"],
        "objective": targets["objective"],
        # OptiMUS only reads symbol/shape/definition; source_key and role are
        # ours and are ignored downstream. Entries marked role=metadata stay in
        # parameters.json -- the record of what was judged non-modelling -- but
        # are not part of what OptiMUS is asked to model.
        "parameters": validate.modelling(params),
    }
    (staging / "input_targets.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def propose(problem_md: str, instance_path: Path, feedback: list[str]) -> dict:
    """Ask the fixed converter model for transform.py / parameters.json / targets.json.

    The converter inspects the instance through `explore` -- code it writes, run
    by us -- so its description of the data is its own rather than a summary we
    chose to show it.
    """
    return converter.converse(
        problem_md,
        lambda snippet: explore(snippet, instance_path),
        feedback,
    )


def generate(paper_id: str, case: dict) -> dict:
    instance_path = config.instance_path(paper_id, case["instance_index"])
    problem_md = config.problem_md_path(paper_id).read_text(encoding="utf-8")
    # Parsed once and reused by every assertion; the largest case measured
    # 9.7 GB resident, well inside the per-task cap.
    with instance_path.open(encoding="utf-8") as handle:
        instance = json.load(handle)

    feedback: list[str] = []
    staging_root = config.DATA_ROOT
    staging_root.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, config.MAX_GENERATION_ROUNDS + 1):
        staging = Path(
            tempfile.mkdtemp(dir=staging_root, prefix=f".staging-{paper_id}-")
        )
        try:
            answer = propose(problem_md, instance_path, feedback)
            for name, text in answer.items():
                (staging / name).write_text(text, encoding="utf-8")

            run_transform(staging, instance_path)
            build_input_targets(staging)
            report, coverage = validate.validate(
                staging, instance, lambda: run_transform(staging, instance_path)
            )
        except ValidationError as exc:
            feedback.append(str(exc))
            _discard(staging)
            continue
        except Exception:
            _discard(staging)
            raise

        manifest.commit(
            config.task_dir(paper_id),
            staging,
            manifest.Manifest(
                paper_id=paper_id,
                instance_index=case["instance_index"],
                instance_sha256=manifest.sha256_file(instance_path),
                transform_sha256=manifest.sha256_file(staging / "transform.py"),
                targets_sha256=manifest.sha256_file(staging / "input_targets.json"),
                converter_model=converter.model_id(),
                generation_rounds=attempt,
                sanity_check_passed=True,
                assertions=report,
                shape_coverage=coverage,
                generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )
        return {"rounds": attempt, "shape_coverage": coverage}

    raise ConversionFailed(
        f"{paper_id}: no accepted conversion after {config.MAX_GENERATION_ROUNDS} rounds. "
        f"Last rejection:\n{feedback[-1] if feedback else '(none)'}"
    )


def _discard(staging: Path) -> None:
    import shutil

    shutil.rmtree(staging, ignore_errors=True)
