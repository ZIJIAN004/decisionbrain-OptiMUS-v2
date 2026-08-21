"""Validation oracle for a generated conversion.

Two layers:

1. OptiMUS's own ``prep_problem_json`` and ``sanity_check`` (utils/misc.py),
   loaded from source so the check that runs is the baseline's, not a copy.
2. Four extra assertions that ``sanity_check`` does not cover -- it only
   inspects parameter symbols and declared shapes, and says nothing about
   whether the conversion was faithful to the instance.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

from . import config


class ValidationError(Exception):
    """Raised when a generated conversion is rejected. Carries text that is fed
    back to the converter verbatim."""


def _load_optimus_misc() -> types.ModuleType:
    """Import utils/misc.py without pulling in its optional LLM clients.

    misc.py imports mistralai at module level purely for other functions in the
    file; stubbing it keeps prep_problem_json and sanity_check byte-identical to
    upstream while removing an irrelevant dependency.
    """
    if "mistralai" not in sys.modules:
        stub = types.ModuleType("mistralai")
        client = types.ModuleType("mistralai.client")
        client.MistralClient = object
        stub.client = client
        sys.modules["mistralai"] = stub
        sys.modules["mistralai.client"] = client

    path = config.REPO_ROOT / "utils" / "misc.py"
    spec = importlib.util.spec_from_file_location("_optimus_misc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- shape classification of raw instance values -----------------------------

def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool, str)) or value is None


def _is_ragged(value: Any) -> bool:
    """A nested list whose sibling elements disagree on length."""
    if not isinstance(value, list) or not value:
        return False
    lengths = {len(v) for v in value if isinstance(v, list)}
    if len(lengths) > 1:
        return True
    return any(_is_ragged(v) for v in value if isinstance(v, list))


# --- the four assertions -----------------------------------------------------

def _assert_no_scalarization(params: list[dict], instance: dict) -> dict:
    """A key that carries structure must keep it. shape == [] is legal in
    sanity_check (utils/misc.py:220 skips the dimension check entirely when the
    shape is empty), which is exactly the degenerate encoding we refuse."""
    offenders = []
    for param in params:
        key = param.get("source_key")
        if key is None or param.get("role") == "mask":
            continue
        if not _is_scalar(instance.get(key)) and not param.get("shape"):
            offenders.append(param["symbol"])
    if offenders:
        raise ValidationError(
            "These parameters are not scalars in the instance but were encoded "
            f"with shape []: {offenders}. Give each one its real dimensions, and "
            "declare every dimension name as its own integer entry in data.json."
        )
    return {"checked": len(params), "offenders": 0}


def _assert_key_coverage(params: list[dict], instance: dict) -> dict:
    """Nothing may be dropped, merged, or pre-aggregated: deciding a field is
    irrelevant is a modelling decision that belongs to OptiMUS, not to us."""
    covered = {p.get("source_key") for p in params}
    missing = sorted(set(instance) - covered)
    if missing:
        raise ValidationError(
            "These top-level keys of the instance are not represented by any "
            f"parameter: {missing}. Every key must appear; none may be dropped "
            "or folded into another."
        )
    unknown = sorted(covered - set(instance) - {None})
    if unknown:
        raise ValidationError(
            f"These source_key values do not exist in the instance: {unknown}."
        )
    return {"instance_keys": len(instance), "missing": 0}


def _assert_ragged_has_mask(params: list[dict], instance: dict) -> dict:
    """Padding a ragged array to a rectangle needs a fill value. Padding with 0
    and publishing a mask states which cells are real without deciding what the
    model should do about them; padding with a semantically neutral value (+inf,
    -1) would make that decision for OptiMUS."""
    masked = {p.get("source_key") for p in params if p.get("role") == "mask"}
    ragged = {k for k, v in instance.items() if _is_ragged(v)}
    unmasked = sorted(ragged - masked)
    if unmasked:
        raise ValidationError(
            "These keys are ragged in the instance and were padded without a "
            f"companion mask: {unmasked}. Pad with 0 and emit, for each one, a "
            "second parameter of the same shape marked role=mask whose entries "
            "are 1 for real cells and 0 for padding."
        )
    return {"ragged_keys": len(ragged), "unmasked": 0}


def _assert_determinism(task_staging: Path, run_transform) -> dict:
    """Same instance in, same data.json out -- otherwise the frozen artifact does
    not describe a reproducible run."""
    from .manifest import sha256_file

    data_json = task_staging / "data.json"
    first = sha256_file(data_json)
    run_transform()
    second = sha256_file(data_json)
    if first != second:
        raise ValidationError(
            "transform.py produced a different data.json on a second run. Remove "
            "every source of nondeterminism: iterate dicts in sorted key order, "
            "do not use set iteration order, and do not seed anything from time."
        )
    return {"sha256": first}


# --- entry point -------------------------------------------------------------

def validate(task_staging: Path, instance: dict, run_transform) -> tuple[dict, dict]:
    """Run both layers against a staged conversion.

    ``instance`` is the already-parsed instance.json; ``run_transform`` re-runs
    transform.py in place. Raises ValidationError with feedback text on failure.
    Returns (assertion report, shape coverage report).
    """
    misc = _load_optimus_misc()

    params_path = task_staging / "parameters.json"
    for path in (task_staging / "input_targets.json", params_path, task_staging / "data.json"):
        if not path.is_file():
            raise ValidationError(f"Missing required artifact: {path.name}")

    params = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(params, list) or not params:
        raise ValidationError("parameters.json must be a non-empty list.")

    report = {
        "no_scalarization": _assert_no_scalarization(params, instance),
        "key_coverage": _assert_key_coverage(params, instance),
        "ragged_has_mask": _assert_ragged_has_mask(params, instance),
    }

    # OptiMUS's own checks. prep_problem_json fills in the `code` field that
    # sanity_check then requires, so the order matters.
    state = {
        "parameters": [dict(p) for p in params],
        "data_json_path": str(task_staging / "data.json"),
    }
    try:
        # prep_problem_json is a staticmethod on NLParamParser; sanity_check is
        # module level. run.py calls them in exactly this order (run.py:120, :143).
        state = misc.NLParamParser.prep_problem_json(state)
        misc.sanity_check(state)
    except Exception as exc:  # noqa: BLE001 - text is fed back to the converter
        raise ValidationError(
            f"OptiMUS sanity_check rejected the conversion:\n{type(exc).__name__}: {exc}"
        ) from exc

    report["determinism"] = _assert_determinism(task_staging, run_transform)

    shaped = sum(1 for p in params if p.get("shape"))
    coverage = {
        "parameters": len(params),
        "shaped": shaped,
        "scalar": len(params) - shaped,
        "mask_parameters": sum(1 for p in params if p.get("role") == "mask"),
    }
    return report, coverage
