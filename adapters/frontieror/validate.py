"""Validation oracle for a generated conversion.

Two layers:

1. OptiMUS's own ``prep_problem_json`` and ``sanity_check`` (utils/misc.py),
   loaded from source so the check that runs is the baseline's, not a copy.
2. Five extra assertions. ``sanity_check`` is not the whole input contract:
   half of what OptiMUS requires of a parameter list is enforced nowhere but
   inside the agents, by string comparison (agents/formulator.py:373,
   agents/programmer.py:557, agents/evaluator.py:105). A conversion can pass
   ``sanity_check`` and still be unusable, so those rules are checked here too,
   alongside the ones about faithfulness to the instance.
"""

from __future__ import annotations

import importlib.util
import json
import keyword
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


# --- roles -------------------------------------------------------------------

def _is_metadata(param: dict) -> bool:
    return param.get("role") == "metadata"


def modelling(params: list[dict]) -> list[dict]:
    """The parameters OptiMUS actually receives.

    Keys declared role=metadata stay in parameters.json -- so the record shows
    what was judged non-modelling, and key coverage still accounts for them --
    but they are not handed to the modelling agents.
    """
    return [p for p in params if not _is_metadata(p)]


# Names bound by the harness around the generated code: agents/evaluator.py
# prep_code (:14-24), get_solver_prep_code (:180) and post_code (:27-42). A
# parameter with one of these names would overwrite the model or the data.
RESERVED_NAMES = frozenset({"json", "np", "math", "gp", "model", "data", "f", "status", "obj_val"})

# Builtins the generated gurobipy code routinely calls. Shadowing one of these
# with a parameter is silent until the line that calls it.
SHADOWED_BUILTINS = frozenset(
    {"sum", "min", "max", "range", "len", "abs", "round", "int", "float", "list", "dict", "set", "str"}
)


# --- the five assertions ------------------------------------------------------

def _assert_symbol_contract(params: list[dict]) -> dict:
    """The half of the input contract that sanity_check does not check.

    utils/misc.py:205 tolerates one underscore in a symbol and utils/misc.py:222
    is satisfied by a dimension that merely exists in data.json. Neither is
    enough:

    * agents/formulator.py:373-374 matches the text inside \\textup{} against
      ``symbol`` by exact equality and raises on anything unmatched (:414), and
      agents/programmer.py:557 selects a constraint's parameters the same way.
      The formulator is instructed to write indices outside the symbol
      (formulator.py:101), so a symbol carrying its own index suffix can never
      match.
    * agents/evaluator.py:105-110 executes the ``code`` line of each *parameter*
      and nothing else, so a dimension that is not itself a parameter is simply
      not a defined name in the namespace the generated code runs in.
    """
    malformed = [p for p in params if not isinstance(p.get("symbol"), str) or not p["symbol"]]
    if malformed:
        raise ValidationError(
            f"{len(malformed)} entries in parameters.json have no usable `symbol`. "
            "Every entry needs a symbol, a shape and a definition."
        )

    real = modelling(params)
    declared = {p["symbol"] for p in real}
    scalars = {p["symbol"] for p in real if not p.get("shape")}

    suffixed = sorted(p["symbol"] for p in real if "_" in p["symbol"])
    if suffixed:
        raise ValidationError(
            f"These symbols carry an index suffix: {suffixed}. A symbol must be "
            "bare -- write `linearCost`, not `linearCost_ij`. OptiMUS matches the "
            "name inside \\textup{...} against the symbol by exact string "
            "equality, so a suffixed symbol can never be resolved. Move the "
            "indices into `shape`, and rename the matching data.json keys so each "
            "symbol is still spelled exactly like its key."
        )

    forbidden = RESERVED_NAMES | SHADOWED_BUILTINS
    unusable = sorted(
        p["symbol"] for p in real
        if not p["symbol"].isidentifier()
        or keyword.iskeyword(p["symbol"])
        or p["symbol"] in forbidden
    )
    if unusable:
        raise ValidationError(
            f"These symbols cannot be used as Python names: {unusable}. A symbol "
            "must be a valid identifier, must not be a Python keyword, and must "
            f"not be one of {sorted(forbidden)} -- names the generated code "
            "already binds or calls. Rename the symbol and its data.json key "
            "together so the two still match."
        )

    undeclared = sorted(
        {d for p in real for d in p.get("shape", [])} - declared
    )
    if undeclared:
        raise ValidationError(
            f"These dimension names are used in a shape but are not parameters: "
            f"{undeclared}. Being present in data.json is not enough -- only "
            "parameters are loaded into the namespace the generated code runs in, "
            "so indexing with one of these raises NameError. Add each as its own "
            'entry with shape [] and the same name, and keep its integer value in '
            "data.json."
        )

    non_scalar_dims = sorted(
        {d for p in real for d in p.get("shape", [])} & declared - scalars
    )
    if non_scalar_dims:
        raise ValidationError(
            f"These names are used as dimensions but are declared with a non-empty "
            f"shape: {non_scalar_dims}. A dimension must be a scalar integer "
            "parameter (shape [])."
        )

    bad_metadata = sorted(
        p["symbol"] for p in params if _is_metadata(p) and p.get("shape")
    )
    if bad_metadata:
        raise ValidationError(
            f"These parameters are role=metadata but declare a shape: "
            f"{bad_metadata}. Metadata carries no modelling content, so it has "
            "shape [] and no entry in data.json."
        )

    return {
        "modelling": len(real),
        "metadata": len(params) - len(real),
        "dimensions": len({d for p in real for d in p.get("shape", [])}),
    }


def _assert_no_scalarization(params: list[dict], instance: dict) -> dict:
    """A key that carries structure must keep it. shape == [] is legal in
    sanity_check (utils/misc.py:220 skips the dimension check entirely when the
    shape is empty), which is exactly the degenerate encoding we refuse.

    Dimension parameters are exempt. A padded array's width is a genuine scalar
    even though the key it comes from is a list, so flagging it would make the
    padding rule and this rule contradict each other.
    """
    real = modelling(params)
    dimension_names = {d for p in real for d in p.get("shape", [])}
    offenders = []
    for param in real:
        key = param.get("source_key")
        if key is None or param.get("role") == "mask":
            continue
        if param["symbol"] in dimension_names:
            continue
        if not _is_scalar(instance.get(key)) and not param.get("shape"):
            offenders.append(param["symbol"])
    if offenders:
        raise ValidationError(
            "These parameters are not scalars in the instance but were encoded "
            f"with shape []: {offenders}. Give each one its real dimensions, and "
            "declare every dimension name as its own integer entry in data.json."
        )
    return {"checked": len(real), "offenders": 0}


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
    as_metadata = {p.get("source_key") for p in params if _is_metadata(p)}
    ragged = {k for k, v in instance.items() if _is_ragged(v)}
    unmasked = sorted(ragged - masked - as_metadata)
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
    if not modelling(params):
        raise ValidationError(
            "Every parameter is marked role=metadata, so OptiMUS would receive "
            "nothing to model. Metadata is for values that cannot enter any "
            "formulation, not for the instance as a whole."
        )

    report = {
        "symbol_contract": _assert_symbol_contract(params),
        "no_scalarization": _assert_no_scalarization(params, instance),
        "key_coverage": _assert_key_coverage(params, instance),
        "ragged_has_mask": _assert_ragged_has_mask(params, instance),
    }

    # OptiMUS's own checks. prep_problem_json fills in the `code` field that
    # sanity_check then requires, so the order matters.
    state = {
        # Metadata never reaches OptiMUS, so it is not what sanity_check should
        # be given either -- input_targets.json carries the modelling half only.
        "parameters": [dict(p) for p in modelling(params)],
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

    real = modelling(params)
    shaped = sum(1 for p in real if p.get("shape"))
    coverage = {
        "parameters": len(real),
        "shaped": shaped,
        "scalar": len(real) - shaped,
        "mask_parameters": sum(1 for p in real if p.get("role") == "mask"),
        "metadata_parameters": len(params) - len(real),
    }
    return report, coverage
