"""Focused tests for incumbent capture and fixed result formatting."""

import json
import tempfile
from pathlib import Path

from agents.evaluator import Evaluator
from agents.result_formatter import ResultFormatter


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        content = next(self.replies)
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Completion", (), {"choices": [choice]})()


def test_formatter_retries_tool_failure_and_writes_solution():
    root = Path(tempfile.mkdtemp())
    raw = root / "raw.json"
    data = root / "data.json"
    schema = root / "schema.json"
    problem = root / "problem.md"
    raw.write_text(
        json.dumps(
            {
                "objective_value": 4.0,
                "variables": {"x[0]": 1.0, "x[1]": 0.0},
            }
        ),
        encoding="utf-8",
    )
    data.write_text(json.dumps({"n": 2}), encoding="utf-8")
    schema.write_text(
        json.dumps({"objective_value": "<float>", "selected": "<list[int]>"}),
        encoding="utf-8",
    )
    problem.write_text("Select items.", encoding="utf-8")
    client = FakeClient(
        [
            "=====\ndef build_solution(raw_solution, data, instance):\n"
            "    raise ValueError('bad mapping')\n=====",
            (
                "=====\ndef build_solution(raw_solution, data, instance):\n"
                "    return {'objective_value': raw_solution['objective_value'], "
                "'selected': [0]}\n====="
            ),
        ]
    )
    formatter = ResultFormatter(client=client, llm="fake", max_attempts=2)
    state = {
        "raw_solution_path": str(raw),
        "target_solution_schema_path": str(schema),
        "data_json_path": str(data),
        "problem_path": str(problem),
        "log_folder": str(root),
        "objective": [],
        "variables": [],
        "constraint": [],
    }

    _, result = formatter.generate_reply("format", state, sender=None)

    assert result["solution_export_status"] == "completed"
    assert result["solution_export_attempts"] == 2
    assert json.loads((root / "solution.json").read_text()) == {
        "objective_value": 4.0,
        "selected": [0],
    }


def test_template_validation_rejects_missing_fields():
    try:
        ResultFormatter._validate_template({}, {"objective_value": "<float>"})
    except ValueError as exc:
        assert "missing required keys" in str(exc)
    else:
        raise AssertionError("missing target fields were accepted")


def test_evaluator_defaults_to_600_seconds():
    evaluator = Evaluator(client=FakeClient([]))
    assert evaluator.solver_time_limit == 600


def test_evaluator_captures_time_limit_incumbent():
    root = Path(tempfile.mkdtemp())
    data = root / "data.json"
    data.write_text("{}", encoding="utf-8")
    evaluator = Evaluator(client=FakeClient([]))
    evaluator.get_solver_prep_code = lambda: (
        """
class GRB:
    OPTIMAL=2; INFEASIBLE=3; INF_OR_UNBD=4; UNBOUNDED=5
    TIME_LIMIT=9; INTERRUPTED=11; SUBOPTIMAL=13
class GP: pass
gp = GP(); gp.GRB = GRB
class Variable:
    VarName='x[0]'; X=1.0
class Model:
    status=9; SolCount=1; ObjVal=7.5
    def setParam(self, name, value): self.time_limit = (name, value)
    def optimize(self): assert self.time_limit == ('TimeLimit', 600)
    def getVars(self): return [Variable()]
model = Model()
"""
    )
    state = {
        "data_json_path": str(data),
        "log_folder": str(root),
        "parameters": [],
        "variables": [],
        "constraint": [],
        "objective": [{"code": "# objective already defined"}],
    }

    result = evaluator._run_code(state)

    assert result["status_name"] == "time_limit"
    assert result["obj_val"] == 7.5
    saved = json.loads((root / "raw_solution.json").read_text())
    assert saved["variables"] == {"x[0]": 1.0}


if __name__ == "__main__":
    test_formatter_retries_tool_failure_and_writes_solution()
    test_template_validation_rejects_missing_fields()
    test_evaluator_defaults_to_600_seconds()
    test_evaluator_captures_time_limit_incumbent()
    print("4/4 as expected")
