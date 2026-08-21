import json
import os
import re
import tempfile
from collections import defaultdict

import openai

from agents.agent import Agent


FORMATTER_PROMPT = """
You convert a saved Gurobi incumbent into the required benchmark solution JSON.
Call the `submit_formatter` tool with Python code defining:

    build_solution(raw_solution, data, instance) -> dict

`raw_solution` contains objective_value and a flat `variables` mapping from exact
Gurobi VarName strings to numeric values. `data` is the normalized OptiMUS data,
and `instance` is the original benchmark instance.
Use variable names, indices, the formulation context, and the target template to
construct the most complete candidate possible. Do not run a solver, read files,
use subprocesses, or invent decision values absent from the incumbent. Preserve
the exact top-level keys requested by the target template.

Problem:
{problem}

Target solution template:
{schema}

Formulation context:
{formulation}

Normalized data structure:
{data_summary}

Original instance structure:
{instance_summary}

Available incumbent variable families and samples:
{families}

"""

FORMATTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_formatter",
            "description": (
                "Execute candidate-only conversion code and validate its result "
                "against the required solution schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Python code defining build_solution(raw_solution, data, instance)"
                        ),
                    }
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    }
]


class ResultFormatter(Agent):
    def __init__(self, client: openai.Client, **kwargs):
        super().__init__(
            name="ResultFormatter",
            description=(
                "Converts a saved Gurobi incumbent into the required target solution.json. "
                "Call this agent whenever solution_export_status is pending."
            ),
            client=client,
            **kwargs,
        )
        self.max_tool_rounds = 10

    def generate_reply(self, task: str, state: dict, sender: Agent) -> tuple[str, dict]:
        raw_path = state.get("raw_solution_path")
        schema_path = state.get("target_solution_schema_path")
        if not raw_path or not os.path.isfile(raw_path):
            state["solution_export_status"] = "failed"
            return "No saved incumbent is available to format.", state
        if not schema_path or not os.path.isfile(schema_path):
            state["solution_export_status"] = "failed"
            return (
                "No target solution schema is available to format the incumbent.",
                state,
            )

        with open(raw_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        with open(schema_path, encoding="utf-8") as handle:
            schema = json.load(handle)
        with open(state["data_json_path"], encoding="utf-8") as handle:
            data = json.load(handle)
        instance = data
        instance_path = state.get("instance_path")
        if instance_path and os.path.isfile(instance_path):
            with open(instance_path, encoding="utf-8") as handle:
                instance = json.load(handle)
        problem = ""
        problem_path = state.get("problem_path")
        if problem_path and os.path.isfile(problem_path):
            with open(problem_path, encoding="utf-8") as handle:
                problem = handle.read()

        prompt = FORMATTER_PROMPT.format(
            problem=problem,
            schema=json.dumps(schema, ensure_ascii=False, indent=2),
            formulation=self._formulation_context(state),
            data_summary=json.dumps(
                self._structure_summary(data), ensure_ascii=False, indent=2
            ),
            instance_summary=json.dumps(
                self._structure_summary(instance), ensure_ascii=False, indent=2
            ),
            families=json.dumps(
                self._variable_families(raw), ensure_ascii=False, indent=2
            ),
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        last_error = "ResultFormatter did not submit valid conversion code."
        for tool_round in range(1, self.max_tool_rounds + 1):
            calls = []
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm,
                    messages=messages,
                    tools=FORMATTER_TOOLS,
                    tool_choice="auto",
                )
                message = completion.choices[0].message
                calls = message.tool_calls or []
                if not calls:
                    last_error = "ValueError: call submit_formatter with complete conversion code"
                    messages.extend(
                        [
                            {"role": "assistant", "content": message.content or ""},
                            {"role": "user", "content": last_error},
                        ]
                    )
                    continue
                call = calls[0]
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                        ],
                    }
                )
                if call.function.name != "submit_formatter":
                    raise ValueError(f"unknown tool: {call.function.name}")
                arguments = json.loads(call.function.arguments)
                source = arguments["source"]
                namespace = {"json": json, "re": re}
                exec(
                    compile(source, "<solution_formatter>", "exec"),
                    namespace,
                    namespace,
                )
                builder = namespace.get("build_solution")
                if not callable(builder):
                    raise TypeError(
                        "code did not define build_solution(raw_solution, data, instance)"
                    )
                solution = builder(raw, data, instance)
                self._validate_template(solution, schema)
                output_path = os.path.join(state["log_folder"], "solution.json")
                self._atomic_write_json(output_path, solution)
                formatter_path = os.path.join(
                    state["log_folder"], "solution_formatter.py"
                )
                with open(formatter_path, "w", encoding="utf-8") as handle:
                    handle.write(source)
                state["solution_path"] = output_path
                state["solution_export_status"] = "completed"
                state["solution_export_tool_rounds"] = tool_round
                return "Candidate incumbent was converted to solution.json.", state
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                feedback = json.dumps(
                    {
                        "ok": False,
                        "error": last_error,
                        "instruction": "Correct the code and call submit_formatter again.",
                    },
                    ensure_ascii=False,
                )
                messages.append(
                    (
                        {
                            "role": "tool",
                            "tool_call_id": calls[0].id,
                            "content": feedback,
                        }
                        if calls
                        else {"role": "user", "content": feedback}
                    )
                )

        state["solution_export_status"] = "failed"
        state["solution_export_tool_rounds"] = self.max_tool_rounds
        state["solution_export_error"] = last_error
        return (
            "The raw incumbent was preserved, but solution.json conversion failed.",
            state,
        )

    @staticmethod
    def _formulation_context(state):
        context = {
            "objective": state.get("objective"),
            "parameters": state.get("parameters"),
            "variables": state.get("variables"),
            "constraints": state.get("constraint"),
        }
        text = json.dumps(context, ensure_ascii=False)
        return text[:60000]

    @staticmethod
    def _variable_families(raw):
        families = defaultdict(list)
        for name, value in raw.get("variables", {}).items():
            family = name.split("[", 1)[0]
            if len(families[family]) < 8:
                families[family].append({"name": name, "value": value})
        counts = defaultdict(int)
        for name in raw.get("variables", {}):
            counts[name.split("[", 1)[0]] += 1
        return {
            name: {"count": counts[name], "samples": samples}
            for name, samples in families.items()
        }

    @classmethod
    def _structure_summary(cls, value, depth=0):
        if depth >= 3:
            return {"type": type(value).__name__}
        if isinstance(value, dict):
            keys = list(value)[:40]
            return {
                "type": "object",
                "key_count": len(value),
                "fields": {
                    str(key): cls._structure_summary(value[key], depth + 1)
                    for key in keys
                },
            }
        if isinstance(value, list):
            return {
                "type": "array",
                "length": len(value),
                "samples": [
                    cls._structure_summary(item, depth + 1) for item in value[:2]
                ],
            }
        if isinstance(value, str):
            return {"type": "string", "sample": value[:120]}
        return {"type": type(value).__name__, "sample": value}

    @classmethod
    def _validate_template(cls, value, template, path="$"):
        if isinstance(template, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be an object")
            missing = [key for key in template if key not in value]
            if missing:
                raise ValueError(f"{path} is missing required keys: {missing}")
            for key, child in template.items():
                cls._validate_template(value[key], child, f"{path}.{key}")
        elif isinstance(template, list):
            if not isinstance(value, list):
                raise ValueError(f"{path} must be an array")
            if template:
                for index, child in enumerate(value):
                    cls._validate_template(child, template[0], f"{path}[{index}]")
        elif isinstance(template, str) and template.startswith("<"):
            expected = template[1:].split(">", 1)[0].lower()
            if expected.startswith("list") and not isinstance(value, list):
                raise ValueError(f"{path} must be an array")
            if expected.startswith("dict") and not isinstance(value, dict):
                raise ValueError(f"{path} must be an object")
            if expected.startswith("float") and not isinstance(value, (int, float)):
                raise ValueError(f"{path} must be numeric")
            if expected.startswith("int") and not isinstance(value, int):
                raise ValueError(f"{path} must be an integer")
            if expected.startswith("str") and not isinstance(value, str):
                raise ValueError(f"{path} must be a string")

    @staticmethod
    def _atomic_write_json(path, payload):
        directory = os.path.dirname(path)
        fd, temporary = tempfile.mkstemp(
            prefix=".solution-", suffix=".json", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
