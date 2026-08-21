"""Instructions given to the converter model.

The rules below are the same ones validate.py enforces mechanically. Stating
them here does not make them true -- every answer is checked -- but a converter
that knows the rules needs fewer rounds.

The constraint-extraction rules in TARGETS_RULES are OptiMUS's own, taken from
utils/target_extractor.py (prompt_templates[0]), so the way a problem statement
is decomposed stays the baseline authors' decision rather than ours.
"""

SYSTEM = """You convert an optimization problem into the two inputs OptiMUS reads: \
a structured description of the model and a numeric data file.

You never see the instance data directly. It may be gigabytes. To inspect it you \
write Python and it is executed for you against the parsed instance.

Work in two phases.

PHASE 1 -- INSPECT. Reply with exactly one block:

```explore
# `data` is the parsed instance.json. Print whatever you need.
print(sorted(data.keys()))
```

Only its stdout comes back to you (truncated to the last 20000 characters), so \
print summaries, not raw data. You may explore over several turns.

PHASE 2 -- ANSWER. When you understand the instance, reply with exactly three \
blocks and no other text:

```transform.py
...
```
```parameters.json
...
```
```targets.json
...
```

Never mix the two phases in one reply."""


TRANSFORM_RULES = """`transform.py` is a standalone script:

    python transform.py --instance <path to instance.json> --out <path to data.json>

It reads the instance, restructures it, and writes data.json. Requirements:

- Plain `json.load` is fine. The largest instance in this benchmark parses in
  about 15 seconds using 9.7 GB, and you are given a much larger cap.
- Deterministic. Iterate dicts in sorted key order; never iterate a set; never
  read the clock or a random seed. The same instance must produce a
  byte-identical data.json every run -- this is checked by running it twice.
- Lossless. Every top-level key of the instance must survive into data.json in
  some form. You may rename, flatten a dict of records into arrays, or split one
  key into several. You may NOT drop a key you judge irrelevant, merge two keys,
  or pre-aggregate values. Deciding what matters is the modelling step, and that
  belongs to the solver team, not to you.
- Every dimension you name must exist in data.json as its own integer entry. If
  you declare a parameter with shape ["n", "m"], then data["n"] and data["m"]
  must be integers equal to the actual array dimensions.
- Ragged arrays must be padded to a rectangle with 0, and each padded key must
  also produce a mask array of the same shape: 1 where the cell is real, 0 where
  it is padding. Pad with 0 and nothing else. Do not pad with a value chosen to
  make constraints fall away on their own (+inf, -1, a large number) -- that
  decides how the model should treat padding, which is not your decision. The
  mask states the facts and lets the solver team decide."""


PARAMETERS_RULES = """`parameters.json` is a JSON list. One entry per quantity:

    {
      "symbol":      "linearCosts",
      "shape":       ["n", "m"],
      "definition":  "unit shipping cost from supplier i to customer j",
      "source_key":  "linear_costs"
    }

- `symbol` must be camelCase and contain at most one underscore. Its part before
  the underscore must be a top-level key of data.json, spelled identically. This
  is how the generated code reaches the data, so `flowCost_ij` reads
  data["flowCost"].
- `shape` lists dimension names, or [] for a genuine scalar. A value that is a
  list or a dict in the instance is NOT a scalar and must carry its real
  dimensions. Encoding structure away as [] is rejected.
- `definition` is the only thing the modelling agents ever see about this
  quantity -- they never see a single number. Say what it means and how it is
  laid out, in one sentence. "cost matrix" is not enough; "unit shipping cost
  from supplier i to customer j, indexed by supplier then customer" is.
- `source_key` is the top-level key of the original instance this came from.
  Every top-level key must appear as some entry's `source_key`.
- A mask array adds `"role": "mask"` and keeps the `source_key` of the array it
  masks."""


TARGETS_RULES = """`targets.json` describes the model in words:

    {"background": "...", "constraints": ["...", "..."], "objective": "..."}

Follow these rules exactly:

1. Define the background and context of the problem.
2. List all constraints, including implicit ones such as non-negativity.
3. State the single primary objective.
4. Preferences are not constraints. Do not include them.
5. Statements that simply define exact values of parameters are not constraints
   (e.g. "The cost of producing an X is Y", "Each X has a size of Y").
6. Statements that define bounds are constraints (e.g. "The cost of producing an
   X is at most Y", "Each X has a size of at least Y").
7. Keep each constraint separate and explicit. Do not merge different
   constraints into a single entry.
8. Refer to quantities by the `symbol` you gave them in parameters.json.

Do not invent a constraint that the problem statement does not state, and do not
leave one out because it looks hard to model."""


def initial(problem_md: str) -> str:
    return f"""Here is the problem statement.

-----
{problem_md}
-----

{TRANSFORM_RULES}

{PARAMETERS_RULES}

{TARGETS_RULES}

Begin with PHASE 1: inspect the instance."""


def rejection(feedback: str) -> str:
    return f"""Your conversion was rejected by the mechanical checker:

-----
{feedback}
-----

Fix it. You may inspect the instance again first if you need to. Then reply with
the three blocks."""
