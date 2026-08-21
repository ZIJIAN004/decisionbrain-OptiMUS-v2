"""Exercise validate.py against synthetic conversions.

Each case pins one clause of the input contract, next to the line in OptiMUS that
enforces it. Run it directly:

    python -m adapters.frontieror.test_validate

The cases that matter most are the ones OptiMUS's own sanity_check accepts and
the agents then reject -- index suffixes and undeclared dimensions. Those two
cost a real run before they were checked here.
"""
import json, sys, shutil, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.frontieror import validate
from adapters.frontieror.validate import ValidationError

def stage(data, params, targets=True):
    d = Path(tempfile.mkdtemp())
    (d / "data.json").write_text(json.dumps(data), encoding="utf-8")
    (d / "parameters.json").write_text(json.dumps(params), encoding="utf-8")
    if targets:
        (d / "input_targets.json").write_text("{}", encoding="utf-8")
    return d

def run(name, instance, data, params, expect):
    d = stage(data, params)
    try:
        rep, cov = validate.validate(d, instance, lambda: None)
        got = "PASS"
        detail = cov
    except ValidationError as e:
        got = "REJECT"
        detail = str(e).split("\n")[0][:110]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    ok = "OK " if got == expect else "FAIL"
    print(f"[{ok}] {name:<34} -> {got:<6} {detail}")
    return got == expect

results = []

# 1. good: real shapes, dim names present in data.json
inst = {"n": 2, "m": 3, "cost": [[1,2,3],[4,5,6]], "cap": 7}
data = {"n": 2, "m": 3, "cost": [[1,2,3],[4,5,6]], "cap": 7}
params = [
    {"symbol":"n","shape":[],"definition":"rows","source_key":"n"},
    {"symbol":"m","shape":[],"definition":"cols","source_key":"m"},
    {"symbol":"cost","shape":["n","m"],"definition":"unit cost","source_key":"cost"},
    {"symbol":"cap","shape":[],"definition":"capacity","source_key":"cap"},
]
results.append(run("well-formed shaped encoding", inst, data, params, "PASS"))

# 2. degenerate shape:[] on a real array -> must be rejected by OUR assertion
bad = [dict(p) for p in params]
bad[2]["shape"] = []
results.append(run("array encoded as shape:[]", inst, data, bad, "REJECT"))

# 3. dropped key
results.append(run("instance key not covered", inst, data, params[:3], "REJECT"))

# 4. ragged without mask
inst4 = {"n": 2, "routes": [[1,2,3],[4,5]]}
data4 = {"n": 2, "routes": [[1,2,3],[4,5,0]], "R": 3}
p4 = [
    {"symbol":"n","shape":[],"definition":"n","source_key":"n"},
    {"symbol":"routes","shape":["n","R"],"definition":"routes","source_key":"routes"},
]
results.append(run("ragged padded without mask", inst4, data4, p4, "REJECT"))
# note: p4 also leaves dim R undeclared, which the symbol contract rejects too

# 5. ragged with mask -> passes (note: R must also be a data.json key)
p5 = p4 + [{"symbol":"routesMask","shape":["n","R"],"definition":"1 = real cell",
            "source_key":"routes","role":"mask"},
           {"symbol":"R","shape":[],"definition":"padded route length","source_key":"routes"}]
data5 = dict(data4); data5["routesMask"] = [[1,1,1],[1,1,0]]
results.append(run("ragged padded with mask", inst4, data5, p5, "PASS"))

# 6. dimension mismatch -> caught by OptiMUS's own sanity_check
data6 = {"n": 5, "m": 3, "cost": [[1,2,3],[4,5,6]], "cap": 7}
results.append(run("wrong dim value (OptiMUS check)", inst, data6, params, "REJECT"))

# 7. symbol with two underscores -> OptiMUS camelCase rule
p7 = [dict(p) for p in params]; p7[2]["symbol"] = "unit_cost_matrix"
data7 = dict(data); data7["unit"] = [[1,2,3],[4,5,6]]
inst7 = dict(inst)
results.append(run("symbol with 2 underscores", inst7, data7, p7, "REJECT"))

# --- symbol contract (the half sanity_check does not enforce) -----------------

# 8. index suffix on a symbol: legal for sanity_check (one underscore), fatal at
#    agents/formulator.py:414 because 	extup{cost} != "cost_ij"
p8 = [dict(x) for x in params]; p8[2]["symbol"] = "cost_ij"
results.append(run("symbol with index suffix", inst, data, p8, "REJECT"))

# 9. dimension present in data.json but not declared as a parameter: legal for
#    sanity_check (utils/misc.py:222), NameError at agents/evaluator.py:105
p9 = [x for x in params if x["symbol"] != "m"]
p9 = p9 + [{"symbol":"numCols","shape":[],"definition":"cols","source_key":"m"}]
results.append(run("dim not declared as parameter", inst, data, p9, "REJECT"))

# 10. dimension declared, but as an array rather than a scalar
p10 = [dict(x) for x in params]; p10[1]["shape"] = ["n"]
data10 = dict(data); data10["m"] = [3, 3]
inst10 = dict(inst); inst10["m"] = [3, 3]
results.append(run("dim declared with a shape", inst10, data10, p10, "REJECT"))

# 11. symbol shadowing a name the harness binds (evaluator.py prep_code)
inst11 = {"n": 2, "model": 4}
data11 = {"n": 2, "model": 4}
p11 = [{"symbol":"n","shape":[],"definition":"n","source_key":"n"},
       {"symbol":"model","shape":[],"definition":"clashes with gp.Model","source_key":"model"}]
results.append(run("symbol shadows exec namespace", inst11, data11, p11, "REJECT"))

# --- metadata role ------------------------------------------------------------

inst12 = {"n": 2, "m": 3, "cost": [[1,2,3],[4,5,6]], "cap": 7, "instance_id": "delage2022_3"}
p12 = params + [{"symbol":"instanceId","shape":[],"definition":"provenance only",
                 "source_key":"instance_id","role":"metadata"}]
# 12. a free-text key declared metadata: no data.json entry, no char-code array
results.append(run("metadata key, absent from data.json", inst12, data, p12, "PASS"))

# 13. metadata is still not an escape hatch for dropping a key entirely
results.append(run("metadata key left uncovered", inst12, data, params, "REJECT"))

# 14. metadata may not carry a shape
p14 = [dict(x) for x in p12]; p14[-1]["shape"] = ["n"]
results.append(run("metadata declaring a shape", inst12, data, p14, "REJECT"))

# 15. a ragged key declared metadata needs no mask (it never reaches data.json)
inst15 = {"n": 2, "generation_info": [[1,2],[3]]}
data15 = {"n": 2}
p15 = [{"symbol":"n","shape":[],"definition":"n","source_key":"n"},
       {"symbol":"generationInfo","shape":[],"definition":"provenance only",
        "source_key":"generation_info","role":"metadata"}]
results.append(run("ragged metadata needs no mask", inst15, data15, p15, "PASS"))

# 16. a Python keyword as a symbol -> SyntaxError when evaluator.py execs it
inst16 = {"n": 2, "class": 4}
p16 = [{"symbol":"n","shape":[],"definition":"n","source_key":"n"},
       {"symbol":"class","shape":[],"definition":"kw","source_key":"class"}]
results.append(run("symbol is a Python keyword", inst16, dict(inst16), p16, "REJECT"))

# 17. a symbol shadowing a builtin the generated code calls
inst17 = {"n": 2, "sum": 4}
p17 = [{"symbol":"n","shape":[],"definition":"n","source_key":"n"},
       {"symbol":"sum","shape":[],"definition":"shadows sum()","source_key":"sum"}]
results.append(run("symbol shadows a builtin", inst17, dict(inst17), p17, "REJECT"))

# 18. a parameter with no symbol at all -> feedback, not a raw KeyError
inst18 = {"n": 2}
p18 = [{"shape":[],"definition":"nameless","source_key":"n"}]
results.append(run("parameter without a symbol", inst18, dict(inst18), p18, "REJECT"))

print("\n%d/%d as expected" % (sum(results), len(results)))
sys.exit(0 if all(results) else 1)
