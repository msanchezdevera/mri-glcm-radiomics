import json

from mri_radiomics.agent.skills import load_skills
from mri_radiomics.agent.tools import TOOLS, ToolContext, run_tool


def ctx(model, store):
    return ToolContext(model, store, load_skills())


def test_schemas_are_valid_json_schema():
    for tool in TOOLS.values():
        s = tool.schema()
        assert s["parameters"]["type"] == "object" and s["description"]


def test_predict_tool(model, store, case_ids):
    r = run_tool(ctx(model, store), "predict", {"case_id": case_ids["glioma"]})
    assert not r.is_error and json.loads(r.content)["label"] == "glioma"


def test_validation_error_is_returned_not_raised(model, store):
    r = run_tool(ctx(model, store), "explain_prediction", {"case_id": "x", "top_k": 99})
    assert r.is_error and "invalid arguments" in r.content


def test_unknown_case_and_tool(model, store):
    assert "not found" in run_tool(ctx(model, store), "predict", {"case_id": "deadbeef"}).content
    assert run_tool(ctx(model, store), "rm_rf", {}).is_error


def test_skills_load(model, store):
    r = run_tool(ctx(model, store), "load_skill", {"name": "case-report"})
    assert "Neighbour agreement" in json.loads(r.content)["instructions"]
    assert run_tool(ctx(model, store), "load_skill", {"name": "nope"}).is_error


def test_long_term_memory(model, store, case_ids):
    c = ctx(model, store)
    run_tool(c, "annotate_case", {"case_id": case_ids["notumor"], "note": "control"})
    cases = json.loads(run_tool(c, "recall_cases", {"label": "notumor"}).content)["cases"]
    assert cases[0]["note"] == "control"
