"""Tools the agent can call.

Every tool declares its arguments as a Pydantic model. The registry turns that
model into the JSON schema the LLM sees, and validates the LLM's arguments before
running anything; a validation error is returned to the model as a tool result so
it can correct itself instead of crashing the turn.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from ..model import CLASSES, DISPLAY, RadiomicsModel
from .memory import Store
from .skills import Skill


@dataclass
class ToolContext:
    model: RadiomicsModel
    store: Store
    skills: dict[str, Skill]


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[[ToolContext, Any], Any]

    def schema(self) -> dict:
        s = self.args.model_json_schema()
        s.pop("title", None)
        return {"name": self.name, "description": self.description, "parameters": s}


class ToolError(Exception):
    """An expected failure the model should see and recover from."""


# --- argument models --------------------------------------------------------
class CaseArg(BaseModel):
    case_id: str = Field(description="Id of an uploaded case, e.g. '3f9a1c2b'.")


class ExplainArgs(CaseArg):
    top_k: int = Field(5, ge=1, le=15, description="How many features to return.")


class SimilarArgs(CaseArg):
    k: int = Field(5, ge=1, le=10, description="How many neighbours to return.")


class RecallArgs(BaseModel):
    label: Literal["glioma", "meningioma", "pituitary", "notumor"] | None = Field(
        None, description="Only cases predicted as this class.")
    limit: int = Field(10, ge=1, le=50)


class NoteArgs(CaseArg):
    note: str = Field(max_length=500, description="Short note to remember about this case.")


class SkillArgs(BaseModel):
    name: str = Field(description="Skill name from the catalog in the system prompt.")


class NoArgs(BaseModel):
    pass


# --- implementations --------------------------------------------------------
def _case(ctx: ToolContext, case_id: str) -> dict:
    case = ctx.store.get_case(case_id)
    if case is None:
        raise ToolError(f"case '{case_id}' not found. Use recall_cases to list known cases.")
    return case


def predict(ctx: ToolContext, a: CaseArg) -> dict:
    case = _case(ctx, a.case_id)
    return {"case_id": a.case_id, **ctx.model.predict(case["features"]).as_dict()}


def explain_prediction(ctx: ToolContext, a: ExplainArgs) -> dict:
    case = _case(ctx, a.case_id)
    pred = ctx.model.predict(case["features"])
    return {"case_id": a.case_id, "predicted": pred.label,
            "method": "features ranked by global permutation importance x deviation "
                      "from the average healthy-tissue profile (in training std units)",
            "drivers": ctx.model.explain(case["features"], a.top_k)}


def find_similar_cases(ctx: ToolContext, a: SimilarArgs) -> dict:
    case = _case(ctx, a.case_id)
    neighbours = ctx.model.similar(case["features"], a.k)
    votes: dict[str, int] = {}
    for n in neighbours:
        votes[n["label"]] = votes.get(n["label"], 0) + 1
    return {"case_id": a.case_id, "neighbours": neighbours, "label_votes": votes}


def recall_cases(ctx: ToolContext, a: RecallArgs) -> dict:
    cases = ctx.store.list_cases(a.label, a.limit)
    return {"count": len(cases), "cases": [
        {"case_id": c["case_id"], "filename": c["filename"], "created_at": c["created_at"],
         "predicted": c["prediction"]["label"], "note": c["note"]} for c in cases]}


def annotate_case(ctx: ToolContext, a: NoteArgs) -> dict:
    _case(ctx, a.case_id)
    ctx.store.annotate_case(a.case_id, a.note)
    return {"case_id": a.case_id, "saved": True}


def model_card(ctx: ToolContext, a: NoArgs) -> dict:
    return {"model": "HistGradientBoostingClassifier on 46 GLCM + first-order features",
            "classes": {c: DISPLAY[c] for c in CLASSES},
            "metrics": ctx.model.metrics,
            "top_features": [f for f, _ in ctx.model.top_features(5)],
            "limitations": ["features computed on the whole 2D slice, not a segmented lesion",
                            "no patient-level split in the public dataset",
                            "academic demo, not a medical device"]}


def load_skill(ctx: ToolContext, a: SkillArgs) -> dict:
    skill = ctx.skills.get(a.name)
    if skill is None:
        raise ToolError(f"unknown skill '{a.name}'. Available: {sorted(ctx.skills)}")
    return {"name": skill.name, "instructions": skill.body}


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("predict", "Classify an uploaded MRI case into glioma, meningioma, pituitary or "
         "notumor. Returns class probabilities.", CaseArg, predict),
    Tool("explain_prediction", "Explain which texture features drove the prediction for a case.",
         ExplainArgs, explain_prediction),
    Tool("find_similar_cases", "Retrieve the most similar training images in feature space, "
         "with their true labels.", SimilarArgs, find_similar_cases),
    Tool("recall_cases", "List previously analyzed cases from long-term memory.",
         RecallArgs, recall_cases),
    Tool("annotate_case", "Save a short note on a case in long-term memory.", NoteArgs, annotate_case),
    Tool("model_card", "Model description, test metrics and limitations.", NoArgs, model_card),
    Tool("load_skill", "Load the full instructions of a skill before doing that task.",
         SkillArgs, load_skill),
]}


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    data: Any = field(default=None, repr=False)


def run_tool(ctx: ToolContext, name: str, arguments: dict) -> ToolResult:
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(f"error: unknown tool '{name}'", True)
    try:
        args = tool.args.model_validate(arguments or {})
    except ValidationError as e:
        return ToolResult(f"error: invalid arguments for {name}: {e.errors(include_url=False)}", True)
    try:
        data = tool.fn(ctx, args)
    except ToolError as e:
        return ToolResult(f"error: {e}", True)
    except Exception as e:  # unexpected: surface to the model, never crash the loop
        return ToolResult(f"error: {name} failed ({type(e).__name__}: {e})", True)
    return ToolResult(json.dumps(data, ensure_ascii=False), False, data)
