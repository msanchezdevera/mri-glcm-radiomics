"""The agent loop.

  user message -> LLM -> (tool calls -> tools -> results -> LLM)* -> answer

Responsibilities kept in the harness rather than the prompt:
  - a hard step budget, so a confused model can't loop forever
  - tool argument validation and error recovery (errors go back to the model)
  - parallel tool calls executed in order and paired with their ids
  - session memory: history loaded, windowed, and persisted after the turn
  - an output guardrail on medical claims
  - a structured trace of every step (latency, tool, args, errors, token usage)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from ..model import RadiomicsModel, get_model
from .llm import LLM, get_llm
from .memory import Store, window
from .skills import catalog, load_skills
from .tools import TOOLS, ToolContext, run_tool

SYSTEM_PROMPT = """You are a radiomics research assistant for a brain MRI texture model.
You help users analyze uploaded MRI slices with the tools provided.

Rules:
- Ground every number and class you mention in a tool result from this conversation. Never guess.
- Call predict before stating what a case is; call explain_prediction before saying why.
- You are not a medical device. Never give a diagnosis, prognosis or treatment advice; if asked,
  say so plainly and suggest a radiologist or physician. You may still report the model's output.
- When a task matches a skill below, call load_skill first and follow its instructions.
- If a tool returns an error, fix the arguments or explain the problem; don't retry blindly.
- Be concise.

Skills:
{skills}
"""

DISCLAIMER = "Note: research demo, not a medical diagnosis."
CLASS_WORDS = ("glioma", "meningioma", "pituitary")


@dataclass
class Step:
    kind: str  # "llm" | "tool"
    latency_ms: float
    name: str = ""
    arguments: dict = field(default_factory=dict)
    is_error: bool = False
    usage: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    reply: str
    steps: list[Step]
    stopped: str  # "answer" | "max_steps"

    @property
    def tool_calls(self) -> list[str]:
        return [s.name for s in self.steps if s.kind == "tool"]

    def trace(self) -> list[dict]:
        return [asdict(s) for s in self.steps]


class Agent:
    def __init__(self, llm: LLM | None = None, store: Store | None = None,
                 model: RadiomicsModel | None = None, max_steps: int = 6, history_window: int = 30):
        self.llm = llm or get_llm()
        self.store = store or Store()
        skills = load_skills()
        self.ctx = ToolContext(model or get_model(), self.store, skills)
        self.system = SYSTEM_PROMPT.format(skills=catalog(skills))
        self.tools = [t.schema() for t in TOOLS.values()]
        self.max_steps = max_steps
        self.history_window = history_window

    def run(self, session_id: str, user_message: str, case_id: str | None = None) -> AgentResult:
        content = f"{user_message}\n\n[active case: {case_id}]" if case_id else user_message
        new: list[dict] = [{"role": "user", "content": content}]
        history = window(self.store.history(session_id), self.history_window)
        steps: list[Step] = []
        reply, stopped = "", "max_steps"

        for _ in range(self.max_steps):
            t0 = time.perf_counter()
            resp = self.llm.complete(self.system, history + new, self.tools)
            steps.append(Step("llm", _ms(t0), usage=resp.usage))
            new.append({"role": "assistant", "content": resp.content,
                        "tool_calls": [asdict(c) for c in resp.tool_calls]})
            if not resp.tool_calls:
                reply, stopped = resp.content, "answer"
                break
            for call in resp.tool_calls:
                t0 = time.perf_counter()
                result = run_tool(self.ctx, call.name, call.arguments)
                steps.append(Step("tool", _ms(t0), call.name, call.arguments, result.is_error))
                new.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                            "content": result.content})
        else:
            reply = "I ran out of steps before finishing. Try a narrower question."
            new.append({"role": "assistant", "content": reply, "tool_calls": []})

        reply = self._guardrail(reply)
        new[-1]["content"] = reply
        self.store.append(session_id, new)
        result = AgentResult(reply, steps, stopped)
        self.store.save_trace(session_id, result.trace())
        return result

    @staticmethod
    def _guardrail(reply: str) -> str:
        """Any answer that names a tumor class carries the disclaimer."""
        if any(w in reply.lower() for w in CLASS_WORDS) and "not a medical" not in reply.lower():
            reply = f"{reply}\n\n{DISCLAIMER}"
        return reply


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
