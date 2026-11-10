"""LLM providers behind one small interface.

Messages use a provider-neutral format:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Each provider converts to and from its own wire format. Calls go over plain HTTP
(httpx) so there is no SDK lock-in and the request shape is visible.

MockLLM is a deterministic, rule-based stand-in. It lets the whole harness, the
API, the tests and the eval suite run in CI without an API key or network access.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


def _post(url: str, headers: dict, body: dict, retries: int = 3) -> dict:
    """POST with retry and exponential backoff on rate limits and 5xx."""
    for attempt in range(retries):
        r = httpx.post(url, headers=headers, json=body, timeout=60)
        if r.status_code in (429, 500, 502, 503, 529) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")


# --- Anthropic --------------------------------------------------------------
class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-5")
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def complete(self, system, messages, tools):
        body = {"model": self.model, "max_tokens": 1500, "system": system,
                "messages": self._to_wire(messages),
                "tools": [{"name": t["name"], "description": t["description"],
                           "input_schema": t["parameters"]} for t in tools]}
        data = _post("https://api.anthropic.com/v1/messages",
                     {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, body)
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        calls = [ToolCall(b["id"], b["name"], b["input"])
                 for b in data["content"] if b["type"] == "tool_use"]
        return LLMResponse(text, calls, data.get("usage", {}))

    @staticmethod
    def _to_wire(messages):
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                           for c in m.get("tool_calls", [])]
                out.append({"role": "assistant", "content": blocks})
            else:  # tool results are user-side blocks; group consecutive ones
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out


# --- OpenAI -----------------------------------------------------------------
class OpenAILLM:
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]

    def complete(self, system, messages, tools):
        wire = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"],
                                                       "arguments": json.dumps(c["arguments"])}}
                                         for c in m["tool_calls"]]
                wire.append(msg)
            elif m["role"] == "tool":
                wire.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                wire.append(m)
        body = {"model": self.model, "messages": wire,
                "tools": [{"type": "function", "function": t} for t in tools]}
        data = _post("https://api.openai.com/v1/chat/completions",
                     {"Authorization": f"Bearer {self.api_key}"}, body)
        msg = data["choices"][0]["message"]
        calls = [ToolCall(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
                 for c in msg.get("tool_calls") or []]
        return LLMResponse(msg.get("content") or "", calls, data.get("usage", {}))


# --- Mock -------------------------------------------------------------------
CASE_RE = re.compile(r"\b[0-9a-f]{8}\b")
DIAGNOSIS_RE = re.compile(r"\b(diagnos\w*|treatment|treat|surgery|prognosis|should i worry|"
                          r"tratamiento|cirug[ií]a|tengo c[aá]ncer|am i sick)\b", re.I)


class MockLLM:
    """Keyword planner + templated writer. Issues its tool calls in one parallel
    batch, then writes the answer from the tool results."""

    name = "mock"

    def complete(self, system, messages, tools):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        turn = messages[last_user:]
        results = {m["name"]: m["content"] for m in turn if m["role"] == "tool"}
        if results:
            return LLMResponse(self._write(turn[0]["content"], results))
        plan = self._plan(turn[0]["content"])
        if not plan:
            return LLMResponse(self._write(turn[0]["content"], {}))
        return LLMResponse("", [ToolCall(f"call_{i}", n, a) for i, (n, a) in enumerate(plan)])

    def _plan(self, text: str) -> list[tuple[str, dict]]:
        t = text.lower()
        ids = CASE_RE.findall(text)
        case = {"case_id": ids[-1]} if ids else None
        if DIAGNOSIS_RE.search(t):
            return [("predict", case)] if case else []
        if re.search(r"\b(previous|history|earlier|past|recall|anteriores|historial)\b", t) or \
                ("cases" in t and not case):
            label = next((c for c in ("glioma", "meningioma", "pituitary", "notumor") if c in t), None)
            return [("recall_cases", {"label": label} if label else {})]
        if re.search(r"\b(accuracy|metrics|how good|how accurate|limitations|model card|precisi[oó]n)\b", t):
            return [("model_card", {})]
        if not case:
            return []
        if re.search(r"\b(note|remember|recorda|anot\w*)\b", t):
            note = text.split(":", 1)[1].strip() if ":" in text else text
            return [("annotate_case", {**case, "note": note[:500]})]
        if re.search(r"\b(report|informe|summary|resumen)\b", t):
            return [("load_skill", {"name": "case-report"}), ("predict", case),
                    ("explain_prediction", case), ("find_similar_cases", case)]
        plan = [("predict", case)]
        if re.search(r"\b(why|explain|drivers|por qu[eé]|explica\w*)\b", t):
            plan.append(("explain_prediction", case))
        if re.search(r"\b(similar|neighbou?rs|parecid\w*)\b", t):
            plan.append(("find_similar_cases", case))
        return plan

    def _write(self, question: str, results: dict[str, str]) -> str:
        parts: list[str] = []
        data = {k: (json.loads(v) if not v.startswith("error") else v) for k, v in results.items()}
        errors = [v for v in data.values() if isinstance(v, str)]
        if errors:
            parts.append(f"I couldn't complete that: {errors[0][len('error: '):]}")
        if DIAGNOSIS_RE.search(question):
            parts.append("I can't give a medical diagnosis or treatment advice. This is a research "
                         "demo trained on public 2D slices; please talk to a radiologist or physician.")
        p = data.get("predict")
        if isinstance(p, dict):
            pct = round(p["probabilities"][p["label"]] * 100, 1)
            parts.append(f"The model classifies case {p['case_id']} as {p['display']} ({pct}% probability).")
        e = data.get("explain_prediction")
        if isinstance(e, dict):
            drivers = ", ".join(f"{d['feature']} ({d['deviation_from_healthy_std']:+.1f} std vs healthy)"
                                for d in e["drivers"][:3])
            parts.append(f"Main drivers: {drivers}.")
        s = data.get("find_similar_cases")
        if isinstance(s, dict):
            votes = ", ".join(f"{k}: {v}" for k, v in s["label_votes"].items())
            parts.append(f"Its {len(s['neighbours'])} nearest training images are labelled {votes}.")
        r = data.get("recall_cases")
        if isinstance(r, dict):
            listed = "; ".join(f"{c['case_id']} ({c['predicted']})" for c in r["cases"][:5]) or "none yet"
            parts.append(f"I have {r['count']} matching case(s) in memory: {listed}.")
        m = data.get("model_card")
        if isinstance(m, dict):
            mt = m["metrics"]
            parts.append(f"On the {mt['test_size']}-image test set the model reaches "
                         f"{mt['test_accuracy']:.1%} accuracy and {mt['test_f1_macro']:.3f} macro F1. "
                         f"Limitations: {'; '.join(m['limitations'])}.")
        a = data.get("annotate_case")
        if isinstance(a, dict):
            parts.append(f"Saved the note on case {a['case_id']}.")
        if not parts:
            parts.append("Upload an MRI slice (POST /cases) and give me its case id, or ask about "
                         "the model, its metrics or previous cases.")
        return " ".join(parts)


def get_llm(provider: str | None = None) -> LLM:
    provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
    return {"anthropic": AnthropicLLM, "openai": OpenAILLM, "mock": MockLLM}[provider]()
