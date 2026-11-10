from mri_radiomics.agent.harness import Agent
from mri_radiomics.agent.llm import LLMResponse, MockLLM, ToolCall
from mri_radiomics.agent.memory import window


class LoopingLLM:
    name = "loop"

    def complete(self, system, messages, tools):
        return LLMResponse("", [ToolCall("c", "model_card", {})])


class BadArgsThenRecoverLLM:
    """First call sends invalid args; after seeing the error it fixes them."""
    name = "recover"

    def complete(self, system, messages, tools):
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if not tool_msgs:
            return LLMResponse("", [ToolCall("a", "predict", {"wrong": 1})])
        if tool_msgs[-1]["content"].startswith("error") and len(tool_msgs) == 1:
            case = messages[0]["content"].split()[-1]
            return LLMResponse("", [ToolCall("b", "predict", {"case_id": case})])
        return LLMResponse("done")


def test_step_budget(model, store):
    r = Agent(LoopingLLM(), store, model, max_steps=3).run("s", "hi")
    assert r.stopped == "max_steps" and r.tool_calls == ["model_card"] * 3


def test_recovers_from_tool_error(model, store, case_ids):
    r = Agent(BadArgsThenRecoverLLM(), store, model).run("s", f"classify {case_ids['glioma']}")
    tool_steps = [s for s in r.steps if s.kind == "tool"]
    assert [s.is_error for s in tool_steps] == [True, False] and r.reply == "done"


def test_session_memory_persists(model, store, case_ids):
    agent = Agent(MockLLM(), store, model)
    agent.run("s1", f"classify {case_ids['glioma']}")
    agent.run("s1", "hello")
    history = store.history("s1")
    assert sum(m["role"] == "user" for m in history) == 2
    assert store.history("other") == []


def test_guardrail_adds_disclaimer(model, store, case_ids):
    r = Agent(MockLLM(), store, model).run("s", f"classify {case_ids['meningioma']}")
    assert "not a medical diagnosis" in r.reply


def test_window_never_splits_tool_pairs():
    hist = []
    for i in range(10):
        hist += [{"role": "user", "content": str(i)},
                 {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
                 {"role": "tool", "tool_call_id": "x", "content": "{}"},
                 {"role": "assistant", "content": "ok", "tool_calls": []}]
    w = window(hist, 7)
    assert w[0]["role"] == "user" and len(w) <= 7
