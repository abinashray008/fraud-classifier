"""Exercise the investigation agent + Jev guardrail with fake models (no network)."""

from __future__ import annotations

from types import SimpleNamespace

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.feature_store import FeatureStore
from app.agent.guardrail import QUESTION_ID, JevToolGuard
from app.agent.investigator import _extract_tool_calls
from app.agent.tools import build_tools
from app.schemas.transaction import Verdict


class FakeJev:
    """Duck-typed TypeSafeClassifier: risky iff the queried card_id is not the one under review."""

    def __init__(self, allowed_card: str) -> None:
        self.allowed_card = allowed_card
        self.seen: list[dict] = []

    def _answer(self, state: dict):
        self.seen.append(state)
        args = state["tool_call"]["args"]
        risky = "card_id" in args and args["card_id"] != self.allowed_card
        return SimpleNamespace(nouls={QUESTION_ID: SimpleNamespace(noul=0.95 if risky else 0.02)})

    def invoke(self, state, config=None):
        return self._answer(state)

    async def ainvoke(self, state, config=None):
        return self._answer(state)


class ToolCapableFakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel with a no-op bind_tools so create_agent can use it."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        return self


def _fake_llm(*messages: AIMessage) -> GenericFakeChatModel:
    return ToolCapableFakeChatModel(messages=iter(messages))


async def test_guard_blocks_out_of_scope_lookup_and_agent_still_returns_verdict():
    store = FeatureStore()
    store.seed_demo()
    tools = build_tools(store)
    jev = FakeJev(allowed_card="card_good_001")
    guard = JevToolGuard(tools=tools, api_key="x", threshold=0.30, classifier=jev)  # type: ignore[arg-type]

    llm = _fake_llm(
        AIMessage(
            content="",
            tool_calls=[
                {"id": "t1", "name": "get_card_history", "args": {"card_id": "card_good_001"}},
                {"id": "t2", "name": "get_card_history", "args": {"card_id": "card_ato_002"}},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "t3",
                    "name": "Verdict",
                    "args": {
                        "fraud_prob": 0.15,
                        "pattern": "legitimate",
                        "rationale": "Known device and consistent history.",
                        "evidence": ["12 prior approved txns"],
                    },
                }
            ],
        ),
    )
    agent = create_agent(llm, tools=tools, middleware=[guard], response_format=Verdict)
    result = await agent.ainvoke({"messages": [HumanMessage(content="investigate card_good_001")]})

    verdict = result["structured_response"]
    assert isinstance(verdict, Verdict) and verdict.pattern == "legitimate"

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    by_id = {m.tool_call_id: m for m in tool_msgs}
    assert '"found": true' in by_id["t1"].content
    assert by_id["t2"].status == "error" and "blocked by Jev guardrail" in by_id["t2"].content

    assert len(jev.seen) == 2
    assert [b["args"]["card_id"] for b in guard.blocked] == ["card_ato_002"]
    assert len(guard.checked) == 2

    calls = _extract_tool_calls(result["messages"])
    assert [c["name"] for c in calls] == ["get_card_history", "get_card_history", "Verdict"]
    assert "blocked" in calls[1]["result"]


async def test_guard_ignores_unlisted_tools():
    store = FeatureStore()
    tools = build_tools(store)
    jev = FakeJev(allowed_card="anything")
    guard = JevToolGuard(tools=["get_card_history"], api_key="x", classifier=jev)  # type: ignore[arg-type]
    llm = _fake_llm(
        AIMessage(
            content="", tool_calls=[{"id": "t1", "name": "email_domain_reputation", "args": {"domain": "yopmail.com"}}]
        ),
        AIMessage(content="done"),
    )
    agent = create_agent(llm, tools=tools, middleware=[guard])
    result = await agent.ainvoke({"messages": [HumanMessage(content="check")]})
    assert jev.seen == []
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "disposable" in tool_msg.content
