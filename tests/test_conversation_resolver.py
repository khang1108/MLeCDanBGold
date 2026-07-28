from __future__ import annotations
import pytest
import hcmai.agents.kisc.resolver as resolver_module
from hcmai.agents.kisc import ConversationResolver, ConversationResolverError
from hcmai.common.schemas import ConversationState, ConversationTurn, FrameFeedback

def _payload(query, positive=(), negative=(), uncertain=(), accepted=(), rejected=()):
    return {
        "standalone_query": query, "positive_constraints": list(positive),
        "negative_constraints": list(negative),
        "uncertain_constraints": list(uncertain),
        "accepted_frame_ids": list(accepted), "rejected_frame_ids": list(rejected),
    }
class FakeCall:
    def __init__(self, value):
        self.value, self.calls, self.request = value, 0, None

    def __call__(self, request):
        self.calls, self.request = self.calls + 1, request
        if isinstance(self.value, Exception):
            raise self.value
        return self.value
TURNS = [ConversationTurn(
    turn_id="t1", sender="user", message="một người", created_at=1)]
CASES = [
    ("empty_history", False, "người đi bộ", None, None, _payload("người đi bộ")),
    ("one_message", False, "xe đỏ", None, None, _payload("xe đỏ", ["xe đỏ"])),
    ("accumulation", True, "gần cửa hàng", _payload("xe đạp", ["xe đạp"]), None,
     _payload("xe đạp gần cửa hàng", ["xe đạp", "gần cửa hàng"])),
    ("pronoun", True, "người đó đội mũ", _payload("một người", ["một người"]), None,
     _payload("một người đội mũ", ["một người", "đội mũ"])),
    ("addition", True, "trời tối", _payload("ô tô", ["ô tô"]), None,
     _payload("ô tô trời tối", ["ô tô", "trời tối"])),
    ("correction", True, "không, áo xanh", _payload("áo đỏ", ["áo đỏ"]), None,
     _payload("áo xanh", ["áo xanh"])),
    ("positive_negative", True, "không áo đỏ", _payload("áo đỏ", ["áo đỏ"]), None,
     _payload("không áo đỏ", negative=["áo đỏ"])),
    ("negative_positive", True, "có đội mũ", _payload("không mũ", negative=["mũ"]), None,
     _payload("đội mũ", ["mũ"])),
    ("uncertain", False, "có lẽ màu xanh", None, None,
     _payload("có lẽ màu xanh", uncertain=["màu xanh"])),
    ("negation", False, "không có xe", None, None,
     _payload("không có xe", negative=["xe"])),
    ("contradiction", True, "ban đêm", _payload("ban ngày", ["ban ngày"]), None,
     _payload("ban đêm", ["ban đêm"])),
    ("accepted", False, "giữ ảnh", None, {"accepted_frame_ids": ["f1"]},
     _payload("giữ ảnh", accepted=["f1"])),
    ("rejected", False, "bỏ ảnh", None, {"rejected_frame_ids": ["f1"]},
     _payload("bỏ ảnh", rejected=["f1"])),
    ("accepted_rejected", False, "bỏ ảnh", _payload("q", accepted=["f1"]),
     {"rejected_frame_ids": ["f1"]}, _payload("q", rejected=["f1"])),
    ("rejected_accepted", False, "giữ ảnh", _payload("q", rejected=["f1"]),
     {"accepted_frame_ids": ["f1"]}, _payload("q", accepted=["f1"])),
    ("duplicate_constraints", False, "xe", None, None,
     _payload("xe", ["xe", "xe"])),
    ("duplicate_frames", False, "giữ", None, {"accepted_frame_ids": ["f1", "f1"]},
     _payload("giữ", accepted=["f1", "f1"])),
]
@pytest.mark.parametrize("case,has_history,current,prior,event,output", CASES,
                         ids=[row[0] for row in CASES])
def test_resolver_contract_cases(case, has_history, current, prior, event, output):
    previous = ConversationState.model_validate(prior) if prior else None
    feedback = FrameFeedback.model_validate(event) if event else None
    before = previous.model_dump() if previous else None
    call = FakeCall(output)
    state = ConversationResolver(call).resolve(
        TURNS if has_history else [], current, feedback, previous)
    assert isinstance(state, ConversationState) and call.calls == 1
    assert state == ConversationState.model_validate(output)
    assert previous is None or previous.model_dump() == before
    request = call.request
    assert request is not None
    expected_history = TURNS if has_history else []
    assert request["history"] == [turn.model_dump(mode="json") for turn in expected_history]
    assert request["current_message"] == current
    assert request["feedback"] == (feedback.model_dump(mode="json") if feedback else None)
    assert request["previous_state"] == (previous.model_dump(mode="json") if previous else None)
    if case == "accumulation":
        text = request["instruction"].lower()
        assert "standalone_query" in request["response_schema"]["properties"]
        for phrase in ("complete conversationstate", "newest-wins",
                       "positive_constraints", "negative_constraints",
                       "uncertain_constraints", "do not retrieve", "use tools",
                       "react", "chain-of-thought", "concise english visual", "never invent"):
            assert phrase in text
        forbidden = {"SearchEngine", "KiscSessionManager", "SearchRequest"}
        assert not forbidden & set(vars(resolver_module))

@pytest.mark.parametrize("output", [
    "malformed", {"standalone_query": "delta"},
    {**_payload("q"), "positive_constraints": [" "]},
    RuntimeError("provider details " * 100),
])
def test_failures_are_bounded_single_call_and_non_mutating(output):
    previous = ConversationState(standalone_query="old", accepted_frame_ids=["f1"])
    before, call = previous.model_dump(), FakeCall(output)
    with pytest.raises(ConversationResolverError) as error:
        ConversationResolver(call).resolve([], "new", previous_state=previous)
    assert call.calls == 1 and previous.model_dump() == before
    assert len(str(error.value)) <= 200
