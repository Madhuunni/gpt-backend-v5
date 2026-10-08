import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.graph import ExtraTask, build_graph, evaluate
from app.schemas import QueryRequest, QueryResponse


class Classifier:
    calls = 0
    def predict(self, query):
        self.calls += 1
        return "case_details"


class Extractor:
    def __init__(self):
        self.queries = []
    def predict(self, query):
        self.queries.append(query)
        return {"The case id is 1111": {"case_id":"1111"},
                "The Employee ID is 2222": {"employee_id":"2222"},
                "The SR NO is 3333": {"sr_no":"3333"}}.get(query, {})


def test_two_turn_numeric_reply_skips_both_models():
    classifier, extractor = Classifier(), Extractor()
    calls = []
    graph = build_graph(classifier, extractor, [ExtraTask("http", lambda s: calls.append(s) or {"ok":True})])
    first = evaluate(graph, "show details regarding the overpayment for the case")
    assert first["status"] == "awaiting_identifier"
    assert first["pending_field"] == "identifier"
    assert first["action_required"]
    assert calls == []
    reply = {**first, "query":"00120003"}
    final = evaluate(graph, reply)
    assert final["identifiers"]["case_id"] == "00120003"
    assert final["context"] == "case_details"
    assert final["status"] == "completed"
    assert final["pending_field"] == final["action_required"] == ""
    assert classifier.calls == 1 and len(extractor.queries) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("query,field,value", [
    ("The case id is 1111", "case_id", "1111"),
    ("The Employee ID is 2222", "employee_id", "2222"),
    ("The SR NO is 3333", "sr_no", "3333"),
])
def test_labeled_reply_uses_extraction(query,field,value):
    classifier, extractor = Classifier(), Extractor()
    result = evaluate(build_graph(classifier, extractor), {"query":query,"context":"case_details"})
    assert classifier.calls == 0
    assert extractor.queries == [query]
    assert result["identifiers"][field] == value
    assert result["status"] == "ready"


def test_non_numeric_single_word_is_not_used_as_case_id():
    extractor = Extractor()
    result = evaluate(build_graph(Classifier(),extractor), {"query":"hello", "context":"case_details"})
    assert result["status"] == "awaiting_identifier"
    assert not any(result["identifiers"].values())
    assert extractor.queries == ["hello"]


def test_carried_ids_merge_and_old_results_ignored():
    result = evaluate(build_graph(Classifier(), Extractor()), {
        "query":"The case id is 1111", "context":"case_details",
        "identifiers":{"employee_id":"009"}, "results":{"fake":{"ok":True}},
        "action_required":"stale", "status":"completed", "pending_field":"employee_id",
    })
    assert result["identifiers"] == {"case_id":"1111","employee_id":"009","sr_no":""}
    assert result["action_required"] == ""
    assert "results" not in result


def test_empty_context_starts_fresh():
    classifier = Classifier()
    result = evaluate(build_graph(classifier, Extractor()), {
        "query":"new query", "context":"", "identifiers":{"case_id":"999"}
    })
    assert classifier.calls == 1
    assert result["status"] == "awaiting_identifier"


def test_api_contract_round_trip():
    graph = build_graph(Classifier(), Extractor())
    app = FastAPI()
    @app.post("/evaluate", response_model=QueryResponse, response_model_exclude_none=True)
    def run(body:QueryRequest):
        return evaluate(graph,body.model_dump())
    with TestClient(app) as client:
        first = client.post("/evaluate",json={"query":"overpayment details"})
        assert first.status_code == 200
        body = first.json()
        assert set(body) == {"query","context","identifiers","action_required","pending_field","status"}
        body["query"] = "120003"
        second = client.post("/evaluate",json=body)
        assert second.status_code == 200
        assert second.json()["identifiers"]["case_id"] == "120003"
        assert client.post("/evaluate",json={"query":"q","identifiers":{"case_id":"ABC"}}).status_code == 422
