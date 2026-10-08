from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.graph import ExtraTask, build_graph, evaluate
from workflow_errors import IntentNotFoundError
from app.main import ROOT, create_app
from identifier_extraction.predictor import IdentifierExtractor
from intent_classification.predictor import IntentClassifier


class Classifier:
    def __init__(self, context):
        self.context = context
    def predict(self, query):
        return self.context


class Extractor:
    calls = 0
    def __init__(self, result=None):
        self.result = result or {}
        self.queries = []
    def predict(self, query):
        self.calls += 1
        self.queries.append(query)
        return self.result


def test_unknown_skips_extraction_and_extensions():
    extractor = Extractor()
    def forbidden(state):
        raise AssertionError("must not run")
    graph = build_graph(Classifier(""), extractor, [ExtraTask("later", forbidden)])
    with pytest.raises(IntentNotFoundError):
        evaluate(graph,"unknown query")
    assert extractor.calls == 0


def test_original_query_passed_to_extractor():
    extractor = Extractor({"employee_id":"00123","case_id":"","sr_no":""})
    query = "  Show overpayment details for employee 00123  "
    result = evaluate(build_graph(Classifier("case_details"),extractor),query)
    assert extractor.queries == [query]
    assert result["identifiers"]["employee_id"] == "00123"


def test_empty_identifiers_request_input():
    result = evaluate(build_graph(Classifier("known"),Extractor()),"text")
    assert result["action_required"]
    assert result["context"] == "known"


def test_extra_tasks_sequential_and_core_immutable():
    def first(state):
        state["context"] = "mutated"
        return {"value":1}
    def second(state):
        assert state["context"] == "known"
        return {"value":state["results"]["first"]["value"]+1}
    graph = build_graph(Classifier("known"),Extractor({"employee_id":"123"}),[ExtraTask("first",first),ExtraTask("second",second)])
    result = evaluate(graph,"q")
    assert result["results"] == {"first":{"value":1},"second":{"value":2}}


def test_extension_can_require_identifiers():
    task=ExtraTask("later",lambda s: {"ran":True},require_identifiers=True)
    result = evaluate(build_graph(Classifier("known"),Extractor(),[task]),"q")
    assert result["action_required"]
    assert "results" not in result


@pytest.mark.parametrize("name", ["context","intent_classification","bad-name"])
def test_invalid_extension_name(name):
    with pytest.raises(ValueError):
        build_graph(Classifier(""),Extractor(),[ExtraTask(name,lambda s: {})])


@pytest.fixture(scope="module")
def real_models():
    return (IntentClassifier(ROOT/"artifacts/intent_classification"),
            IdentifierExtractor(ROOT/"artifacts/identifier_extraction"))


@pytest.mark.parametrize("identifier,field,value", [("employee ID 0012345","employee_id","0012345"),
                                                    ("case ID 9001","case_id","9001"),("SR number 8001","sr_no","8001")])
def test_real_weights(real_models,identifier,field,value):
    result = evaluate(build_graph(*real_models),"Show overpayment details for " + identifier)
    assert result["context"] == "case_details"
    assert result["identifiers"][field] == value
    assert set(result["identifiers"]) == {"case_id","employee_id","sr_no"}


def test_real_unsupported_and_no_id(real_models):
    graph=build_graph(*real_models)
    with pytest.raises(IntentNotFoundError):
        evaluate(graph,"zzzz qqqqq")
    assert evaluate(graph,"Show overpayment details")["action_required"]


def test_original_duplicate_policy(real_models):
    assert real_models[1].predict("Show overpayment details for employee 12345 and employee 54321")["employee_id"] == "12345"


def test_low_confidence_abstains(real_models):
    classifier=IntentClassifier(ROOT/"artifacts/intent_classification")
    classifier.threshold=1.0
    assert classifier.predict("Show overpayment details for employee 12345") == ""
    extractor=IdentifierExtractor(ROOT/"artifacts/identifier_extraction")
    extractor.threshold=1.0
    assert extractor.predict("Show overpayment details for employee 12345") == {}


def test_concurrent_request_isolation(real_models):
    graph=build_graph(*real_models)
    ids=["12345","0067890","901234"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda x:evaluate(graph,"Show overpayment details for employee ID " + x),ids))
    assert [r["identifiers"]["employee_id"] for r in results] == ids


def test_missing_artifacts_fail_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path)):
            pass


def test_api():
    with TestClient(create_app(extra_tasks=[])) as client:
        assert client.get("/health").json() == {"status":"ready"}
        assert client.post("/evaluate",json={"query":"Show overpayment details for employee 12345"}).json()["identifiers"]["employee_id"] == "12345"
        unknown = client.post("/evaluate",json={"query":"zzzz qqqqq"})
        assert unknown.status_code == 422
        assert unknown.json()["detail"]["code"] == "INTENT_NOT_FOUND"
        for payload in [{},{"query":" "},{"query":123},{"query":"x","extra":1}]:
            assert client.post("/evaluate",json=payload).status_code == 422


def test_custom_artifact_path(tmp_path):
    import shutil
    shutil.copytree(ROOT/"artifacts",tmp_path/"custom")
    with TestClient(create_app(tmp_path/"custom", extra_tasks=[])) as client:
        assert client.post("/evaluate",json={"query":"Show overpayment details for case ID 9001"}).json()["identifiers"]["case_id"] == "9001"


def test_task_error_is_not_disguised_as_unknown_intent():
    def broken(state):
        raise RuntimeError("Internal details")
    with TestClient(create_app(extra_tasks=[ExtraTask("broken",broken)])) as client:
        response=client.post("/evaluate",json={"query":"Show overpayment details for employee 12345"})
        assert response.status_code == 500
        assert response.json() == {"detail":{"task":"broken", "code":"TASK_EXECUTION_FAILED", "message":"Task execution failed."}}


def test_original_classes_and_other_contexts(real_models):
    from intent_classification.model import ContextClassifier
    from identifier_extraction.model import IDTagger
    assert isinstance(real_models[0].model, ContextClassifier)
    assert isinstance(real_models[1].model, IDTagger)
    for query, context in [("Show installment details for employee ID 12345", "installment_details"),
                           ("Show payment history for employee ID 12345", "payment_history")]:
        assert real_models[0].predict(query) == context


def test_extraction_bounds_and_numeric_contract(real_models):
    assert real_models[1].predict("word " * 513) == {}
    assert real_models[1].predict("Show overpayment details for employee ID ABC123") == {}


def test_api_missing_ids_stops_http(monkeypatch):
    import httpx
    def forbidden(*args, **kwargs):
        pytest.fail("Downstream HTTP must not run")
    monkeypatch.setattr(httpx, "post", forbidden)
    with TestClient(create_app()) as client:
        response = client.post("/evaluate", json={"query":"Show overpayment details"})
        assert response.status_code == 200
        assert response.json()["action_required"]
        assert "detail" not in response.json()
