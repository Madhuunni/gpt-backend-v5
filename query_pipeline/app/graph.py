"""Conditional two-task graph with optional sequential extension tasks."""
import copy
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from .state import QueryState
from workflow_errors import IntentNotFoundError, task_boundary


@dataclass(frozen=True)
class ExtraTask:
    name: str
    run: Callable[[QueryState], dict]
    require_identifiers: bool = False


def build_graph(classifier, extractor, extra_tasks: Sequence[ExtraTask] = ()):
    graph = StateGraph(QueryState)

    def classify(state):
        context = classifier.predict(state["query"])
        if context is None or (isinstance(context, str) and not context.strip()):
            raise IntentNotFoundError()
        if not isinstance(context, str):
            raise TypeError("Classifier must return a string")
        return {"context": context}

    def extract(state):
        fields = ("case_id", "employee_id", "sr_no")
        identifiers = {field: state.get("identifiers", {}).get(field, "") for field in fields}
        reply = state["query"].strip()
        if state.get("resuming") and re.fullmatch(r"[0-9]+", reply):
            # Explicit product rule: an unlabeled numeric reply means case ID.
            extracted = {"case_id": reply}
        else:
            extracted = extractor.predict(state["query"])
        for field in fields:
            value = extracted.get(field, "")
            if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
                identifiers[field] = value
        if not any(identifiers.values()):
            return {
                "identifiers": identifiers,
                "action_required": "Please enter a case ID, or reply with 'employee ID <number>' or 'SR number <number>'. A number alone is treated as a case ID.",
                "pending_field": "identifier",
                "status": "awaiting_identifier",
            }
        return {"identifiers": identifiers, "action_required": "", "pending_field": "", "status": "ready"}

    graph.add_node("intent_classification", task_boundary("intent_classification", classify))
    graph.add_node("extract_identifiers", task_boundary("extract_identifiers", extract))
    graph.add_conditional_edges(
        START, lambda state: "resume" if state.get("context") else "new",
        {"resume": "extract_identifiers", "new": "intent_classification"},
    )
    graph.add_edge("intent_classification", "extract_identifiers")
    previous = "extract_identifiers"
    reserved = {START, END, "intent_classification", "extract_identifiers", "query", "context", "identifiers", "results", "action_required", "pending_field", "status", "resuming"}
    for task in extra_tasks:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", task.name) or task.name in reserved:
            raise ValueError("Invalid or duplicate extra task name: " + task.name)
        reserved.add(task.name)
        def run_extra(state, task=task):
            if task.require_identifiers and not state["identifiers"]:
                return {}
            # Extension code cannot mutate input state or overwrite core outputs.
            result = task.run(copy.deepcopy(state))
            if not isinstance(result, dict):
                raise TypeError("Extra tasks must return a JSON object")
            json.dumps(result, allow_nan=False)
            return {"results": {**state.get("results", {}), task.name: result}}
        graph.add_node(task.name, task_boundary(task.name, run_extra))
        if previous != "extract_identifiers":
            graph.add_edge(previous, task.name)
        previous = task.name
    if previous != "extract_identifiers":
        graph.add_edge(previous, END)
    next_task = extra_tasks[0].name if extra_tasks else END
    graph.add_conditional_edges(
        "extract_identifiers",
        lambda state: "needs_input" if state.get("action_required") else "continue",
        {"needs_input": END, "continue": next_task},
    )
    return graph.compile()


def evaluate(graph, query):
    # State is explicitly supplied by the client, never held in a global session.
    message = {"query": query} if isinstance(query, str) else query
    context = message.get("context", "").strip()
    fields = ("case_id", "employee_id", "sr_no")
    carried = message.get("identifiers", {}) if context else {}
    state = graph.invoke({
        "query": message["query"], "context": context,
        "identifiers": {field: carried.get(field, "") for field in fields},
        "resuming": bool(context), "results": {}, "action_required": "",
        "pending_field": "", "status": "",
    })
    response = {key: state[key] for key in (
        "query", "context", "identifiers", "action_required", "pending_field", "status"
    )}
    if state.get("results"):
        response["results"] = state["results"]
        response["status"] = "completed"
    return response
