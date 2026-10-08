"""Run: uvicorn examples.extended_app:app --port 8000"""
from app.graph import ExtraTask
from app.main import create_app


def build_search_parameters(state):
    return {"context": state["context"], "filters": {k:v for k,v in state["identifiers"].items() if v}}


app = create_app(extra_tasks=[ExtraTask("build_search_parameters", build_search_parameters, require_identifiers=True)])
