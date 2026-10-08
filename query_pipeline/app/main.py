import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from .schemas import QueryRequest, QueryResponse

from identifier_extraction.predictor import IdentifierExtractor
from intent_classification.predictor import IntentClassifier
from workflow_errors import TaskExecutionError
from .error_handlers import register_error_handlers
from .extensions import EXTRA_TASKS
from .graph import build_graph, evaluate

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


def create_app(artifact_dir=None, extra_tasks=None):
    configured = Path(artifact_dir or os.getenv("ARTIFACT_DIR", "artifacts"))
    directory = configured if configured.is_absolute() else ROOT / configured
    tasks = EXTRA_TASKS if extra_tasks is None else extra_tasks

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
        # Fail at startup for missing/incompatible files. Do not turn a broken
        # deployment into a misleading empty intent response.
        classifier = IntentClassifier(directory / "intent_classification")
        extractor = IdentifierExtractor(directory / "identifier_extraction")
        app.state.allowed_contexts = set(classifier.metadata["labels"])
        app.state.graph = build_graph(classifier, extractor, tasks)
        yield
        del app.state.graph

    app = FastAPI(title="Intent and identifier query pipeline", lifespan=lifespan)

    register_error_handlers(app)

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.post("/evaluate", response_model=QueryResponse, response_model_exclude_none=True)
    def evaluate_query(body: QueryRequest):
        try:
            if body.context and body.context not in app.state.allowed_contexts:
                raise TaskExecutionError(task="request_validation", code="INVALID_CONTEXT",
                                         message="The supplied context is not supported.", status_code=422)
            return evaluate(app.state.graph, body.model_dump())
        except TaskExecutionError:
            raise
        except Exception as exc:
            log.exception("Query pipeline execution failed")
            raise TaskExecutionError(task="workflow", code="WORKFLOW_EXECUTION_FAILED",
                                     message="Query evaluation failed.", status_code=500) from exc

    return app


app = create_app()
