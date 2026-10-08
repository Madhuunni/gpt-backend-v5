"""Shared task error contract, independent of FastAPI and model implementations."""
import logging
from functools import wraps

log = logging.getLogger(__name__)


class TaskExecutionError(Exception):
    def __init__(self, *, task, code, message, status_code=500):
        super().__init__(message)
        self.task = task
        self.code = code
        self.status_code = status_code

    def detail(self):
        return {"task": self.task, "code": self.code, "message": str(self)}


class IntentNotFoundError(TaskExecutionError):
    def __init__(self):
        super().__init__(task="intent_classification", code="INTENT_NOT_FOUND",
                         message="Could not identify the request context. Please rephrase your query.",
                         status_code=422)


class MissingIdentifiersError(TaskExecutionError):
    def __init__(self, message="No identifiers found. Provide an employee ID, case ID, or SR number."):
        super().__init__(task="extract_identifiers", code="IDENTIFIERS_NOT_FOUND",
                         message=message, status_code=422)


class HTTPTaskError(TaskExecutionError):
    def __init__(self, message, status_code=502, code="DOWNSTREAM_HTTP_ERROR"):
        super().__init__(task="call_http_api", code=code, message=message,
                         status_code=status_code)


def task_boundary(name, function):
    """Wrap every graph node: preserve expected errors; label unexpected failures."""
    @wraps(function)
    def run(state):
        try:
            return function(state)
        except TaskExecutionError:
            raise
        except Exception as exc:
            log.exception("Task %s failed", name)
            raise TaskExecutionError(task=name, code="TASK_EXECUTION_FAILED",
                                     message="Task execution failed.", status_code=500) from exc
    return run
