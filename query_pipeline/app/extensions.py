"""Register additional tasks here, in execution order."""
from http_task.task import call_http_api
from .graph import ExtraTask


EXTRA_TASKS: list[ExtraTask] = [
    ExtraTask("call_http_api", call_http_api),
]
