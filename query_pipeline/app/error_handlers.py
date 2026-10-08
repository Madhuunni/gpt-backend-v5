"""Single HTTP translation point for all workflow task errors."""
from fastapi.responses import JSONResponse
from workflow_errors import TaskExecutionError


def register_error_handlers(app):
    @app.exception_handler(TaskExecutionError)
    async def handle_task_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail()})
