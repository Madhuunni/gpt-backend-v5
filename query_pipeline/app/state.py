from typing import Any, TypedDict


class QueryState(TypedDict, total=False):
    query: str
    pending_field: str
    status: str
    resuming: bool
    context: str
    action_required: str
    identifiers: dict[str, str]
    results: dict[str, Any]
