"""Client-carried conversation contract. Models perform request validation."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Identifiers(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_id: str = Field(default="", pattern=r"^[0-9]*$")
    employee_id: str = Field(default="", pattern=r"^[0-9]*$")
    sr_no: str = Field(default="", pattern=r"^[0-9]*$")


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=4000)
    context: str = Field(default="", max_length=200)
    identifiers: Identifiers = Field(default_factory=Identifiers)
    action_required: str = Field(default="", max_length=1000)
    pending_field: Literal["", "identifier", "case_id", "employee_id", "sr_no"] = ""
    status: Literal["", "awaiting_identifier", "ready", "completed"] = ""
    # Accept an echoed response, but never use old results for execution.
    results: dict[str, Any] | None = None

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value

    @field_validator("context")
    @classmethod
    def trim_context(cls, value):
        return value.strip()


class QueryResponse(BaseModel):
    query: str
    context: str
    identifiers: Identifiers
    action_required: str = ""
    pending_field: str = ""
    status: str
    results: dict[str, Any] | None = None
