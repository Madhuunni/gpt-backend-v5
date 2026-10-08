# Query pipeline: FastAPI, LangGraph, ContextClassifier and IDTagger

This release carries conversation state in the JSON request and response.
Original trained PyTorch models and weights are included. Inference uses no LLM.
The dependency baseline remains Python 3.10 and PyTorch 2.2.1.

## Run

```bash
python -m pip install -r requirements.txt
export DOWNSTREAM_API_URL='http://localhost:9000/api/details'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs. Optional settings:
`DOWNSTREAM_TIMEOUT_SECONDS=15`, `DOWNSTREAM_API_TOKEN` (bearer token),
`INTENT_THRESHOLD=0.7`, `TORCH_NUM_THREADS=2`, and `ARTIFACT_DIR`.
Relative artifact paths resolve against the project directory.
Models load once per worker. Missing/incompatible artifacts fail startup.
If `import torch` fails with iJIT_NotifyEvent, repair the native environment;
application changes cannot fix a shared-library import error.

## Standard message

POST `/evaluate` with:

```json
{
  "query": "show details regarding the overpayment for the case",
  "context": "",
  "identifiers": {"case_id":"", "employee_id":"", "sr_no":""},
  "action_required": "",
  "pending_field": "",
  "status": ""
}
```

Only `query` is mandatory, preserving compatibility with query-only clients.
If the model recognizes case_details but finds no ID, HTTP 200 returns:

```json
{
  "query": "show details regarding the overpayment for the case",
  "context": "case_details",
  "identifiers": {"case_id":"", "employee_id":"", "sr_no":""},
  "action_required": "Please enter a case ID, or reply with 'employee ID <number>' or 'SR number <number>'. A number alone is treated as a case ID.",
  "pending_field": "identifier",
  "status": "awaiting_identifier"
}
```

Save that response for the current conversation. For the next turn, replace
`query` with the new user text and POST the object again:

```json
{
  "query": "120003",
  "context": "case_details",
  "identifiers": {"case_id":"", "employee_id":"", "sr_no":""},
  "action_required": "Please enter a case ID, or reply with 'employee ID <number>' or 'SR number <number>'. A number alone is treated as a case ID.",
  "pending_field": "identifier",
  "status": "awaiting_identifier"
}
```

The supplied context bypasses classification. A numeric-only reply sets
case_id to `120003` without calling IDTagger, preserving leading zeros.
The HTTP task runs and its result is included under `results.call_http_api`.
The response has `status: "completed"`, `pending_field: ""`, and
`action_required: ""`. Without extension tasks, successful extraction returns
`status: "ready"`.

## Routing rules

| Input | Behavior |
|---|---|
| Empty context | Run classification, then extraction |
| Supported nonempty context | Skip classification and process identifiers |
| Nonempty context + numeric-only query | Set case_id directly; skip IDTagger |
| Nonempty context + labeled/multiword query | Run IDTagger on the exact query |
| No ID available | Return action_required; stop before HTTP/extensions |
| At least one ID available | Clear action_required and pending_field; run extensions |
| Unsupported supplied context | HTTP 422, INVALID_CONTEXT |

Examples routed through IDTagger: `The case id is 1111`,
`The Employee ID is 2222`, and `The SR NO is 3333`.
The original model supports numeric IDs only; xxxx/ZZZZ are placeholders,
not accepted alphabetic IDs. A single word such as `hello` is not assigned
as an ID. The numeric shortcut applies only when context was supplied on input.

Newly extracted values overwrite the same carried identifier field. Other
carried fields remain populated. If multiple fields are present, the downstream
service must implement its intended lookup rules; this API does not choose one.
If no new ID is extracted but a carried ID exists, processing continues with it.

## Client state and reset

The client owns conversation state; the backend stores no conversation session.
Send the returned context and identifiers with each follow-up. Send empty context
and empty identifiers to start a new request or change topic. When context is
empty, carried identifiers are discarded to avoid using IDs from another request.
A nonempty context always bypasses classification, even if the text changes topic.

`action_required`, `pending_field` and `status` are recomputed on each call.
Incoming values do not force routing. The generic pending field `identifier`
allows any of the three labeled IDs; a numeric-only reply always defaults to case_id.
Response `results` may be echoed back but are discarded before execution.
The six standard fields are always returned, with optional `results` after tasks run.
Each POST is a fresh execution, not an idempotent resume: resending a completed
message can repeat downstream work.

Context is validated against classifier metadata at the API boundary. The bundled
labels are case_details, installment_details and payment_history. Train and deploy
new artifacts to change them. A supplied context/identifier is client input, not
proof of permission to access an employee/case; enforce authorization in your service.

## Project layout

- app/schemas.py: request/response models for the standard JSON message.
- app/state.py: internal LangGraph state.
- app/graph.py: routing, numeric shortcut, extraction and extension sequence.
- app/main.py: model startup, context validation and FastAPI endpoint.
- app/extensions.py: ordered ExtraTask registration.
- workflow_errors.py and app/error_handlers.py: shared task errors and HTTP mapping.
- intent_classification/: original ContextClassifier and TF-IDF adapter.
- identifier_extraction/: original IDTagger, tokenizer and extraction adapter.
- http_task/task.py: HTTP POST task.
- artifacts/: original paired weights and metadata for both models.
- data/: original datasets, separated by task.

## HTTP task and errors

The HTTP task posts `query`, `context` and `identifiers` to DOWNSTREAM_API_URL.
It returns `{ "status_code": 200, "data": ... }` under results.call_http_api.
Any JSON response value is supported; HTTP 204 yields null data. No redirects
or automatic retries. The timeout is an HTTPX network-operation timeout.
Later tasks read `state["results"]["call_http_api"]["data"]`.

Missing IDs are a normal HTTP 200 action-required response. Other task failures
retain `{ "detail": { "task": "...", "code": "...", "message": "..." } }`:

- Unknown classified intent: 422 INTENT_NOT_FOUND.
- Unsupported supplied context: 422 INVALID_CONTEXT.
- Invalid HTTP configuration: 500 HTTP_CONFIGURATION_ERROR.
- Downstream non-2xx: 502 DOWNSTREAM_HTTP_ERROR.
- Invalid JSON: 502 DOWNSTREAM_INVALID_RESPONSE.
- Connection failure: 502 DOWNSTREAM_CONNECTION_ERROR.
- Timeout: 504 DOWNSTREAM_TIMEOUT.
- Unexpected task failure: 500 TASK_EXECUTION_FAILED.

Task errors stop execution; earlier side effects are not rolled back. Request
validation uses FastAPI's standard 422 response. New tasks can raise
TaskExecutionError(task=..., code=..., message=..., status_code=...). Register
ExtraTask instances in app/extensions.py; results are stored by task name.

## Training and verification

```bash
python -m intent_classification.classifier train \
  --data data/intent_classification/demo_train.json \
  --output candidate_artifacts/intent_classification
python -m identifier_extraction.train \
  --data data/identifier_extraction \
  --output candidate_artifacts/identifier_extraction
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Deploy each matching weight/metadata pair together and restart workers. Original
reports describe source-model performance, not production accuracy. The standalone
`evaluate.py` CLI remains a single-turn, query-only path running the two model tasks.
