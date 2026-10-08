# Compatibility and verification

This integration was tested with Python 3.10.21, PyTorch 2.2.1+cpu,
NumPy 1.26.4, scikit-learn 1.4.2, FastAPI 0.115.8 and LangGraph 0.2.76.
All 24 tests passed. Two dependency deprecation warnings did not affect results.
The user's specific Conda CUDA/MKL environment was not available for testing.

The original ContextClassifier is a TF-IDF-fed neural classifier; IDTagger is a
bidirectional LSTM token classifier. Both original weight files load directly
with torch.load(weights_only=True) under PyTorch 2.2.1. They were copied unchanged,
not retrained or converted. Their metadata records the original training runtime;
that field does not impose an inference-runtime version requirement.

Changes from the preceding query_pipeline release:

- Replaced the GRU models and artifacts with the original ContextClassifier and
  IDTagger architectures, trained weights and preprocessing.
- Added scikit-learn for the original classifier's TF-IDF feature reconstruction.
- Preserved the three original context labels and numeric-only ID decoding.
- Added a configurable classifier confidence cutoff (API default 0.7).
- Converted all-empty extraction results to {}, preserving the HTTP contract.
- Kept conditional LangGraph routing, extension tasks, startup loading and JSON API.

Your installed CUDA-enabled torch 2.2.1 satisfies the torch pin; CPU-only wheels
are not required. Inference runs on CPU and does not need torchvision/torchaudio.
If your environment still raises iJIT_NotifyEvent during import torch, resolve the
native PyTorch/MKL dependency mismatch first. Python application changes cannot
repair an import-time shared-library symbol error. check_environment.py provides
the relevant diagnostic. Existing environment repair guidance still applies.

HTTP task update: added HTTPX 0.28.1 and a default POST task. The earlier
24-test result above describes the model integration release. HTTP task checks
use mocked HTTP responses; no real downstream service was supplied.

All 7 new HTTP task tests passed with Python 3.12 and HTTPX 0.28.1.
The full PyTorch model suite was not rerun for this HTTP-only update.

Missing-ID validation update: the extraction graph node now raises
MissingIdentifiersError for empty/all-blank supported ID fields. FastAPI maps
it to HTTP 422. Graph and API regression tests were updated; Python syntax
compilation passed. Runtime tests were not rerun for this update because the
test dependency setup could not be completed in the execution environment.

Unified errors update: 15 focused graph/HTTP/error-handler tests passed under
Python 3.12 with FastAPI 0.115.8, LangGraph 0.2.76 and HTTPX 0.28.1.
Expected and unexpected task errors share one contract; intent rejection now
returns HTTP 422. Model weights are unchanged; the full model suite was not
rerun for this update.

Action-required update: missing IDs now end the graph normally and return
HTTP 200 with context, empty ID fields and an action_required message.
The HTTP task and later tasks are skipped. Successful extraction sets
action_required to an empty string. All 16 focused graph/HTTP/error-handler
tests passed on Python 3.12; syntax compilation passed. The full PyTorch
model/API suite was not rerun. Original model weights are unchanged.

Client-carried conversation update: added standard request/response schemas,
context-based classification bypass, and a numeric-only follow-up shortcut
to case_id. Labeled replies still use the original IDTagger. All 24 focused
conversation/graph/HTTP/error-handler tests passed on Python 3.12, including
an HTTP schema round-trip with stub model predictions. The original weights
are unchanged; full PyTorch model inference tests were not rerun.
