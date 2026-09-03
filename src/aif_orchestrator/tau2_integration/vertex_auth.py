"""Vertex AI Model Garden access via the generic OpenAI-compatible
passthrough endpoint (`.../endpoints/openapi/chat/completions`), used
for `google/gemini-2.5-flash` (confirmed via curl and litellm: proper
`tool_calls`, no leaked reasoning text -- unlike MiniMax-M2 through the
same endpoint, which returned empty `tool_calls` with the real call
leaked as unparsed XML in `content`, and unlike Groq's `openai/gpt-oss-
120b`, which hit an intermittent litellm provider-resolution race
under concurrency).

litellm's native `vertex_ai/` provider (Application Default
Credentials) hit a permission-denied error in this project that a
plain `gcloud auth print-access-token` bearer call did not -- rather
than debug the ADC/quota-project mismatch, this uses the
already-proven-working path: treat the endpoint as a generic
OpenAI-compatible API (`custom_llm_provider="openai"`) with an
explicit bearer token as `api_key`.

`gcloud auth print-access-token` tokens expire in ~1 hour, well inside
a real sweep's runtime, so a background thread refreshes the token in
place inside a shared, mutable kwargs dict -- callers spread
`**VERTEX_KWARGS` into each `generate()`/completion call, and because
dict unpacking reads current values at call time (not at whatever
point the dict reference was captured), each call picks up a fresh
token without needing to touch call sites again after a refresh.
"""
import os
import subprocess
import threading
import time

REFRESH_INTERVAL_SECONDS = 45 * 60  # tokens live ~60min; refresh well inside that


def _fetch_token() -> str:
    # shell=True: on Windows, "gcloud" resolves to gcloud.cmd, which
    # subprocess can't exec directly without going through a shell.
    result = subprocess.run(
        "gcloud auth print-access-token",
        shell=True, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _endpoint_url(project_id: str, region: str) -> str:
    # litellm's "openai" custom provider appends "/chat/completions"
    # itself, so api_base must stop at ".../openapi" -- the curl example
    # Vertex's own console shows includes the suffix because that's a
    # raw HTTP call, not something routed through litellm's URL builder.
    host = os.environ.get("ENDPOINT", "aiplatform.googleapis.com")
    return f"https://{host}/v1/projects/{project_id}/locations/{region}/endpoints/openapi"


VERTEX_KWARGS: dict = {}
_refresher_started = False
_refresher_lock = threading.Lock()


def start_token_refresher() -> dict:
    """Populates and returns VERTEX_KWARGS (custom_llm_provider/api_key/
    api_base), starting a daemon thread that refreshes api_key in place.
    Safe to call more than once -- only starts the thread the first time.
    """
    global _refresher_started
    project_id = os.environ["PROJECT_ID"].strip('"')
    region = os.environ.get("REGION", "global")

    with _refresher_lock:
        VERTEX_KWARGS["custom_llm_provider"] = "openai"
        VERTEX_KWARGS["api_base"] = _endpoint_url(project_id, region)
        VERTEX_KWARGS["api_key"] = _fetch_token()

        if _refresher_started:
            return VERTEX_KWARGS
        _refresher_started = True

        def _loop():
            while True:
                time.sleep(REFRESH_INTERVAL_SECONDS)
                try:
                    VERTEX_KWARGS["api_key"] = _fetch_token()
                except Exception:
                    # Keep the stale token rather than crash the refresher
                    # thread -- calls will start failing loudly on their
                    # own if it's genuinely expired, which is diagnosable;
                    # a dead refresher thread with no visible symptom
                    # until every call fails an hour later is not.
                    pass

        threading.Thread(target=_loop, daemon=True, name="vertex-token-refresher").start()

    return VERTEX_KWARGS
