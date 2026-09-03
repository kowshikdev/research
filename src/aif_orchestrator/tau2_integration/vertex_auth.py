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

REFRESH_INTERVAL_SECONDS = 10 * 60
# Tokens live ~60min in theory, but observed going bad (401
# ACCESS_TOKEN_TYPE_UNSUPPORTED, consistent across retries) well before
# that in a real sweep -- shorter interval bounds how long a bad token
# can strand a run before the background thread replaces it.


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
        # tau2's own generate() unconditionally sets num_retries to its
        # default, and litellm's internal retry wrapper has been observed
        # (with Groq, same symptom) to drop an explicit custom_llm_provider/
        # api_key override on retried attempts -- a single, no-retry-needed
        # call always succeeds, but any transient failure that triggers
        # litellm's own retry surfaces as "Missing credentials" on the
        # retried attempt. Disabling litellm's internal retry and relying
        # on tau2's task-level retry (which rebuilds the call from scratch,
        # not a litellm-internal retry) avoids the bug entirely.
        VERTEX_KWARGS["num_retries"] = 0

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


def overlay_live_kwargs(kwargs: dict) -> dict:
    """Returns kwargs with VERTEX_KWARGS's current values merged on top.

    Passing `dict(VERTEX_KWARGS)` into a tau2 TextRunConfig/agent gets
    deepcopied on the way in (confirmed: pydantic does not preserve
    dict object identity across model construction), which freezes the
    api_key at whatever value it had when that agent/config was built --
    for a domain/agent run spanning many tasks, later tasks silently
    keep using an old token even after the background thread has
    refreshed VERTEX_KWARGS. Called at the point of the actual
    completion() call (efe_agent.py's generate wrapper, and the
    NL-assertion-judge/user-simulator patches in register.py), this
    always applies whatever is currently live, regardless of how stale
    the kwargs threaded through tau2's config machinery have gone.
    """
    return {**kwargs, **VERTEX_KWARGS}


def force_refresh() -> None:
    """Fetches a new token immediately, outside the background thread's
    interval -- for use when a call just got a 401 and waiting up to
    REFRESH_INTERVAL_SECONDS for the next scheduled refresh isn't worth
    it (observed: a burst of ~5 consecutive real tasks all failed with
    AuthenticationError while the *same* gcloud-issued token, tested
    moments later with a direct curl call, worked fine -- a transient
    Vertex-side/OAuth-refresh hiccup, not a bad token per se, but cheap
    to route around by just forcing an immediate re-fetch)."""
    try:
        VERTEX_KWARGS["api_key"] = _fetch_token()
    except Exception:
        pass
