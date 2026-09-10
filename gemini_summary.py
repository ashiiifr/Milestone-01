"""
gemini_summary.py
-----------------
Milestone 2 — Meeting Intelligence Pipeline
LLM service layer for structured meeting analysis via Google Gemini 3.5 Flash.

MODEL VERIFICATION (checked 2026-09-02 against official docs):
  • Model ID: "gemini-3.5-flash"  ← confirmed correct at
    https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash
    "Model code: gemini-3.5-flash"  (GA, stable, 1M input tokens)
  • SDK:  google-genai >= 2.0.0
  • Call: client.interactions.create(model=..., input=..., response_format=...)
  • Structured output shape: response_format={"type":"text",
                                              "mime_type":"application/json",
                                              "schema": <dict>}
  No changes to model ID or SDK call shape were required.

DESIGN
──────
All prompt text lives in named constants/functions (TASK 1).
Input is validated and truncated before any API call (TASK 1).
The API call is wrapped in a retry loop (TASK 1).
Output is fully validated and coerced (TASK 1).
Schema includes decisions, participants (TASK 2) and priority/status on
action items (TASK 3).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_ID = "gemini-3.5-flash"

# Gemini 3.5 Flash has a 1 M token input window.
# 1 token ≈ 4 characters on average.  We leave a large margin for the
# prompt wrapper and reserve output tokens: 800 000 chars is safe.
_MAX_TRANSCRIPT_CHARS = 800_000

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — PROMPT TEMPLATES
# All prompt text is defined here as named constants / small builder
# functions.  Editing prompt wording never requires touching call logic.
# ─────────────────────────────────────────────────────────────────────────────

# System instruction: processed before the user message.
# This is the primary anti-hallucination mechanism.
SYSTEM_INSTRUCTION: str = """You are a precise meeting analysis assistant.
Your ONLY source of information is the transcript provided by the user.

STRICT RULES — violating any of these is an error:
1. NEVER invent, guess, or assume information not explicitly stated in the transcript.
2. NEVER invent a person's name.
   - assigned_to MUST be null if no person is explicitly named for a task.
   - participants MUST contain only names explicitly spoken/written in the transcript.
3. NEVER invent a deadline. deadline MUST be null if none is mentioned.
4. NEVER invent a decision. decisions MUST contain only conclusions explicitly
   stated in the transcript (e.g. "we decided to …", "agreed to …"). Empty list if none.
5. NEVER add topics, key points, or action items not clearly present in the transcript.
6. priority: set ONLY when urgency is explicitly signalled by words like
   "urgent", "ASAP", "critical", "immediately", "by end of day", "EOD", "top priority".
   Use "high" for those. Use "medium" when a near-term deadline (today/tomorrow/
   this week) is mentioned. Otherwise return null. NEVER guess priority.
7. status: return "completed" ONLY if the transcript explicitly states the task
   is already done/finished/completed. Otherwise return "pending".
8. Return ONLY the JSON object matching the required schema. No other text.
9. If the transcript is empty or contains no meaningful content, return empty
   lists and an empty string for summary."""


def build_user_message(transcript: str) -> str:
    """Wrap the transcript in the standard analysis request message."""
    divider = "─" * 60
    return (
        f"MEETING TRANSCRIPT:\n"
        f"{divider}\n"
        f"{transcript.strip()}\n"
        f"{divider}\n\n"
        f"Analyse the transcript above and return the JSON object."
    )


def build_full_prompt(transcript: str) -> str:
    """Combine system instruction + user message into the single input string
    used by client.interactions.create()."""
    return f"{SYSTEM_INSTRUCTION}\n\n{build_user_message(transcript)}"


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 + 2 + 3 — STRICT JSON SCHEMA
# Defines every field the model may return. additionalProperties:false
# at every level prevents the model adding anything outside this shape.
# ─────────────────────────────────────────────────────────────────────────────

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # ── TASK 2 ─────────────────────────────────────────────────────────
        "summary": {
            "type": "string",
            "description": (
                "A single coherent paragraph summarising the meeting. "
                "Based strictly on the transcript. No invented content."
            ),
        },
        "key_points": {
            "type": "array",
            "description": "The most important points made during the meeting.",
            "items": {"type": "string"},
        },
        "decisions": {
            "type": "array",
            "description": (
                "Concrete decisions explicitly made during the meeting "
                "(e.g. 'Continue with the planned mobile launch'). "
                "Empty list if no decisions were explicitly stated."
            ),
            "items": {"type": "string"},
        },
        "participants": {
            "type": "array",
            "description": (
                "Names of people explicitly mentioned or speaking in the "
                "transcript. Do NOT invent names. Empty list if none found."
            ),
            "items": {"type": "string"},
        },
        "topics": {
            "type": "array",
            "description": "Broad subjects or themes discussed in the meeting.",
            "items": {"type": "string"},
        },
        # ── TASK 3 ─────────────────────────────────────────────────────────
        "action_items": {
            "type": "array",
            "description": (
                "Tasks or follow-ups explicitly mentioned. "
                "Do NOT invent tasks."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The exact task or action to be done.",
                    },
                    "assigned_to": {
                        "type": ["string", "null"],
                        "description": (
                            "Person assigned, exactly as named. "
                            "null if not explicitly named."
                        ),
                    },
                    "deadline": {
                        "type": ["string", "null"],
                        "description": (
                            "Deadline exactly as mentioned. "
                            "null if not mentioned."
                        ),
                    },
                    "priority": {
                        "type": ["string", "null"],
                        "description": (
                            "One of 'high', 'medium', 'low', or null. "
                            "Set 'high' only for explicit urgency words "
                            "(urgent/ASAP/critical/EOD/top priority). "
                            "Set 'medium' for near-term deadlines only. "
                            "null otherwise. NEVER guess."
                        ),
                        "enum": ["high", "medium", "low", None],
                    },
                    "status": {
                        "type": "string",
                        "description": (
                            "'completed' only if transcript explicitly states "
                            "task is done. Otherwise 'pending'."
                        ),
                        "enum": ["pending", "completed"],
                    },
                },
                "required": [
                    "task", "assigned_to", "deadline", "priority", "status"
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "summary", "key_points", "decisions", "participants",
        "topics", "action_items",
    ],
    "additionalProperties": False,
}

# Allowed values for output validation / coercion
_VALID_PRIORITIES = {"high", "medium", "low"}
_VALID_STATUSES   = {"pending", "completed"}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — INPUT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

# Minimum meaningful transcript length (characters).
# Anything shorter is likely a silence artefact, not a real meeting.
_MIN_TRANSCRIPT_CHARS = 20


def validate_transcript(transcript: Any) -> tuple[str, str | None]:
    """
    Validate and normalise the transcript before sending to Gemini.

    Returns:
        (cleaned_transcript, warning_message_or_None)

    Raises:
        TypeError  — input is not a string
        ValueError — input is too short to be meaningful
    """
    if not isinstance(transcript, str):
        raise TypeError(
            f"transcript must be a str, got {type(transcript).__name__}. "
            "Pass the text string returned by Whisper."
        )

    cleaned = transcript.strip()

    if len(cleaned) < _MIN_TRANSCRIPT_CHARS:
        raise ValueError(
            f"Transcript is too short ({len(cleaned)} chars, minimum "
            f"{_MIN_TRANSCRIPT_CHARS}). Nothing to analyse."
        )

    # ── Truncation guard ────────────────────────────────────────────────────
    # If the transcript exceeds the safe character budget we truncate at the
    # last sentence boundary before the limit and surface an honest warning.
    # We do NOT silently drop content or crash — the caller decides what to
    # show the user (see generate_meeting_summary return dict).
    warning: str | None = None
    if len(cleaned) > _MAX_TRANSCRIPT_CHARS:
        cutoff = cleaned.rfind(".", 0, _MAX_TRANSCRIPT_CHARS)
        if cutoff == -1:
            cutoff = _MAX_TRANSCRIPT_CHARS
        cleaned  = cleaned[:cutoff + 1].strip()
        warning  = (
            f"⚠️ Transcript truncated to {len(cleaned):,} characters "
            f"(original: {len(transcript.strip()):,} chars). "
            f"The model's context window limit was approached. "
            f"Analysis covers only the portion shown."
        )

    return cleaned, warning


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — OUTPUT VALIDATION & NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_action_items(raw: list) -> list[dict[str, Any]]:
    """
    Validate and coerce each action item against the schema.
    Invalid values are coerced to safe defaults rather than crashing.
    """
    result = []
    for item in raw:
        if not isinstance(item, dict) or "task" not in item:
            continue  # silently skip malformed items

        # priority — must be "high"/"medium"/"low" or null
        raw_priority = item.get("priority")
        if isinstance(raw_priority, str):
            priority = raw_priority.lower() if raw_priority.lower() in _VALID_PRIORITIES else None
        else:
            priority = None  # null or anything non-string → null

        # status — must be "pending" or "completed"; default to "pending"
        raw_status = item.get("status")
        if isinstance(raw_status, str) and raw_status.lower() in _VALID_STATUSES:
            status = raw_status.lower()
        else:
            status = "pending"

        result.append({
            "task":        str(item.get("task", "")).strip(),
            "assigned_to": item.get("assigned_to") or None,
            "deadline":    item.get("deadline")    or None,
            "priority":    priority,
            "status":      status,
        })
    return result


def _normalise_response(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate all top-level fields, coerce types to safe defaults.
    Never crashes the UI — always returns a usable dict.
    """
    # summary: must be a string
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)

    # list fields: must be lists of strings
    def _safe_str_list(key: str) -> list[str]:
        val = data.get(key, [])
        if not isinstance(val, list):
            return []
        return [str(x) for x in val if x]

    return {
        "summary":      summary,
        "key_points":   _safe_str_list("key_points"),
        "decisions":    _safe_str_list("decisions"),
        "participants": _safe_str_list("participants"),
        "topics":       _safe_str_list("topics"),
        "action_items": _normalise_action_items(data.get("action_items", [])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — CLIENT + RETRY LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _get_client():
    """Build and return a google-genai Client from GEMINI_API_KEY env var."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Add it to your .env file:  GEMINI_API_KEY=your_key_here\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    return genai.Client(api_key=api_key)


# Exceptions that are worth retrying (transient network / rate-limit errors).
# Everything else (auth, bad schema, etc.) should fail fast.
_RETRY_ATTEMPTS   = 3
_RETRY_BASE_DELAY = 2.0   # seconds; doubles each attempt


def _call_with_retry(client, prompt: str) -> str:
    """
    Call the Gemini API with exponential-backoff retry on transient errors.

    Returns the raw output_text string on success.
    Raises RuntimeError on permanent failure (auth, bad key, exhausted retries).
    Raises ValueError on schema / JSON errors (caller handles separately).
    """
    import httpx  # installed as google-genai dependency

    last_exc: Exception | None = None
    delay = _RETRY_BASE_DELAY

    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            interaction = client.interactions.create(
                model=_MODEL_ID,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": RESPONSE_SCHEMA,
                },
            )
            return interaction.output_text

        # ── Transient errors — retry ────────────────────────────────────────
        except (TimeoutError, ConnectionError) as exc:
            last_exc = exc
        except Exception as exc:
            # httpx transport errors and google-genai rate-limit errors
            # surface as generic exceptions; check message to distinguish.
            msg = str(exc).lower()
            transient_signals = (
                "timeout", "rate limit", "rate_limit", "429",
                "503", "502", "connection", "temporarily",
            )
            if any(sig in msg for sig in transient_signals):
                last_exc = exc
            else:
                # Permanent failure — auth error, bad model ID, etc.
                raise RuntimeError(
                    f"Gemini API call failed (attempt {attempt}): {exc}"
                ) from exc

        if attempt < _RETRY_ATTEMPTS:
            time.sleep(delay)
            delay *= 2   # exponential backoff

    raise RuntimeError(
        f"Gemini API call failed after {_RETRY_ATTEMPTS} attempts. "
        f"Last error: {last_exc}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_meeting_summary(transcript: str) -> dict[str, Any]:
    """
    Call Gemini 3.5 Flash and return a fully validated structured analysis.

    Args:
        transcript: Plain-text meeting transcript from Whisper.

    Returns:
        {
            "summary":        str,
            "key_points":     [str, ...],
            "decisions":      [str, ...],       # TASK 2
            "participants":   [str, ...],       # TASK 2
            "topics":         [str, ...],
            "action_items":   [{               # TASK 3
                "task":        str,
                "assigned_to": str | null,
                "deadline":    str | null,
                "priority":    "high"|"medium"|"low"|null,
                "status":      "pending"|"completed",
            }, ...],
            "_truncation_warning": str | None,  # present if transcript was cut
        }

    Raises:
        TypeError   — transcript is not a string
        ValueError  — transcript too short, or Gemini returned invalid JSON
        RuntimeError — API key missing, auth failure, or retries exhausted
    """
    # ── Input validation + truncation guard ──────────────────────────────────
    cleaned, trunc_warning = validate_transcript(transcript)

    # ── Build prompt from templates ───────────────────────────────────────────
    prompt = build_full_prompt(cleaned)

    # ── API call with retry ───────────────────────────────────────────────────
    client   = _get_client()
    raw_text = _call_with_retry(client, prompt)

    # ── Parse JSON ────────────────────────────────────────────────────────────
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned invalid JSON.\nRaw response (first 500 chars):\n"
            f"{raw_text[:500]}"
        ) from exc

    # ── Full output validation + normalisation ────────────────────────────────
    result = _normalise_response(data)

    # Surface truncation warning in the result dict so app.py can display it
    result["_truncation_warning"] = trunc_warning

    return result
