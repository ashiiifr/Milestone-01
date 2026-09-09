"""
gemini_summary.py
-----------------
Natural-language meeting summary via the Google Gemini API.

WHY GEMINI ONLY FOR THE SUMMARY?
─────────────────────────────────
Every other extraction task in this project (topics, key points, action items,
dates, speaker count) is *deterministic* — the answer is either in the text or
it isn't.  Python regex and NLP libraries can handle those reliably and for
free, with zero hallucination risk.

A *summary* is different.  It requires the model to:
  • Read the whole transcript as a coherent narrative.
  • Decide what is important vs. incidental.
  • Express that in fluent, readable English.

That is genuinely a natural-language understanding task where an LLM adds real
value.  We still constrain the output tightly (see below) to prevent invention.

HOW WE PREVENT HALLUCINATION
─────────────────────────────
1. STRICT SYSTEM INSTRUCTION — the system prompt explicitly forbids the model
   from inventing names, tasks, or deadlines not present in the transcript.

2. STRUCTURED JSON OUTPUT — we pass a `response_format` with a hard JSON schema
   via the official google-genai v2 SDK.  The model CANNOT return free text; it
   must return a JSON object that matches the schema exactly.  Invalid fields
   are rejected at the API level.

3. SCHEMA DESIGN — every optional field (assigned_to, deadline) uses a nullable
   type `["string", "null"]`.  This forces the model to explicitly return null
   rather than guessing.

SDK USED: google-genai >= 2.0.0
  • Import:  from google import genai
  • Client:  genai.Client()  (reads GEMINI_API_KEY from environment)
  • Structured output via: response_format={"type":"text",
                                            "mime_type":"application/json",
                                            "schema": <dict>}
  Verified against: https://ai.google.dev/gemini-api/docs/interactions/structured-output.md
  (Last checked: 2026-09-02)
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

# Load GEMINI_API_KEY from .env if present (never hard-coded here)
load_dotenv()

# ── Model configuration (as specified in the task) ───────────────────────────
_MODEL_ID = "gemini-3.5-flash"

# ── Strict JSON schema for Gemini's response ─────────────────────────────────
#
# WHY A STRICT SCHEMA?
# When you hand an LLM a blank sheet and say "summarise this", it will add
# headers, bullet points, disclaimers, and invented details.  A JSON schema
# acts like a form — the model can only fill in the fields we define.  The
# API enforces this at the transport level: if the model tries to return
# something outside the schema, the request fails rather than silently
# returning garbage.
#
# Fields:
#   summary      – one coherent paragraph in plain English
#   key_points   – list of the most important statements from the meeting
#   topics       – list of broad subject areas discussed
#   action_items – list of tasks, each with optional assigned_to and deadline
#
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "A single coherent paragraph summarising the meeting. "
                "Must be based strictly on what was said in the transcript. "
                "Do not add any information not present in the transcript."
            ),
        },
        "key_points": {
            "type": "array",
            "description": "The most important points made during the meeting.",
            "items": {"type": "string"},
        },
        "topics": {
            "type": "array",
            "description": "The broad subjects or themes discussed in the meeting.",
            "items": {"type": "string"},
        },
        "action_items": {
            "type": "array",
            "description": (
                "Tasks or follow-ups explicitly mentioned. "
                "Do NOT invent tasks. Only include tasks clearly stated."
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
                            "The person assigned, exactly as named in the transcript. "
                            "Return null if no person is explicitly named."
                        ),
                    },
                    "deadline": {
                        "type": ["string", "null"],
                        "description": (
                            "The deadline exactly as mentioned in the transcript. "
                            "Return null if no deadline is mentioned."
                        ),
                    },
                },
                "required": ["task", "assigned_to", "deadline"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "key_points", "topics", "action_items"],
    "additionalProperties": False,
}

# ── System instruction sent before the transcript ────────────────────────────
#
# This is the primary anti-hallucination mechanism.  It is a system-level
# instruction (processed before the user input) so the model treats it as
# a hard constraint, not a polite request.
#
_SYSTEM_INSTRUCTION = """You are a precise meeting analysis assistant.
Your ONLY source of information is the transcript provided by the user.

STRICT RULES — violating any of these is an error:
1. NEVER invent, guess, or assume information not explicitly stated in the transcript.
2. NEVER invent a person's name. If no name is given for a task, assigned_to MUST be null.
3. NEVER invent a deadline. If no date or time is given for a task, deadline MUST be null.
4. NEVER add topics, key points, or action items that are not clearly present in the transcript.
5. The summary MUST be based solely on the transcript content.
6. Return ONLY the JSON object matching the required schema. No other text.
7. If the transcript is empty or contains no meaningful content, return empty lists
   and an empty string for summary."""


def _get_client():
    """
    Build and return a google-genai Client.

    The API key is read from the GEMINI_API_KEY environment variable.
    Raises a clear RuntimeError if the key is missing so the Streamlit UI
    can display a helpful message rather than a cryptic SDK error.
    """
    from google import genai  # imported lazily so the module loads without the key

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Add it to your .env file:  GEMINI_API_KEY=your_key_here\n"
            "Get a key at: https://aistudio.google.com/app/apikey"
        )
    return genai.Client(api_key=api_key)


def generate_meeting_summary(transcript: str) -> dict[str, Any]:
    """
    Call Gemini 3.5 Flash to produce a structured meeting summary.

    Args:
        transcript: The full plain-text transcript string.

    Returns:
        A dict matching the schema above:
        {
            "summary":      str,
            "key_points":   [str, ...],
            "topics":       [str, ...],
            "action_items": [{"task": str,
                              "assigned_to": str | null,
                              "deadline":    str | null}, ...],
        }

    Raises:
        RuntimeError:  GEMINI_API_KEY missing or API call fails.
        ValueError:    Response JSON does not match the expected schema.
    """
    if not transcript or not transcript.strip():
        return {
            "summary": "",
            "key_points": [],
            "topics": [],
            "action_items": [],
        }

    client = _get_client()

    # Compose the user message: system instruction + transcript
    user_input = (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        f"MEETING TRANSCRIPT:\n"
        f"{'─' * 60}\n"
        f"{transcript.strip()}\n"
        f"{'─' * 60}\n\n"
        f"Analyse the transcript above and return the JSON object."
    )

    # ── API call — structured output enforced by response_format ─────────────
    #
    # The `response_format` parameter tells the API to:
    #   1. Return content-type application/json
    #   2. Validate the response against our schema before returning it
    #
    # This is the official google-genai v2 Interactions API pattern:
    #   client.interactions.create(model=..., input=..., response_format=...)
    #
    interaction = client.interactions.create(
        model=_MODEL_ID,
        input=user_input,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": _RESPONSE_SCHEMA,
        },
    )

    raw_text = interaction.output_text

    # ── Parse and validate the JSON ───────────────────────────────────────────
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned invalid JSON.\n"
            f"Raw response:\n{raw_text[:500]}"
        ) from exc

    # Ensure all required top-level keys are present (defensive check)
    for key in ("summary", "key_points", "topics", "action_items"):
        if key not in data:
            data[key] = [] if key != "summary" else ""

    # Normalise action items: ensure each has all three keys
    normalised_items = []
    for item in data.get("action_items", []):
        if isinstance(item, dict) and "task" in item:
            normalised_items.append({
                "task":        str(item.get("task", "")),
                "assigned_to": item.get("assigned_to") or None,
                "deadline":    item.get("deadline") or None,
            })
    data["action_items"] = normalised_items

    return data
