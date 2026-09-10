# Milestone 1 – Audio Processing & Transcription

## Task 1 – Whisper Transcription

Test the complete transcription workflow.

**Flow:**
- Upload Meeting Recording
- Process Audio
- Run Whisper
- Generate Transcript
- Display Transcript

---

## Task 2 – File Upload Validation

Verify:
- Audio/video upload works
- Multiple formats supported
- Invalid files are rejected
- Proper error messages are shown

---

## Task 3 – Transcript Validation

Transcript is generated correctly:
- Transcript is not empty
- Transcript matches the recording
- Transcript is saved correctly

---

## Task 4 – Streamlit Interface

Verify:
- File upload
- Transcribe button
- Processing status
- Transcript display

---

## Task 5 – Accuracy Testing

Test multiple recordings:
- Compare transcript with actual speech
- Check missing/incorrect words
- Achieve ≥90% transcription accuracy

---

---

# Milestone 1 – Text Ingestion & Baseline Sentiment

## Task 1 – Validate Text Ingestion Workflow

Test the complete text input process.

**Flow:**
- Create/enter text input
- Upload `.txt` file
- Upload `.csv` file
- Read input data
- Validate input format
- Pass valid text to preprocessing
- Handle empty or invalid inputs
- Verify that all supported input methods work correctly

---

## Task 2 – Preprocessing Validation

Verify that the text preprocessing pipeline works correctly.

**Check:**
- Tokenization
- Stop-word removal
- Lemmatization
- Noise filtering
- Special characters
- Punctuation
- Empty text
- Repeated spaces
- Different text lengths

---

## Task 3 – VADER Sentiment Validation

Verify the baseline sentiment analysis module.

**Check:**
- Positive sentiment detection
- Negative sentiment detection
- Neutral sentiment detection
- Sentiment compound score
- Positive score
- Negative score
- Neutral score
- Test VADER using different sample inputs

---

## Task 4 – Initial Emotion/Sentiment Report Validation

Generate the initial classification report using the sample text corpus.

---

## Task 5 – Complete Pipeline Integration Testing

- Create a GitHub repo: project name
- Get the MIT License
- Push the Milestone 1 code to that repo
- Public repository (not private)

---
---

# Milestone 2 – Meeting Intelligence Pipeline

## Task 1 – LLM Service Configuration

Select and harden the Gemini 3.5 Flash LLM service layer.

**Implemented in `gemini_summary.py`:**

- Reusable prompt templates extracted into named constants
  - `SYSTEM_INSTRUCTION` — anti-hallucination rules, priority/status rules
  - `build_user_message(transcript)` — wraps transcript in standard request
  - `build_full_prompt(transcript)` — combines both into the final API input
- Structured output via strict JSON schema (`RESPONSE_SCHEMA`) with
  `additionalProperties: false` at every level
- Input validation (`validate_transcript`):
  - Rejects non-string input with `TypeError`
  - Rejects transcripts shorter than 20 characters with `ValueError`
  - Truncates transcripts exceeding 800,000 characters at the last sentence
    boundary and surfaces a visible warning in the UI
- Retry and failure handling (`_call_with_retry`):
  - 3 attempts with exponential backoff (2s → 4s)
  - Retries on: timeout, connection error, rate limit (429), 503/502
  - Fails fast with `RuntimeError` on: auth errors, bad model ID, bad schema
- Output validation (`_normalise_response` + `_normalise_action_items`):
  - Validates `summary` is a string
  - Validates `key_points`, `topics`, `decisions`, `participants`,
    `action_items` are lists
  - Coerces invalid `priority` to `null`; invalid `status` to `"pending"`
  - Never crashes the UI — always returns a usable dict

**Model verified:** `gemini-3.5-flash` — confirmed GA stable at  
`https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash`

---

## Task 2 – Meeting Summarization Module

Extend the Gemini output with decisions and participants.

**New fields added to schema, system instruction, normalisation, and UI:**

- `decisions` — concrete decisions explicitly stated in the meeting
  (e.g. "Continue with the planned mobile launch").
  Empty list if none explicitly stated.
- `participants` — names explicitly mentioned or speaking in the transcript.
  Never invented. Empty list if none found.

**UI display (Step 6 panel):**
- 🎯 Key Decisions — bullet list below summary
- 🙋 Participants — colour-coded name pills

---

## Task 3 – Action Item Extraction Engine

Add `priority` and `status` to action items in both extraction paths.

**Gemini path (`gemini_summary.py`):**
- `priority` — `"high"` | `"medium"` | `"low"` | `null`
  - `"high"` only for explicit urgency language: urgent, ASAP, critical,
    by EOD, top priority, immediately
  - `"medium"` for near-term deadlines only (today/tomorrow/this week)
  - `null` otherwise — never guessed
- `status` — `"pending"` | `"completed"`
  - `"completed"` only if transcript explicitly states the task is done
  - `"pending"` in all other cases

**Deterministic path (`meeting_analyzer.py`):**
- `_HIGH_PRIORITY_RE` — regex for high-urgency words → `"high"`
- `_determine_priority(sentence)` — high → medium (date present) → null
- `status` always `"pending"` (a transcript records intent, not completion)

**UI display:**
- Coloured priority badges: 🔴 HIGH / 🟠 MEDIUM / 🟢 LOW
- Status badges: PENDING / COMPLETED
- Shared `_render_action_items()` helper used for both Python and Gemini paths

---

## Task 5 – Integration & Verification

- All three tasks run end-to-end in the existing Step 6 Streamlit panel
- Truncation warning tested by padding transcript past 800,000 characters
- Retry behaviour verified by temporarily invalidating the API key
- README and Docs updated to reflect Milestone 2 coverage
- Code committed and pushed to `https://github.com/ashiiifr/Milestone-01`
