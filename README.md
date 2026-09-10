# 🎙️ Meeting Transcriber & Analyser

> **Milestone 1 & 2** — Audio Processing, Transcription & Meeting Intelligence Pipeline  
> Powered by **OpenAI Whisper** + **Google Gemini 3.5 Flash**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)](https://streamlit.io)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)](https://github.com/openai/whisper)
[![Gemini](https://img.shields.io/badge/Google-Gemini%203.5%20Flash-orange)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Full Output Structure](#-full-output-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the App](#-running-the-app)
- [Using the App — Step by Step](#-using-the-app--step-by-step)
- [Demo Audio Files](#-demo-audio-files)
- [Module Reference](#-module-reference)
- [Design Decisions](#-design-decisions)
- [Error Handling](#-error-handling)
- [Security](#-security)
- [Milestone Tasks](#-milestone-tasks)

---

## 🌟 Overview

A full end-to-end **meeting transcription and intelligence pipeline** that takes any audio or video recording, transcribes it using OpenAI Whisper, and then extracts structured intelligence in two layers:

**Layer 1 — Python deterministic extraction (offline, no API key)**  
Uses regex, TF-IDF, and NLTK to extract speaker count, topics, key points, action items (with priority & status), assigned persons, and deadlines directly from the text. Reproducible, zero hallucination risk.

**Layer 2 — Gemini AI meeting intelligence (requires API key)**  
Google Gemini 3.5 Flash produces a fluent summary, key decisions, participant names, and richer action items — all enforced by a strict JSON schema so the model cannot invent anything not present in the transcript. The LLM service layer is hardened with input validation, a truncation guard, retry logic, and full output validation.

Everything is accessible through a clean six-step **Streamlit web interface**.

---

## ✨ Features

### 🎙️ Transcription
- Upload audio/video in **7 formats**: `mp3` `wav` `m4a` `mp4` `ogg` `flac` `webm`
- Powered by **OpenAI Whisper** — runs fully offline on your machine
- Choose model size: `tiny` → `base` → `small` → `medium` → `large`
- Full transcript with **timed segment timestamps** (start → end)
- Save to `.txt` or download directly from the browser
- **Bundled `ffmpeg`** via `imageio-ffmpeg` — no system install required

### 🔍 Python Analysis (offline, no key needed)

| What | How |
|---|---|
| 👥 Speaker count estimate | KMeans on Whisper segment acoustic features |
| 📌 Key topics | TF-IDF over noun-phrases (NLTK POS chunking) |
| 💡 Key discussion points | TF-IDF sentence scoring (TextRank-lite) |
| ✅ Action items | Regex action-verb vocabulary + NLTK NER |
| 👤 Assigned person | NLTK PERSON entity + Title-Case heuristic |
| 📅 Deadlines / dates | Regex over 8 date/time pattern families |
| 🔴🟠 Priority | Urgency keyword regex (`high`) · date present (`medium`) · else `null` |
| ⏳ Status | Always `"pending"` — transcript records intent, not completion |

### ✨ Gemini AI Intelligence (requires `GEMINI_API_KEY`)

| What | How |
|---|---|
| 📝 Meeting summary | Fluent natural-language paragraph |
| 🎯 Key decisions | Explicit decisions only — empty list if none stated |
| 🙋 Participants | Names mentioned in transcript — never invented |
| 💡 Key points | Most important statements |
| 📌 Topics | Broad subjects discussed |
| ✅ Action items | Task + assigned person + deadline + priority + status |

### 🛡️ Anti-Hallucination Safeguards
- `additionalProperties: false` on every schema object — model cannot add fields
- `"type": ["string", "null"]` on optional fields — model must return `null`, not guess
- System instruction explicitly forbids inventing names, decisions, tasks, priorities, or dates
- `response_format` with `mime_type: application/json` enforced at API transport level
- Every field re-validated in Python after parsing — invalid values coerced to safe defaults

### 🔒 LLM Service Hardening (Milestone 2)
- Prompt text lives in named constants — change wording without touching call logic
- Input rejected if not a string or shorter than 20 characters
- Transcripts over **800,000 characters** cut at last sentence boundary with a visible UI warning
- **3 retry attempts** with exponential backoff (2 s → 4 s) on timeout/rate-limit/503
- Fast fail with clear message on auth errors — no raw stack traces shown to user

---

## 📁 Project Structure

```
Milestone 01/
│
├── app.py                    # Streamlit web app — main entry point (6 steps)
├── audio_processor.py        # File validation & temp file handling
├── transcriber.py            # OpenAI Whisper wrapper + bundled ffmpeg patch
├── meeting_analyzer.py       # Python deterministic NLP extraction
├── gemini_summary.py         # Gemini 3.5 Flash LLM service layer
├── generate_demo_audio.py    # Script to regenerate demo MP3 files via gTTS
│
├── demo_audio/               # Pre-generated demo recordings
│   ├── demo_10s.mp3          # ~10 seconds
│   ├── demo_30s.mp3          # ~30 seconds
│   ├── demo_1m.mp3           # ~1 minute
│   ├── demo_2m.mp3           # ~2 minutes
│   └── demo_5m.mp3           # ~5 minutes
│
├── Docs/
│   └── milestones.md         # Full task breakdown — Milestone 1 & 2
│
├── requirements.txt          # All Python dependencies
├── .env.example              # API key template (safe to commit)
├── .env                      # Your real key — NEVER committed (.gitignore)
├── .gitignore                # Excludes .env, myenv/, __pycache__, *.pt …
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## ⚙️ How It Works

```
┌────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT UI                              │
│                       (app.py — 6 steps)                          │
└────────────────────────────┬───────────────────────────────────────┘
                             │
             ┌───────────────▼───────────────┐
             │       Step 1 & 2              │
             │     audio_processor.py        │
             │  • Validate format + size     │
             │  • Save to temp file          │
             └───────────────┬───────────────┘
                             │
             ┌───────────────▼───────────────┐
             │          Step 3               │
             │        transcriber.py         │
             │  • Load Whisper model         │
             │  • ffmpeg decode (bundled)    │
             │  • Returns text + segments    │
             └────────┬──────────┬───────────┘
                      │          │
        ┌─────────────▼──┐  ┌────▼────────────────────┐
        │    Step 6a      │  │        Step 6b           │
        │ meeting_        │  │    gemini_summary.py     │
        │ analyzer.py     │  │                          │
        │                 │  │  validate_transcript()   │
        │ • Speaker count │  │  build_full_prompt()     │
        │ • Topics        │  │  _call_with_retry()      │
        │ • Key points    │  │  → gemini-3.5-flash      │
        │ • Action items  │  │  _normalise_response()   │
        │   + priority    │  │                          │
        │   + status      │  │  Returns:                │
        │ • All dates     │  │  • summary               │
        │                 │  │  • decisions             │
        │ No API needed   │  │  • participants          │
        └────────┬────────┘  │  • key_points            │
                 │           │  • topics                │
                 │           │  • action_items          │
                 │           │    + priority + status   │
                 │           └────┬────────────────────-┘
                 │                │
             ┌───▼────────────────▼───┐
             │    Step 6 Display       │
             │  Priority + status      │
             │  badges, decisions,     │
             │  participants, topics   │
             └─────────────────────────┘
```

---

## 📦 Full Output Structure

### Python Analysis (`analyze_transcript`)

```python
{
    "speaker_count":  int,          # estimated number of distinct voices
    "topics":         [str, ...],   # top 8 noun-phrase topics (TF-IDF)
    "key_points":     [str, ...],   # top 6 sentences by TF-IDF score
    "action_items": [{
        "task":        str,
        "assigned_to": str | None,
        "deadline":    str | None,
        "priority":    "high" | "medium" | None,  # keyword heuristic
        "status":      "pending",                 # always pending
    }, ...],
    "all_deadlines":  [str, ...],   # all date expressions found
}
```

### Gemini AI Analysis (`generate_meeting_summary`)

```python
{
    "summary":      str,            # fluent meeting summary paragraph
    "key_points":   [str, ...],     # most important statements
    "decisions":    [str, ...],     # explicit decisions made (empty if none)
    "participants": [str, ...],     # names mentioned (never invented)
    "topics":       [str, ...],     # broad subjects discussed
    "action_items": [{
        "task":        str,
        "assigned_to": str | None,  # null if no name stated
        "deadline":    str | None,  # null if no date stated
        "priority":    "high" | "medium" | "low" | None,  # null if not explicit
        "status":      "pending" | "completed",
    }, ...],
    "_truncation_warning": str | None,  # set if transcript was cut
}
```

---

## 📦 Prerequisites

- **Python 3.10 or higher**
- A **Google Gemini API key** — free tier available, needed only for the AI layer
  - Get one at: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

> `ffmpeg` is bundled automatically via `imageio-ffmpeg`. No system install needed.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ashiiifr/Milestone-01.git
cd "Milestone-01"
```

### 2. Create and activate a virtual environment

```powershell
# Windows (PowerShell)
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python -m venv myenv
source myenv/bin/activate
```

### 3. Install all dependencies

```powershell
pip install -r requirements.txt
```

**What gets installed:**

| Package | Purpose |
|---------|---------|
| `openai-whisper` | Speech-to-text transcription |
| `ffmpeg-python` | ffmpeg Python bindings |
| `imageio[ffmpeg]` | Bundled ffmpeg binary |
| `streamlit` | Web interface |
| `google-genai>=2.0.0` | Gemini API SDK (v2) |
| `nltk` | Tokenisation, POS tagging, NER |
| `scikit-learn` | TF-IDF vectoriser, KMeans clustering |
| `scipy` | Required by scikit-learn |
| `python-dotenv` | Loads `.env` file |
| `pydantic` | Data validation (google-genai dependency) |
| `httpx` | HTTP client (google-genai dependency) |

### 4. Download NLTK data (once only)

```powershell
.\myenv\Scripts\python.exe -c "
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
print('Done')
"
```

---

## 🔑 Configuration

### Gemini API key

```powershell
# 1. Copy the template
Copy-Item .env.example .env

# 2. Open .env and set your key
# GEMINI_API_KEY=your_actual_key_here
```

```env
# .env
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

> ⚠️ `.env` is in `.gitignore` — it is **never committed to Git**.  
> `.env.example` has no real key and is safe to share.

You can also paste the key directly in the **Streamlit sidebar** for quick testing without editing the file.

The Gemini key is **only required for the AI summary layer**. All Python extraction runs completely offline.

### Environment variable reference

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | For AI summary only | Your Google Gemini API key |

---

## ▶️ Running the App

```powershell
.\myenv\Scripts\streamlit.exe run app.py
```

Browser opens automatically at **http://localhost:8501**.  
Stop with `Ctrl + C`.

---

## 🖥️ Using the App — Step by Step

### Step 1 · Upload Meeting Recording
Click the file uploader and choose your audio or video file.

| Supported formats | `mp3` `wav` `m4a` `mp4` `ogg` `flac` `webm` |
|---|---|
| Max file size | 500 MB |

### Step 2 · Process Audio
The app validates format and size, then displays the filename and file size.

### Step 3 · Run Whisper
Select a model in the sidebar, then click **🚀 Transcribe**.

| Model | Speed | Accuracy | Approx. VRAM |
|-------|-------|----------|-------------|
| `tiny` | Fastest | Lower | ~1 GB |
| `base` | Fast | Good | ~1 GB |
| `small` | Medium | Better | ~2 GB |
| `medium` | Slow | High | ~5 GB |
| `large` | Slowest | Best | ~10 GB |

First run downloads the model weights automatically (~140 MB for `base`).

### Step 4 · View Transcript
Full transcript text with:
- Language detected
- Model used
- Segment count
- Expandable **🕐 View timed segments** panel (start → end timestamps)

### Step 5 · Save / Download
- **💾 Save to file** — write to a `.txt` path on your machine
- **⬇️ Download** — download directly from the browser

### Step 6 · Meeting Analysis

#### 🧠 Python Analysis (no API key)
Click **🧠 Run Analysis**. Results appear immediately:

| Section | What you see |
|---------|-------------|
| 👥 Estimated Speakers | Metric card with count |
| 📌 Key Topics | Colour-coded pill badges |
| 💡 Key Discussion Points | Numbered sentence list |
| ✅ Action Items | Cards with task, assigned-to, deadline, 🔴/🟠/🟢 priority badge, PENDING/COMPLETED status badge |
| 📅 All Dates/Deadlines | Expandable list of every date expression found |

#### ✨ Gemini AI Intelligence (requires API key)
Click **🚀 Generate AI Summary**. Gemini 3.5 Flash produces:

| Section | What you see |
|---------|-------------|
| 📝 Summary | Fluent paragraph covering the full meeting |
| 🎯 Key Decisions | Bullet list of explicit decisions made |
| 🙋 Participants | Pink name pills for everyone mentioned |
| 💡 Key Points | Bullet list of important statements |
| 📌 Topics | Inline code-style topic labels |
| ✅ Action Items | Same card layout as Python path with all 5 fields |
| 🔎 Raw JSON | Expandable code block showing exact Gemini response |

**If the transcript exceeds 800,000 characters:** a yellow warning banner appears showing how many characters were kept.  
**If the API key is missing or invalid:** a clear error message is shown — no raw stack trace.

---

## 🎵 Demo Audio Files

Ready-to-use test recordings generated with Google Text-to-Speech.

| File | Duration | Content |
|------|----------|---------|
| `demo_10s.mp3` | ~10 s | Short welcome message |
| `demo_30s.mp3` | ~30 s | Brief meeting intro |
| `demo_1m.mp3` | ~1 min | Agenda overview with speaker introductions |
| `demo_2m.mp3` | ~2 min | Sprint update, features, next steps |
| `demo_5m.mp3` | ~5 min | Full project update — architecture, testing, roadmap |

To regenerate them (requires internet connection for gTTS):

```powershell
.\myenv\Scripts\python.exe generate_demo_audio.py
```

---

## 📖 Module Reference

### `app.py`
Streamlit web application — entry point.

- `set_page_config`, title, sidebar (Whisper model selector + Gemini key input)
- Steps 1–5: upload → validate → transcribe → display → save
- Step 6a: Python analysis panel (`analyze_transcript`)
- Step 6b: Gemini AI panel (`generate_meeting_summary`)
- `_render_action_items(items)` — shared helper rendering priority/status badges for both paths
- `_priority_badge(priority)` / `_status_badge(status)` — coloured HTML badge generators

---

### `audio_processor.py`
File validation and temporary storage.

| Function | Returns | Description |
|----------|---------|-------------|
| `validate_audio_file(file)` | `(bool, str)` | Checks extension against 7 allowed types, enforces 500 MB limit |
| `save_uploaded_file(file)` | `str` (path) | Writes uploaded bytes to a named temp file |
| `cleanup_temp_file(path)` | `None` | Silently deletes temp file after processing |

Supported extensions: `.mp3` `.wav` `.m4a` `.mp4` `.ogg` `.flac` `.webm`

---

### `transcriber.py`
OpenAI Whisper wrapper with bundled ffmpeg support.

| Symbol | Description |
|--------|-------------|
| `WHISPER_MODELS` | `["tiny", "base", "small", "medium", "large"]` |
| `transcribe_audio(path, model_size)` | Loads model, runs transcription, returns `{text, segments, language, model, file}` |
| `save_transcript(text, path)` | Writes transcript string to `.txt` |
| `_ensure_ffmpeg()` | Finds bundled ffmpeg via `imageio_ffmpeg`; monkey-patches `whisper.audio.load_audio` to use the full binary path instead of relying on system PATH |
| `_FFMPEG_EXE` | Resolved ffmpeg path (set at import time) |

**Why the ffmpeg patch?** Whisper hardcodes `"ffmpeg"` in a subprocess call. On Windows without a system ffmpeg, this fails. The patch replaces that bare name with the absolute path to the `imageio_ffmpeg` binary so the app works out of the box.

---

### `meeting_analyzer.py`
All Python-only, deterministic NLP extraction. No API calls. Same input always produces the same output.

| Symbol | Description |
|--------|-------------|
| `_ACTION_VERBS` | Compiled regex — 30+ task-indicating verbs |
| `_DATE_RE` | Compiled regex — 8 date/time pattern families |
| `_HIGH_PRIORITY_RE` | Compiled regex — explicit urgency words |
| `_determine_priority(sentence)` | `"high"` for urgency words · `"medium"` if date present · `None` otherwise |
| `estimate_speaker_count(segments)` | KMeans (k=1–6) on `[duration, position, avg_logprob, words_per_sec]` — knee detection picks best k |
| `extract_topics(transcript)` | NLTK noun-phrase extraction → TF-IDF over phrases → top 8 |
| `extract_key_points(transcript)` | TF-IDF sentence matrix → sum of weights → top 6 in document order |
| `extract_action_items(transcript)` | Sentence filter by `_ACTION_VERBS` → NER person + deadline regex + `_determine_priority` + `status="pending"` |
| `extract_deadlines(transcript)` | All `_DATE_RE` matches, deduplicated |
| `analyze_transcript(transcript, segments)` | Calls all of the above, returns combined dict |

**Return shape of `analyze_transcript`:**
```python
{
    "speaker_count":  int,
    "topics":         [str],
    "key_points":     [str],
    "action_items":   [{"task": str, "assigned_to": str|None,
                        "deadline": str|None, "priority": str|None,
                        "status": "pending"}],
    "all_deadlines":  [str],
}
```

---

### `gemini_summary.py`
Gemini 3.5 Flash LLM service layer. Hardened for production use.

**Constants / templates:**

| Symbol | Type | Description |
|--------|------|-------------|
| `_MODEL_ID` | `str` | `"gemini-3.5-flash"` — verified GA at ai.google.dev |
| `_MAX_TRANSCRIPT_CHARS` | `int` | `800_000` — safe budget for 1M-token context window |
| `_MIN_TRANSCRIPT_CHARS` | `int` | `20` — minimum meaningful transcript length |
| `SYSTEM_INSTRUCTION` | `str` | Anti-hallucination rules + priority/status rules |
| `build_user_message(transcript)` | `str` | Wraps transcript in standard request message |
| `build_full_prompt(transcript)` | `str` | `SYSTEM_INSTRUCTION + build_user_message(transcript)` |
| `RESPONSE_SCHEMA` | `dict` | Full JSON schema with `additionalProperties: false` everywhere |

**Functions:**

| Function | Raises | Description |
|----------|--------|-------------|
| `validate_transcript(transcript)` | `TypeError`, `ValueError` | Type check · min length · truncation guard → returns `(cleaned, warning)` |
| `_get_client()` | `RuntimeError` | Builds `genai.Client` from `GEMINI_API_KEY` env var |
| `_call_with_retry(client, prompt)` | `RuntimeError` | 3 attempts, exponential backoff, transient vs permanent error discrimination |
| `_normalise_action_items(raw)` | — | Validates/coerces each action item; skips malformed entries |
| `_normalise_response(data)` | — | Validates all top-level fields; coerces types to safe defaults |
| `generate_meeting_summary(transcript)` | `TypeError`, `ValueError`, `RuntimeError` | Public entry — validate → prompt → retry → parse → normalise → return |

**JSON schema summary:**
```
{
  summary:      string
  key_points:   [string]
  decisions:    [string]        ← explicit decisions only
  participants: [string]        ← named persons only
  topics:       [string]
  action_items: [{
    task:        string
    assigned_to: string | null
    deadline:    string | null
    priority:    "high"|"medium"|"low"|null
    status:      "pending"|"completed"
  }]
}
```

---

### `generate_demo_audio.py`
Standalone script. Creates `demo_audio/*.mp3` using gTTS.  
Five scripts of increasing length (~10s, 30s, 1m, 2m, 5m) covering a realistic meeting scenario.

---

## 🧠 Design Decisions

### Python for extraction, Gemini only for summarisation

Every extraction task (dates, action verbs, named entities, TF-IDF scores) has a **deterministic correct answer** — either the pattern is in the text or it isn't. Python handles this reliably, cheaply, and without any risk of the model inventing content.

A **meeting summary** is different. It requires reading a transcript as a coherent narrative, deciding what matters, and expressing that fluently. That's a genuine natural-language understanding task where an LLM adds value. But we still constrain the output tightly.

**Decisions and participants** are also handled by Gemini rather than Python because:
- Detecting that "we agreed to proceed" is a *decision* requires semantic understanding
- Participant lists may come from conversational context ("As I mentioned to Sarah earlier…") that pure NER misses

### Why a strict JSON schema rather than prompt-only constraints?

A prompt instruction like "don't invent names" is a *polite request*. A JSON schema with `additionalProperties: false` and nullable types is a *hard constraint enforced at the API transport layer*. The model cannot return a field we didn't define, and it must use `null` for optional fields rather than guessing. We then re-validate the parsed JSON in Python as a second line of defence.

### Why KMeans for speaker count?

Full diarisation (attributing segments to specific speakers) requires dedicated audio models like pyannote. KMeans on Whisper's own segment metadata — duration, position in the recording, log-probability, and words-per-second — gives a lightweight and offline estimate of *how many* distinct voices are present. It's not perfect, but it's honest, fast, and requires no extra downloads.

### Why truncate at 800,000 characters?

Gemini 3.5 Flash has a 1M token input window. 1 token ≈ 4 characters, giving ~4M char capacity. But the prompt wrapper, system instruction, and output tokens all consume budget. 800,000 characters leaves a comfortable margin for a 5-hour meeting while keeping costs predictable. The truncation happens at the last sentence boundary before the limit, and the user sees an honest warning rather than a silent content drop or a crash.

### Why `status: "pending"` always in the Python path?

A meeting transcript records what participants *intend* to do — "Sarah will send the report by Friday." We have no way of knowing from the text alone whether that task was later completed. Setting `"pending"` is the honest default. The Gemini path can set `"completed"` only when the transcript *explicitly* says a task is done (e.g., "John has already sent the report"), which is a semantic judgement the LLM can make but a regex cannot.

---

## ⚠️ Error Handling

| Situation | What happens |
|-----------|-------------|
| Invalid file format | Red error banner in Step 2, `st.stop()` |
| File too large (> 500 MB) | Red error banner in Step 2, `st.stop()` |
| Whisper transcription fails | Red error banner in Step 3 |
| `GEMINI_API_KEY` not set | `RuntimeError` with setup instructions, shown as red banner |
| API auth failure / bad key | `RuntimeError`, shown as red banner, no retry |
| Rate limit / timeout / 503 | Retried up to 3 times with backoff; failure shown as red banner |
| Transcript non-string | `TypeError` shown as red banner |
| Transcript < 20 chars | `ValueError` shown as red banner |
| Transcript > 800k chars | Yellow warning banner; analysis continues on truncated text |
| Gemini returns invalid JSON | `ValueError` shown as red banner |
| Gemini returns wrong types | Values coerced to safe defaults silently; UI never crashes |

---

## 🔐 Security

- **API key never committed.** `.env` is in `.gitignore`. The key is only ever read from the environment variable `GEMINI_API_KEY`.
- **`.env.example` is safe to commit.** It contains only the variable name with a placeholder value.
- **No secret in source code.** `_get_client()` reads from `os.environ` — the key is never hardcoded.
- **No user data sent externally beyond Gemini.** The transcript is sent to the Gemini API only when the user explicitly clicks "Generate AI Summary". All Python analysis is fully local.
- **Temp files cleaned up.** Uploaded audio is saved to a temp file for Whisper processing and deleted immediately after transcription regardless of success or failure (`finally` block in `app.py`).

---

## 📋 Milestone Tasks

Full task details in [`Docs/milestones.md`](Docs/milestones.md).

### Milestone 1 — Audio Processing & Transcription
- [x] Task 1 — Whisper transcription workflow (upload → validate → transcribe → display → save)
- [ ] Task 2 — File upload validation testing
- [ ] Task 3 — Transcript validation testing
- [ ] Task 4 — Streamlit interface verification
- [ ] Task 5 — Accuracy testing (≥ 90% WER target)

### Milestone 1 — Text Ingestion & Baseline Sentiment
- [ ] Task 1 — Text ingestion workflow
- [ ] Task 2 — Preprocessing validation
- [ ] Task 3 — VADER sentiment validation
- [ ] Task 4 — Initial emotion/sentiment report
- [x] Task 5 (partial) — GitHub repo created, MIT license, code pushed

### Milestone 1 — Meeting Analysis (Task 10)
- [x] Speaker count estimation (KMeans on segment features)
- [x] Topic extraction (TF-IDF noun-phrases)
- [x] Key discussion points (TF-IDF sentence scoring)
- [x] Action item extraction (regex + NLTK NER)
- [x] Assigned person detection (NLTK PERSON entity + heuristic)
- [x] Deadline/date extraction (regex — 8 pattern families)
- [x] Gemini 3.5 Flash structured summary
- [x] Full Streamlit Step 6 integration

### Milestone 2 — Meeting Intelligence Pipeline
- [x] **Task 1 — LLM service hardening**
  - [x] Prompt templates as named constants (`SYSTEM_INSTRUCTION`, `build_user_message`, `build_full_prompt`)
  - [x] Input validation — `TypeError` for non-string, `ValueError` for < 20 chars
  - [x] Truncation guard — 800k char limit, sentence-boundary cut, UI warning
  - [x] Retry loop — 3 attempts, exponential backoff, transient vs permanent discrimination
  - [x] Output validation — full type coercion, safe defaults, never crashes UI
- [x] **Task 2 — Meeting Summarization Module**
  - [x] `decisions` field — explicit decisions only; empty list if none
  - [x] `participants` field — named persons only; never invented
  - [x] Schema + system instruction + normalisation updated
  - [x] UI: 🎯 Key Decisions section + 🙋 Participants name pills
- [x] **Task 3 — Action Item Extraction Engine**
  - [x] Gemini path: `priority` (`high`/`medium`/`low`/`null`) — explicit urgency only
  - [x] Gemini path: `status` (`pending`/`completed`) — completed only if stated
  - [x] Python path: `_determine_priority()` — urgency regex → `high`; date → `medium`; else `null`
  - [x] Python path: `status` always `"pending"` (honest default, documented)
  - [x] UI: coloured priority badges 🔴🟠🟢 + status badges, shared `_render_action_items()` helper

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  Milestone 1 & 2 · Python · OpenAI Whisper · Google Gemini 3.5 Flash · Streamlit
</p>
