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
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the App](#-running-the-app)
- [Using the App](#-using-the-app)
- [Demo Audio Files](#-demo-audio-files)
- [Module Reference](#-module-reference)
- [Design Decisions](#-design-decisions)
- [Milestone Tasks](#-milestone-tasks)

---

## 🌟 Overview

This project is a full end-to-end **meeting transcription and intelligence pipeline** built across Milestone 1 and Milestone 2. It takes an audio or video recording of a meeting, transcribes it using OpenAI Whisper, and then runs a two-layer analysis:

1. **Python-based deterministic extraction** — speaker count estimation, topics, key points, action items (with priority & status), assigned persons, and deadlines. All extracted directly from the transcript using NLP algorithms, with zero API calls and zero hallucination risk.

2. **Gemini AI meeting intelligence** — a structured, validated analysis produced by Google Gemini 3.5 Flash: summary, key decisions, participants, key points, topics, and action items — all constrained by a strict JSON schema. The LLM service layer includes input validation, a truncation guard, retry logic, and full output validation.

---

## ✨ Features

### 🎙️ Transcription (Task 1)
- Upload audio/video in **7 formats**: `mp3`, `wav`, `m4a`, `mp4`, `ogg`, `flac`, `webm`
- Powered by **OpenAI Whisper** — runs fully offline on your machine
- Choose model size: `tiny` → `base` → `small` → `medium` → `large`
- Displays full transcript with **timed segments** (start → end timestamps)
- Save transcript to `.txt` or download directly from the browser
- Bundled `ffmpeg` binary — no separate system install needed

### 🔍 Meeting Analysis (Milestone 1 Task 10 + Milestone 2)

| Extraction | Method | API needed? |
|---|---|---|
| 👥 Estimated speaker count | KMeans clustering on Whisper segment acoustics | ❌ No |
| 📌 Key topics | TF-IDF noun-phrase scoring | ❌ No |
| 💡 Key discussion points | TF-IDF sentence scoring | ❌ No |
| ✅ Action items + priority + status | Regex + NLTK NER + urgency keywords | ❌ No |
| 👤 Assigned person | NLTK NER (PERSON entities) + heuristic | ❌ No |
| 📅 Deadlines / dates | Regex over 8 date pattern categories | ❌ No |
| ✨ Meeting summary | **Gemini 3.5 Flash** (structured JSON) | ✅ Yes |
| 🎯 Key decisions | **Gemini 3.5 Flash** (schema-enforced) | ✅ Yes |
| 🙋 Participants | **Gemini 3.5 Flash** (no invention) | ✅ Yes |

### 🛡️ Anti-Hallucination Safeguards (Gemini)
- Strict JSON schema with `additionalProperties: false` at every level
- `null` enforced for missing persons, deadlines, and priorities — model cannot guess
- System instruction explicitly forbids inventing names, decisions, tasks, or dates
- Schema enforced at the API transport level via `response_format`
- Output fully re-validated in Python after parsing — invalid values coerced to safe defaults

### 🔒 LLM Service Hardening (Milestone 2 Task 1)
- **Prompt templates** — system instruction and user message are named constants, editable without touching call logic
- **Input validation** — rejects non-string input and transcripts under 20 characters
- **Truncation guard** — transcripts over 800,000 characters are cut at the last sentence boundary; a warning is shown in the UI instead of silently dropping content or crashing
- **Retry loop** — 3 attempts with exponential backoff (2 s → 4 s) on transient errors (timeout, rate limit, 429, 503); fast fail on auth errors
- **Output validation** — every field type-checked and coerced; priority/status validated against allowed values

---

## 📁 Project Structure

```
Milestone 01/
│
├── app.py                  # Streamlit web application (main entry point)
├── audio_processor.py      # File validation & temporary storage
├── transcriber.py          # OpenAI Whisper wrapper + ffmpeg patching
├── meeting_analyzer.py     # All Python-based deterministic NLP extraction
├── gemini_summary.py       # Gemini 3.5 Flash structured JSON summary
├── generate_demo_audio.py  # Script to generate TTS demo audio files
│
├── demo_audio/             # Pre-generated demo recordings (gTTS)
│   ├── demo_10s.mp3        # ~10 second recording
│   ├── demo_30s.mp3        # ~30 second recording
│   ├── demo_1m.mp3         # ~1 minute recording
│   ├── demo_2m.mp3         # ~2 minute recording
│   └── demo_5m.mp3         # ~5 minute recording
│
├── Docs/
│   └── milestones.md       # Full task breakdown for all milestones
│
├── requirements.txt        # Python dependencies
├── .env.example            # API key template (safe to commit)
├── .env                    # Your real API key (NEVER committed — in .gitignore)
├── .gitignore              # Excludes .env, myenv/, __pycache__, *.pt, etc.
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   1. File Upload           │
              │   audio_processor.py       │
              │   • Validate format/size   │
              │   • Save to temp file      │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   2. Transcription         │
              │   transcriber.py           │
              │   • Load Whisper model     │
              │   • Run ffmpeg decode      │
              │   • Return text + segments │
              └──────┬──────────┬──────────┘
                     │          │
          ┌──────────▼──┐  ┌────▼──────────────────┐
          │  3. Python  │  │  4. Gemini Summary     │
          │  Analysis   │  │  gemini_summary.py     │
          │             │  │                        │
          │ • Speakers  │  │ • model: gemini-3.5-   │
          │ • Topics    │  │   flash                │
          │ • Key pts   │  │ • Strict JSON schema   │
          │ • Actions   │  │ • No hallucination     │
          │ • Dates     │  │ • GEMINI_API_KEY env   │
          └──────┬──────┘  └────┬───────────────────┘
                 │              │
              ┌──▼──────────────▼──┐
              │   Streamlit Display │
              │   (Step 6 panel)    │
              └─────────────────────┘
```

---

## 📦 Prerequisites

- **Python 3.10 or higher**
- A **Google Gemini API key** (free tier available) — only needed for the AI summary feature
  - Get one at: https://aistudio.google.com/app/apikey

> **ffmpeg** is bundled automatically via `imageio-ffmpeg` — you do **not** need to install it separately.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ashiiifr/Milestone-01.git
cd "Milestone-01"
```

### 2. Create a virtual environment

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

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

This installs:
- `openai-whisper` — transcription engine
- `streamlit` — web interface
- `google-genai>=2.0.0` — Gemini API SDK
- `nltk` — tokenisation, POS tagging, NER
- `scikit-learn` — TF-IDF vectoriser, KMeans
- `python-dotenv` — loads `.env` file
- `imageio[ffmpeg]` — bundled ffmpeg binary
- `ffmpeg-python` — ffmpeg Python bindings

### 4. Download NLTK data

```python
python -c "
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

### Set up your Gemini API key

```powershell
# Copy the template
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder:

```env
GEMINI_API_KEY=your_actual_key_here
```

> ⚠️ The `.env` file is in `.gitignore` and will **never** be committed to Git.  
> The `.env.example` file contains no real key and is safe to share.

The Gemini key is **only required for the AI summary feature**. All Python extraction (speaker count, topics, action items, etc.) works completely offline with no key.

---

## ▶️ Running the App

```powershell
.\myenv\Scripts\streamlit.exe run app.py
```

Your browser opens automatically at **http://localhost:8501**.

To stop the app: press `Ctrl + C` in the terminal.

---

## 🖥️ Using the App

### Step 1 — Upload
Click the file uploader and select an audio or video file.  
Supported: `mp3`, `wav`, `m4a`, `mp4`, `ogg`, `flac`, `webm` · Max size: 500 MB

### Step 2 — Process Audio
The app validates the file format and size and displays basic metadata.

### Step 3 — Transcribe
Select a Whisper model in the sidebar (default: `base`), then click **🚀 Transcribe**.

| Model | Speed | Accuracy | VRAM |
|-------|-------|----------|------|
| tiny | Fastest | Lower | ~1 GB |
| base | Fast | Good | ~1 GB |
| small | Medium | Better | ~2 GB |
| medium | Slow | High | ~5 GB |
| large | Slowest | Best | ~10 GB |

### Step 4 — View Transcript
The full transcript appears with language detection, model used, and segment count.
Expand **🕐 View timed segments** for word-level timestamps.

### Step 5 — Save / Download
Save the transcript to a `.txt` file or download it directly from the browser.

### Step 6 — Meeting Analysis

**Python Analysis (no API key needed):**  
Click **🧠 Run Analysis** to get:
- 👥 Estimated number of distinct speakers
- 📌 Key topics (colour-coded pills)
- 💡 Key discussion points (top sentences by importance)
- ✅ Action items — with assigned person, deadline, 🔴/🟠 priority badge, and PENDING/COMPLETED status badge
- 📅 All dates and deadlines found in the transcript

**Gemini AI Intelligence (requires `GEMINI_API_KEY`):**  
Click **🚀 Generate AI Summary** to get:
- 📝 Fluent meeting summary paragraph
- 🎯 Key decisions explicitly made in the meeting
- 🙋 Participant names mentioned in the transcript
- 💡 Key points, 📌 Topics
- ✅ Action items with assigned person, deadline, priority, and status

If the transcript is too long (> 800,000 chars) a truncation warning appears above the results. If the API key is invalid or the network fails, a clean retry-then-fail error is shown instead of a raw stack trace.

You can also paste your API key directly in the **sidebar** for quick testing.

---

## 🎵 Demo Audio Files

Pre-generated demo recordings are included for testing. They were created using Google Text-to-Speech (gTTS) and simulate a realistic meeting scenario.

| File | Duration | Content |
|------|----------|---------|
| `demo_10s.mp3` | ~10 seconds | Short welcome message |
| `demo_30s.mp3` | ~30 seconds | Brief meeting intro |
| `demo_1m.mp3` | ~1 minute | Agenda overview |
| `demo_2m.mp3` | ~2 minutes | Sprint update & feature discussion |
| `demo_5m.mp3` | ~5 minutes | Full project update meeting |

To regenerate them (requires internet for gTTS):

```powershell
.\myenv\Scripts\python.exe generate_demo_audio.py
```

---

## 📖 Module Reference

### `audio_processor.py`
Handles everything before transcription.

| Function | Description |
|----------|-------------|
| `validate_audio_file(file)` | Checks extension and file size. Returns `(bool, message)` |
| `save_uploaded_file(file)` | Writes uploaded bytes to a temp file. Returns path |
| `cleanup_temp_file(path)` | Deletes the temp file after transcription |

### `transcriber.py`
Wraps OpenAI Whisper.

| Function | Description |
|----------|-------------|
| `transcribe_audio(path, model_size)` | Runs Whisper. Returns `{text, segments, language, model, file}` |
| `save_transcript(text, path)` | Saves transcript string to `.txt` |
| `_ensure_ffmpeg()` | Locates bundled ffmpeg and patches `whisper.audio.load_audio` |

### `meeting_analyzer.py`
All Python-only, deterministic NLP extraction (Milestone 2: priority + status added).

| Function | Description |
|----------|-------------|
| `analyze_transcript(transcript, segments)` | Main entry — runs all extractions, returns combined dict |
| `estimate_speaker_count(segments)` | KMeans on segment acoustic features (duration, position, logprob, wps) |
| `extract_topics(transcript)` | TF-IDF noun-phrase scoring, returns top 8 topics |
| `extract_key_points(transcript)` | TF-IDF sentence scoring, returns top 6 sentences |
| `extract_action_items(transcript)` | Regex action-verb detection + NER + `_determine_priority()` + status="pending" |
| `_determine_priority(sentence)` | `"high"` for urgency words · `"medium"` for near-term date · `null` otherwise |
| `extract_deadlines(transcript)` | Regex over 8 date pattern categories |

### `gemini_summary.py`
Gemini 3.5 Flash LLM service layer (Milestone 2 hardened).

| Symbol | Description |
|--------|-------------|
| `SYSTEM_INSTRUCTION` | Named constant — anti-hallucination + priority/status rules |
| `build_user_message(transcript)` | Wraps transcript in standard request message |
| `build_full_prompt(transcript)` | Combines system instruction + user message |
| `RESPONSE_SCHEMA` | Full JSON schema: summary, key_points, decisions, participants, topics, action_items (with priority + status) |
| `validate_transcript(transcript)` | Type check, min-length check, 800k-char truncation guard |
| `_call_with_retry(client, prompt)` | 3-attempt retry with exponential backoff |
| `_normalise_response(data)` | Full output type validation + coercion |
| `generate_meeting_summary(transcript)` | Public entry point — validates, calls, normalises, returns dict |

---

## 🧠 Design Decisions

### Why Python for extraction, Gemini only for summary?

Every extraction task except the summary has a **correct and verifiable answer** — a date is either in the text or it isn't, a task verb either appears or it doesn't. Python regex and NLP libraries handle these deterministically with no API cost and no hallucination risk.

A **summary** is genuinely different — it requires understanding the transcript as a narrative and expressing that in fluent English. That's where an LLM adds real value. But we still constrain it heavily.

### How hallucination is prevented in Gemini

1. **Strict JSON schema** — `additionalProperties: false` means the model cannot add any field not in our schema
2. **Nullable types** — `"type": ["string", "null"]` forces explicit `null` instead of guessing
3. **System instruction** — processed before the transcript, explicitly forbids inventing names, tasks, or dates
4. **Transport-level enforcement** — `response_format` with `mime_type: application/json` causes the API to reject any response that doesn't parse as valid JSON matching the schema

### Why KMeans for speaker estimation?

Full speaker diarisation (knowing *who* said *what*) requires deep audio models. KMeans on Whisper's segment features (duration, position, log-probability, words-per-second) gives a lightweight but reasonable estimate of *how many* distinct voices are present — no extra model download required.

---

## 📋 Milestone Tasks

See [`Docs/milestones.md`](Docs/milestones.md) for the full breakdown.

### Milestone 1 — Audio Processing & Transcription
- [x] Task 1 — Whisper Transcription workflow (end-to-end)
- [ ] Task 2 — File Upload Validation (testing)
- [ ] Task 3 — Transcript Validation (testing)
- [ ] Task 4 — Streamlit Interface verification (testing)
- [ ] Task 5 — Accuracy Testing (≥90% WER)

### Milestone 1 — Text Ingestion & Baseline Sentiment
- [ ] Task 1 — Text Ingestion Workflow
- [ ] Task 2 — Preprocessing Validation
- [ ] Task 3 — VADER Sentiment Validation
- [ ] Task 4 — Initial Emotion/Sentiment Report
- [x] Task 5 (partial) — GitHub repo setup & code pushed

### Milestone 1 — Meeting Analysis (Task 10)
- [x] Speaker estimation from audio (KMeans)
- [x] Topic extraction (TF-IDF)
- [x] Key discussion points (TF-IDF sentence scoring)
- [x] Action item extraction (regex + NER)
- [x] Assigned person detection (NLTK NER)
- [x] Deadline/date extraction (regex)
- [x] Gemini 3.5 Flash structured summary
- [x] Streamlit integration (Step 6 panel)

### Milestone 2 — Meeting Intelligence Pipeline
- [x] Task 1 — LLM service hardening
  - [x] Prompt templates (named constants)
  - [x] Input validation (type check + min length)
  - [x] Truncation guard (800k chars, sentence-boundary, UI warning)
  - [x] Retry loop (3 attempts, exponential backoff)
  - [x] Output validation + coercion (all fields)
- [x] Task 2 — Meeting Summarization Module
  - [x] `decisions` field (explicit decisions only)
  - [x] `participants` field (named persons only)
  - [x] UI display: 🎯 Key Decisions + 🙋 Participants
- [x] Task 3 — Action Item Extraction Engine
  - [x] `priority` field — Gemini path (explicit urgency language only)
  - [x] `status` field — Gemini path (pending/completed)
  - [x] `priority` field — Python path (`_determine_priority()` heuristic)
  - [x] `status` field — Python path (always "pending", documented assumption)
  - [x] Priority + status badges in UI (both paths, shared helper)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Built across Milestone 1 & 2 · Python + Whisper + Gemini 3.5 Flash</p>
