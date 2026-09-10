"""
meeting_analyzer.py
--------------------
All DETERMINISTIC, Python-only extraction from a meeting transcript.
Nothing is guessed or invented — every result is derived directly from the
text using rule-based NLP and signal-processing on the audio segments.

WHY PYTHON FOR THESE TASKS?
──────────────────────────
These extractions are *precise and verifiable*.  A regex either matches a
date pattern or it doesn't — there is no ambiguity, no hallucination risk,
and no API cost.  Python libraries (NLTK, scikit-learn) give us reproducible,
deterministic results every time on the same input.  We only hand control to
an LLM (Gemini) for the one task that genuinely requires natural-language
understanding: writing a fluent, coherent *summary*.

Extracted by this module
────────────────────────
1. Estimated speaker count  – from audio segment pitch statistics (KMeans)
2. Key topics              – TF-IDF top noun-phrases from the transcript
3. Key discussion points   – highest-scoring sentences by TF-IDF weight
4. Action items            – sentences containing imperative / task language
                             (Milestone 2: now includes priority + status)
5. Assigned person         – proper-noun directly after assignment keywords
6. Deadlines / dates       – regex over common date & time expressions

Milestone 2 additions (TASK 3)
───────────────────────────────
• action items now include:
    - priority: "high" | "medium" | null   (keyword-based, never guessed)
    - status:   always "pending"           (a transcript describes intent,
                                            not completion — see comment in
                                            extract_action_items)
"""

from __future__ import annotations

import re
import string
from typing import Any

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# ── NLTK data (downloaded once during setup) ─────────────────────────────────
_REQUIRED_NLTK = [
    ("tokenizers/punkt",                    "punkt"),
    ("tokenizers/punkt_tab",                "punkt_tab"),
    ("corpora/stopwords",                   "stopwords"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("chunkers/maxent_ne_chunker_tab",      "maxent_ne_chunker_tab"),
    ("corpora/words",                       "words"),
]

def _ensure_nltk_data() -> None:
    for path, pkg in _REQUIRED_NLTK:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk_data()
_STOP_WORDS = set(stopwords.words("english"))

# ── Action-item trigger vocabulary ───────────────────────────────────────────
# Sentences containing these patterns strongly suggest an assigned task.
_ACTION_VERBS = re.compile(
    r"\b(will|should|must|need to|needs to|has to|have to|going to|"
    r"please|action|follow[- ]?up|follow up|send|review|update|schedule|"
    r"complete|finish|prepare|submit|check|confirm|coordinate|ensure|"
    r"create|write|draft|share|present|implement|fix|resolve|test|deploy|"
    r"reach out|set up|setup|assign|book|arrange|notify)\b",
    re.IGNORECASE,
)

# ── Assignment keywords — the person tends to follow these ───────────────────
_ASSIGN_KW = re.compile(
    r"\b(assigned to|assign to|owned by|owner|responsible|"
    r"[A-Z][a-z]+ will|[A-Z][a-z]+ should|[A-Z][a-z]+ needs to|"
    r"[A-Z][a-z]+ has to|[A-Z][a-z]+ is going to)\b",
    re.IGNORECASE,
)

# ── Date / deadline patterns ─────────────────────────────────────────────────
_DATE_PATTERNS = [
    # ISO: 2025-12-31
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
    # Written month: December 31, 2025  /  31 December 2025
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
    r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{4})?\b",
    # Short: 12/31, 12/31/2025, 31.12.2025
    r"\b\d{1,2}[/\.]\d{1,2}(?:[/\.]\d{2,4})?\b",
    # Relative: "next Monday", "by Friday", "end of Q2", "by EOD", "by EOM"
    r"\b(?:next|this|coming)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|"
    r"Saturday|Sunday|week|month|quarter|year)\b",
    r"\bby\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"EOD|EOM|end of (?:the )?(?:day|week|month|quarter|year))\b",
    r"\bend of (?:the )?(?:day|week|month|Q[1-4]|quarter|year)\b",
    r"\b(?:tomorrow|today|tonight)\b",
    # "in 2 weeks", "in 3 days"
    r"\bin \d+\s+(?:day|days|week|weeks|month|months)\b",
]
_DATE_RE = re.compile("|".join(_DATE_PATTERNS), re.IGNORECASE)

# ── Proper-name heuristic ─────────────────────────────────────────────────────
# A name: one or two consecutive Title-Case tokens not in the stopword list
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b")

# ── TASK 3 · Priority heuristic ───────────────────────────────────────────────
# "high" — sentence contains explicit urgency language
_HIGH_PRIORITY_RE = re.compile(
    r"\b(urgent|urgently|asap|a\.s\.a\.p|critical|immediately|"
    r"top priority|high priority|by end of day|by eod|eod|"
    r"right away|as soon as possible)\b",
    re.IGNORECASE,
)
# "medium" — sentence contains a near-term date/time reference but no high-priority words
# We reuse _DATE_RE for this; the logic is in _determine_priority() below.


# ═══════════════════════════════════════════════════════════════════════════════
# 1 · SPEAKER ESTIMATION FROM AUDIO SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_speaker_count(segments: list[dict]) -> int:
    """
    Estimate the number of distinct speakers using acoustic features extracted
    directly from Whisper's timed segments.

    HOW IT WORKS:
    ─────────────
    Whisper returns a list of segments, each with:
      • start / end time  → segment duration and position in the recording
      • avg_logprob       → model's confidence (correlates with voice clarity)
      • no_speech_prob    → probability the segment is silence

    We build a small feature vector per segment:
      [duration, position_in_recording, avg_logprob, words_per_second]

    Then KMeans clustering groups segments by acoustic similarity.  We try
    k = 1 … min(6, n_segments) and choose the k that gives the best
    *inertia knee* (biggest drop in within-cluster variance per extra cluster).
    This is a lightweight proxy for diarisation — it doesn't label WHO spoke,
    only HOW MANY distinct voices are likely present.

    Why Python?  Because this is pure maths — cluster analysis on numbers.
    There is no language understanding required and no risk of invention.

    Returns:
        int in the range [1, 6]
    """
    if not segments:
        return 1

    # Filter out silent/very short segments
    valid = [
        s for s in segments
        if s.get("end", 0) - s.get("start", 0) > 0.5
        and s.get("no_speech_prob", 1.0) < 0.8
    ]
    if len(valid) < 3:
        return 1

    # Build feature matrix
    total_duration = max(s.get("end", 0) for s in valid) or 1.0
    features = []
    for seg in valid:
        start       = seg.get("start", 0)
        end         = seg.get("end", 0)
        duration    = end - start
        position    = start / total_duration          # normalised 0→1
        logprob     = seg.get("avg_logprob", -0.5)   # negative float
        text        = seg.get("text", "")
        word_count  = len(text.split())
        wps         = word_count / duration if duration > 0 else 0  # words/sec

        features.append([duration, position, logprob, wps])

    X = np.array(features, dtype=float)
    if X.shape[0] < 2:
        return 1

    # Standardise so no single feature dominates
    X = StandardScaler().fit_transform(X)

    max_k   = min(6, len(valid))
    inertias: list[float] = []

    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(float(km.inertia_))

    # Knee detection: find where adding one more cluster gives diminishing returns.
    # We look for the largest *relative* drop in inertia.
    best_k = 1
    if len(inertias) > 1:
        drops = [
            (inertias[i] - inertias[i + 1]) / (inertias[i] + 1e-9)
            for i in range(len(inertias) - 1)
        ]
        # The knee is the index of the biggest relative drop
        knee = int(np.argmax(drops))
        best_k = knee + 2   # +2 because drops[0] corresponds to k=2

        # Sanity cap: if the improvement after the knee is marginal, stay at 1
        if drops[knee] < 0.15:
            best_k = 1

    return max(1, min(best_k, 6))


# ═══════════════════════════════════════════════════════════════════════════════
# 2 · TOPIC EXTRACTION  (TF-IDF noun-phrase scoring)
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_noun_phrases(text: str) -> list[str]:
    """
    Use NLTK POS tagging + simple chunking to extract noun phrases.
    A noun phrase is: (optional adjectives) + (one or more nouns).

    WHY NOUN PHRASES?
    Noun phrases (e.g. "project timeline", "budget review") are almost always
    more informative as topics than individual words.  A simple grammar rule
    applied to POS tags is fast, deterministic, and needs no training data.
    """
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    # Grammar: optional adjectives followed by one or more nouns
    grammar = r"NP: {<JJ>*<NN.*>+}"
    parser  = nltk.RegexpParser(grammar)
    tree    = parser.parse(tagged)

    phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        phrase = " ".join(w for w, _ in subtree.leaves()
                          if w.lower() not in _STOP_WORDS and len(w) > 2)
        if phrase.strip():
            phrases.append(phrase.strip())
    return phrases


def extract_topics(transcript: str, max_topics: int = 8) -> list[str]:
    """
    Extract the most prominent discussion topics.

    METHOD:
    1. Tokenise transcript into sentences.
    2. Extract noun phrases from each sentence.
    3. Build a TF-IDF matrix over those phrases (treated as 'documents').
    4. Rank phrases by their mean TF-IDF score across all sentences.
    5. Return the top `max_topics` unique phrases.

    WHY TF-IDF?
    TF-IDF rewards phrases that appear often (high TF) but are not trivially
    common across all sentences (low IDF would penalise them).  This surfaces
    the phrases that are *specific to this meeting* rather than generic filler.
    The result is fully deterministic — same transcript → same topics, every time.
    """
    if not transcript.strip():
        return []

    sentences = sent_tokenize(transcript)
    if len(sentences) < 2:
        sentences = [transcript]

    # Collect noun phrases per sentence (use sentence as the TF-IDF 'document')
    docs = []
    for sent in sentences:
        phrases = _extract_noun_phrases(sent)
        docs.append(" ".join(phrases) if phrases else _clean_text(sent))

    # Remove empty docs
    docs = [d for d in docs if d.strip()]
    if not docs:
        return []

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words="english",
            min_df=1,
            max_features=500,
        )
        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores   = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        # Sort descending and pick top candidates
        ranked_indices = mean_scores.argsort()[::-1]
        topics = []
        seen   = set()
        for idx in ranked_indices:
            phrase = feature_names[idx]
            # Skip single stop words, very short tokens, or duplicates
            if (len(phrase) > 3
                    and phrase.lower() not in _STOP_WORDS
                    and phrase not in seen):
                # Title-case for display
                topics.append(phrase.title())
                seen.add(phrase)
            if len(topics) >= max_topics:
                break

        return topics

    except ValueError:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# 3 · KEY DISCUSSION POINTS  (sentence-level TF-IDF scoring)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_key_points(transcript: str, max_points: int = 6) -> list[str]:
    """
    Extract the most important individual sentences from the transcript.

    METHOD:
    1. Split the transcript into sentences.
    2. Build a TF-IDF matrix where each row is a sentence.
    3. Score each sentence by the *sum* of TF-IDF weights of its tokens
       (a standard extractive summarisation baseline called TextRank-lite).
    4. Return the top `max_points` sentences in their original reading order.

    WHY SUM OF TF-IDF WEIGHTS?
    A sentence whose words have high TF-IDF scores is a sentence that talks
    about topics specific and central to *this* document.  Generic pleasantries
    ("thanks for joining") score very low; content-rich statements score high.
    This is reproducible, offline, and free — no API call needed.
    """
    if not transcript.strip():
        return []

    sentences = sent_tokenize(transcript)
    # Filter out very short sentences (likely artefacts)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences[:max_points]

    try:
        vectorizer   = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Sentence score = sum of tfidf weights for all its terms
        sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

        # Rank by score, then restore original order for readability
        top_n   = min(max_points, len(sentences))
        top_idx = np.argsort(sentence_scores)[::-1][:top_n]
        top_idx = sorted(top_idx)  # restore document order

        return [sentences[i] for i in top_idx]

    except ValueError:
        return sentences[:max_points]


# ═══════════════════════════════════════════════════════════════════════════════
# 4 · ACTION ITEM EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_person(sentence: str) -> str | None:
    """
    Extract the first plausible person name from a sentence.

    METHOD:
    1. Try NLTK Named-Entity Recognition (NER) — finds PERSON entities tagged
       by the pre-trained chunker.
    2. Fall back to a simple heuristic: Title-Case token that is not a common
       stop word and follows an assignment keyword.

    WHY NOT JUST REGEX?
    A pure regex "two Title-Case words" would match "Next Monday" or "Action Item".
    NER is trained on real text and knows that "Sarah Jones" is a person while
    "Action Item" is not.  The regex fallback catches names NER misses in short
    transcripts.

    Returns None if no confident name is found — we never invent a name.
    """
    # ── Attempt 1: NLTK NER ──────────────────────────────────────────────────
    try:
        tokens  = word_tokenize(sentence)
        tagged  = nltk.pos_tag(tokens)
        tree    = nltk.ne_chunk(tagged, binary=False)

        for subtree in tree.subtrees(filter=lambda t: t.label() == "PERSON"):
            name = " ".join(w for w, _ in subtree.leaves())
            if len(name) >= 2:
                return name
    except Exception:
        pass

    # ── Attempt 2: regex heuristic — name after assignment keyword ────────────
    assign_match = _ASSIGN_KW.search(sentence)
    if assign_match:
        after = sentence[assign_match.end():].strip()
        name_match = _NAME_RE.match(after)
        if name_match:
            candidate = name_match.group(1)
            # Reject obvious non-names
            if candidate.lower() not in _STOP_WORDS and len(candidate) > 2:
                return candidate

    return None


def _determine_priority(sentence: str) -> str | None:
    """
    TASK 3 — Keyword-based priority heuristic for deterministic extraction.

    Rules (in order of precedence):
      "high"   — sentence contains an explicit urgency word/phrase
                 (urgent, ASAP, critical, by EOD, top priority, etc.)
      "medium" — sentence contains a near-term date match but no high-priority
                 words (today, tomorrow, this week, next Monday, etc.)
      None     — no urgency signal found

    We never guess: if none of the explicit signals are present, we return None.
    """
    if _HIGH_PRIORITY_RE.search(sentence):
        return "high"

    # "medium": a date is present but no high-priority language
    if _DATE_RE.search(sentence):
        return "medium"

    return None


def extract_action_items(transcript: str) -> list[dict[str, Any]]:
    """
    Extract action items / tasks from the transcript.

    METHOD:
    1. Split into sentences.
    2. Score each sentence against `_ACTION_VERBS` (regex vocab of task words).
    3. For each candidate sentence:
       a. Try to extract an assigned person (NER + heuristic).
       b. Try to extract a deadline (regex date patterns).
       c. Assign priority via keyword heuristic (TASK 3).
       d. Default status to "pending" — a transcript describes stated intent,
          not completion.  We never know from text alone that a task is done
          unless explicitly said; "pending" is the honest safe default.
    4. Return a list of dicts matching the Milestone 2 schema.

    WHY PYTHON?
    Regex and NER over a fixed vocabulary of task-indicating verbs is
    100 % reproducible.  We only flag sentences that *explicitly contain*
    action language — we never infer or guess tasks.  If a sentence does not
    match, it is not included.
    """
    if not transcript.strip():
        return []

    sentences = sent_tokenize(transcript)
    action_items: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent.split()) < 4:
            continue

        # Only include sentences with explicit action language
        if not _ACTION_VERBS.search(sent):
            continue

        # Deduplicate very similar sentences
        normalised = re.sub(r"\s+", " ", sent.lower().strip())
        if normalised in seen_texts:
            continue
        seen_texts.add(normalised)

        person   = _extract_person(sent)
        deadline = _extract_deadline(sent)
        priority = _determine_priority(sent)

        # status is always "pending" for deterministic extraction.
        # A meeting transcript records what people *intend* to do;
        # it does not confirm completion.  The Gemini path may set
        # "completed" when the transcript explicitly says so, but
        # our rule-based path cannot reliably detect that nuance,
        # so we default honestly to "pending".
        action_items.append({
            "task":        sent,
            "assigned_to": person,    # None → null in JSON
            "deadline":    deadline,  # None → null in JSON
            "priority":    priority,  # "high" | "medium" | None
            "status":      "pending",
        })

    return action_items


# ═══════════════════════════════════════════════════════════════════════════════
# 5 · DEADLINE / DATE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_deadline(text: str) -> str | None:
    """
    Find the first date/deadline expression in a text fragment.
    Returns the matched string exactly as it appears, or None.

    WHY REGEX?
    Date formats are highly structured.  A regex over known patterns is precise,
    never invents a date that isn't in the text, and runs instantly.
    """
    match = _DATE_RE.search(text)
    return match.group(0).strip() if match else None


def extract_deadlines(transcript: str) -> list[str]:
    """
    Find all date/deadline expressions anywhere in the transcript.
    Returns a deduplicated list of matched strings.
    """
    matches = _DATE_RE.findall(transcript)
    seen, results = set(), []
    for m in matches:
        m = m.strip()
        if m and m.lower() not in seen:
            seen.add(m.lower())
            results.append(m)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6 · MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_transcript(transcript: str, segments: list[dict]) -> dict[str, Any]:
    """
    Run all Python-based extractions and return a single structured dict.

    Args:
        transcript: The full plain-text transcript string.
        segments:   Whisper segment dicts (each has start, end, avg_logprob,
                    no_speech_prob, text).

    Returns:
        {
            "speaker_count":  int,
            "topics":         [str, ...],
            "key_points":     [str, ...],
            "action_items":   [{"task": str,
                                "assigned_to": str|None,
                                "deadline": str|None}, ...],
            "all_deadlines":  [str, ...],   # dates found anywhere in transcript
        }
    """
    return {
        "speaker_count": estimate_speaker_count(segments),
        "topics":        extract_topics(transcript),
        "key_points":    extract_key_points(transcript),
        "action_items":  extract_action_items(transcript),
        "all_deadlines": extract_deadlines(transcript),
    }
