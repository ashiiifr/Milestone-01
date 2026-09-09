"""
app.py
------
Streamlit app for Milestone 1 – Tasks 1 & 10:
  Whisper Transcription  +  Meeting Analysis (Python extraction + Gemini summary)

Flow:
  Upload → Validate → Transcribe → Display Transcript
  → Python Analysis → Gemini Summary (optional)
"""

import streamlit as st
import time
from audio_processor import validate_audio_file, save_uploaded_file, cleanup_temp_file
from transcriber import transcribe_audio, save_transcript, WHISPER_MODELS
from meeting_analyzer import analyze_transcript
from gemini_summary import generate_meeting_summary

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Transcriber",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Meeting Transcriber & Analyser")
st.caption("Powered by OpenAI Whisper + Google Gemini 3.5 Flash — Milestone 1 · Tasks 1 & 10")

# ── Sidebar: model selection ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_size = st.selectbox(
        "Whisper Model",
        options=WHISPER_MODELS,
        index=1,  # default: 'base'
        help="Larger models are more accurate but slower.",
    )
    st.markdown("---")
    st.markdown(
        "**Supported formats:** mp3, wav, m4a, mp4, ogg, flac, webm  \n"
        "**Max file size:** 500 MB"
    )
    st.markdown("---")
    st.subheader("🔑 Gemini API Key")
    st.markdown(
        "Set `GEMINI_API_KEY` in a `.env` file next to `app.py`, "
        "or paste it here for this session only:"
    )
    gemini_key_input = st.text_input(
        "GEMINI_API_KEY (optional override)",
        type="password",
        help="Leave blank to use the value from your .env file.",
    )
    if gemini_key_input.strip():
        import os
        os.environ["GEMINI_API_KEY"] = gemini_key_input.strip()

# ── Step 1: Upload ────────────────────────────────────────────────────────────
st.subheader("Step 1 · Upload Meeting Recording")
uploaded_file = st.file_uploader(
    "Choose an audio or video file",
    type=["mp3", "wav", "m4a", "mp4", "ogg", "flac", "webm"],
    help="Upload your meeting recording to transcribe.",
)

tmp_path = None  # will hold the temp file path during processing

if uploaded_file is not None:
    # ── Step 2: Process / validate audio ─────────────────────────────────────
    st.subheader("Step 2 · Process Audio")
    is_valid, validation_msg = validate_audio_file(uploaded_file)

    if not is_valid:
        st.error(f"❌ {validation_msg}")
        st.stop()

    st.success(f"✅ {validation_msg}")

    # Show basic file info
    col1, col2 = st.columns(2)
    col1.metric("File Name", uploaded_file.name)
    col2.metric("Size", f"{uploaded_file.size / (1024 * 1024):.2f} MB")

    # ── Step 3: Transcribe ────────────────────────────────────────────────────
    st.subheader("Step 3 · Run Whisper")
    transcribe_btn = st.button("🚀 Transcribe", use_container_width=True, type="primary")

    if transcribe_btn:
        try:
            # Save uploaded bytes to a temp file
            uploaded_file.seek(0)
            tmp_path = save_uploaded_file(uploaded_file)

            # Progress feedback
            status = st.status("Processing…", expanded=True)
            with status:
                st.write("📂 Audio file saved.")
                time.sleep(0.3)

                st.write(f"🤖 Loading Whisper **{model_size}** model…")
                start_time = time.time()

                # ── Step 3 core: run Whisper ──────────────────────────────────
                result = transcribe_audio(tmp_path, model_size=model_size)

                elapsed = time.time() - start_time
                st.write(f"✅ Transcription complete in **{elapsed:.1f}s**.")
                status.update(label="Transcription complete!", state="complete")

            # Persist result in session state so it survives re-runs
            st.session_state["transcript"] = result

        except Exception as exc:
            st.error(f"❌ Transcription failed: {exc}")
        finally:
            cleanup_temp_file(tmp_path)

# ── Step 4 & 5: Generate & Display Transcript ─────────────────────────────────
if "transcript" in st.session_state:
    result = st.session_state["transcript"]

    st.subheader("Step 4 · Transcript")

    # Metadata row
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.metric("Language", result["language"].upper())
    meta_col2.metric("Model", result["model"])
    meta_col3.metric("Segments", len(result["segments"]))

    # Full transcript
    st.markdown("#### Full Transcript")
    if result["text"]:
        st.text_area(
            label="Transcript",
            value=result["text"],
            height=300,
            label_visibility="collapsed",
        )
    else:
        st.warning("⚠️ Transcript is empty — the audio may contain no speech.")

    # Timed segments (optional expandable view)
    if result["segments"]:
        with st.expander("🕐 View timed segments"):
            for seg in result["segments"]:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "").strip()
                st.markdown(
                    f"`[{start:6.1f}s → {end:6.1f}s]`  {text}"
                )

    # ── Step 5: Save transcript ───────────────────────────────────────────────
    st.subheader("Step 5 · Save Transcript")

    col_save, col_download = st.columns(2)

    with col_save:
        save_path = st.text_input(
            "Save path (.txt)",
            value=f"transcript_{result['file']}.txt",
        )
        if st.button("💾 Save to file"):
            try:
                saved = save_transcript(result["text"], save_path)
                st.success(f"Saved to `{saved}`")
            except Exception as exc:
                st.error(f"Failed to save: {exc}")

    with col_download:
        st.download_button(
            label="⬇️ Download transcript",
            data=result["text"],
            file_name=f"transcript_{result['file']}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 · Meeting Analysis
# ══════════════════════════════════════════════════════════════════════════════
if "transcript" in st.session_state and st.session_state["transcript"]["text"]:
    result = st.session_state["transcript"]

    st.markdown("---")
    st.header("🔍 Step 6 · Meeting Analysis")

    # ── 6a. Python-based analysis ────────────────────────────────────────────
    run_analysis = st.button(
        "🧠 Run Analysis",
        use_container_width=True,
        type="primary",
        help="Runs fully offline Python extraction (no API key needed).",
    )

    if run_analysis:
        with st.spinner("Running Python analysis…"):
            analysis = analyze_transcript(
                transcript=result["text"],
                segments=result["segments"],
            )
        st.session_state["analysis"] = analysis

    if "analysis" in st.session_state:
        analysis = st.session_state["analysis"]

        # ── Speaker count ────────────────────────────────────────────────────
        st.subheader("👥 Estimated Speakers")
        speaker_count = analysis["speaker_count"]
        st.metric(
            label="Distinct voices detected",
            value=speaker_count,
            help=(
                "Estimated using KMeans clustering on Whisper segment features "
                "(duration, position, log-probability, words-per-second). "
                "This is an acoustic estimate — it may differ from the true count "
                "for short or monotone recordings."
            ),
        )

        # ── Topics ───────────────────────────────────────────────────────────
        st.subheader("📌 Key Topics")
        topics = analysis["topics"]
        if topics:
            cols = st.columns(min(len(topics), 4))
            for i, topic in enumerate(topics):
                cols[i % 4].markdown(
                    f"<span style='background:#1f77b4;color:white;"
                    f"padding:4px 10px;border-radius:12px;"
                    f"font-size:0.85rem;'>{topic}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No distinct topics extracted from this transcript.")

        st.markdown(" ")  # spacing after tags

        # ── Key discussion points ────────────────────────────────────────────
        st.subheader("💡 Key Discussion Points")
        key_points = analysis["key_points"]
        if key_points:
            for i, point in enumerate(key_points, 1):
                st.markdown(f"**{i}.** {point}")
        else:
            st.info("No key points extracted.")

        # ── Action items ─────────────────────────────────────────────────────
        st.subheader("✅ Action Items")
        action_items = analysis["action_items"]
        if action_items:
            for item in action_items:
                assigned = item["assigned_to"] or "—"
                deadline = item["deadline"] or "—"
                with st.container(border=True):
                    st.markdown(f"**Task:** {item['task']}")
                    col_a, col_d = st.columns(2)
                    col_a.markdown(f"👤 **Assigned to:** {assigned}")
                    col_d.markdown(f"📅 **Deadline:** {deadline}")
        else:
            st.info("No action items detected in this transcript.")

        # ── All dates / deadlines found anywhere ─────────────────────────────
        all_deadlines = analysis["all_deadlines"]
        if all_deadlines:
            with st.expander(f"📅 All dates/deadlines found ({len(all_deadlines)})"):
                for d in all_deadlines:
                    st.markdown(f"• {d}")

        # ── 6b. Gemini AI summary ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("✨ AI Meeting Summary  *(Gemini 3.5 Flash)*")
        st.caption(
            "Gemini reads the full transcript and writes a coherent summary. "
            "Requires a valid `GEMINI_API_KEY`. "
            "All output is strictly constrained to what is in the transcript — "
            "the model cannot invent names, tasks, or dates."
        )

        run_gemini = st.button(
            "🚀 Generate AI Summary",
            use_container_width=True,
            help="Calls Gemini 3.5 Flash. Requires GEMINI_API_KEY.",
        )

        if run_gemini:
            with st.spinner("Calling Gemini 3.5 Flash…"):
                try:
                    gemini_result = generate_meeting_summary(result["text"])
                    st.session_state["gemini_result"] = gemini_result
                except RuntimeError as exc:
                    st.error(f"❌ {exc}")
                except ValueError as exc:
                    st.error(f"❌ Schema error: {exc}")
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")

        if "gemini_result" in st.session_state:
            gr = st.session_state["gemini_result"]

            # Summary paragraph
            if gr.get("summary"):
                st.markdown("#### 📝 Summary")
                st.markdown(gr["summary"])

            # Key points from Gemini
            if gr.get("key_points"):
                st.markdown("#### 💡 Key Points")
                for pt in gr["key_points"]:
                    st.markdown(f"- {pt}")

            # Topics from Gemini
            if gr.get("topics"):
                st.markdown("#### 📌 Topics")
                st.markdown("  ".join(
                    f"`{t}`" for t in gr["topics"]
                ))

            # Action items from Gemini
            if gr.get("action_items"):
                st.markdown("#### ✅ Action Items")
                for item in gr["action_items"]:
                    assigned = item.get("assigned_to") or "—"
                    deadline = item.get("deadline") or "—"
                    with st.container(border=True):
                        st.markdown(f"**Task:** {item['task']}")
                        col_a, col_d = st.columns(2)
                        col_a.markdown(f"👤 **Assigned to:** {assigned}")
                        col_d.markdown(f"📅 **Deadline:** {deadline}")

            # Raw JSON expander (useful for debugging / verification)
            with st.expander("🔎 View raw Gemini JSON response"):
                import json
                st.code(json.dumps(gr, indent=2), language="json")
