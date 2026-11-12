"""
LabMultimodal.py
Multimodal Lab: summarize an uploaded image or audio file and answer a question.

Pipeline (from the slides):
    Image:  PNG/JPEG  -> prepare_image()    -> build_content() -> analyze_content()
    Audio:  MP3/WAV/M4A -> transcribe_audio() -> build_content() -> analyze_content()
    Output: short summary + exactly 3 key points + optional answer
"""

import base64
import io
import wave
from pathlib import Path

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
MODEL = "gpt-4o-mini"               # vision-capable model used for analysis
TRANSCRIPTION_MODEL = "whisper-1"   # speech-to-text model for audio
MAX_BYTES = 5 * 1024 * 1024         # lab rule: 5 MB max (app rule, not an API limit)
MAX_AUDIO_SECONDS = 60              # lab rule: audio <= 60 seconds

IMAGE_FORMATS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
AUDIO_FORMATS = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}

# System prompt: evidence rules + output format (slide 12)
SYSTEM_PROMPT = """You are a careful multimodal assistant.
Use ONLY the evidence in the provided image or transcript.

Respond in this exact format:
**Summary:** 2-3 sentences describing the content.
**Key points:**
1. ...
2. ...
3. ...
**Answer:** Answer the user's question if one was asked; otherwise write "No question asked."

Rules:
- Give exactly three key points.
- If text, numbers, or details are blurry, cut off, inaudible, or missing, say so
  explicitly instead of guessing.
- Do not invent exact values (e.g., chart numbers or coordinates) you cannot read clearly.
"""

# Revised prompt (Part B: "compare the output with the source and revise one prompt").
# Problem seen with SYSTEM_PROMPT on a photo of a woman in a white coat:
#   - guesses were stated as facts ("medical coat" -> "clinical setting" -> "consultation")
#   - "exactly three key points" forced a filler point ("wooden paneling adds to the
#     professional atmosphere")
#   - nothing was flagged as unclear or missing, and the answer hedged ("likely...
#     possibly...") before admitting the image doesn't say.
# Fix: separate what is seen from what is inferred, allow an honest third point,
# add a required "Unclear or missing" section, and make the answer lead with
# whether the source actually contains it.
SYSTEM_PROMPT_V2 = """You are a careful multimodal assistant.
Use ONLY the evidence in the provided image or transcript. Do not use outside
knowledge or assume a setting, role, or purpose the source does not show.

Respond in this exact format:
**Summary:** 2-3 sentences describing only what is directly visible or said.
**Key points:**
1. ...
2. ...
3. ...
**Answer:** Answer the user's question if one was asked; otherwise write "No question asked."
**Unclear or missing:** List anything blurry, cut off, inaudible, ambiguous, or absent
that matters for understanding the content. Write "None" only if nothing is.

Rules:
- Each key point must be something you can point to in the source. For text, quote it.
- Mark anything you infer rather than directly see or hear with "(inferred)".
- If the source only supports fewer than three meaningful points, write
  "Not enough evidence for another point." instead of adding filler.
- For the answer, first say whether the source contains it. If it does not, start
  with "The source does not show this." and then say what is missing.
- If text, numbers, or details are blurry, cut off, inaudible, or missing, say so
  explicitly instead of guessing.
- Do not invent exact values (e.g., chart numbers or coordinates) you cannot read clearly.
"""

PROMPTS = {"Original prompt": SYSTEM_PROMPT, "Revised prompt": SYSTEM_PROMPT_V2}

# ---------------------------------------------------------------------------
# 2. CLIENT (reuse your existing OpenAI setup / secret name)
# ---------------------------------------------------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ---------------------------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------------------------
def prepare_image(upload):
    """Encode an uploaded image as a base64 data URL block for the Responses API."""
    mime = IMAGE_FORMATS[Path(upload.name).suffix.lower()]          # Step 1: MIME type
    encoded = base64.b64encode(upload.getvalue()).decode("utf-8")   # Step 2: base64 payload
    return {
        "type": "input_image",
        "image_url": f"data:{mime};base64,{encoded}",
    }


def transcribe_audio(client, upload):
    """Send binary audio to the transcription endpoint and return the transcript text.
    Note: transcription does not preserve every sound, pause, or speaker cue
    (this is the Telephone Problem / Path 1: LLM + Tools)."""
    transcript = client.audio.transcriptions.create(
        model=TRANSCRIPTION_MODEL,
        file=(
            upload.name,
            upload.getvalue(),
            AUDIO_FORMATS[Path(upload.name).suffix.lower()],
        ),
    )
    return transcript.text


def build_content(question, image_block=None, transcript=None):
    """Combine the user's question with EITHER an image block OR transcript text."""
    q = question.strip() if question else ""
    question_text = f"User question: {q}" if q else "User question: (none)"

    content = [{"type": "input_text", "text": question_text}]
    if image_block is not None:
        content.append(image_block)
    if transcript is not None:
        content.append({"type": "input_text", "text": f"Audio transcript:\n{transcript}"})
    return content


def analyze_content(client, content, prompt=SYSTEM_PROMPT):
    """Same analysis function for image content or transcript text."""
    response = client.responses.create(
        model=MODEL,
        instructions=prompt,
        input=[{"role": "user", "content": content}],
        max_output_tokens=600,
    )
    return response.output_text


def validate_upload(upload, allowed_formats):
    """Return an error message string, or None if the upload is OK."""
    suffix = Path(upload.name).suffix.lower()
    if suffix not in allowed_formats:
        return f"Unsupported file type: {suffix}"
    if upload.size > MAX_BYTES:
        return f"File is {upload.size / 1_048_576:.1f} MB; the limit is 5 MB."
    return None


def wav_duration_seconds(upload):
    """Duration check for WAV files (standard library only).
    Returns None for MP3/M4A, where we rely on the 5 MB limit instead."""
    if Path(upload.name).suffix.lower() != ".wav":
        return None
    try:
        with wave.open(io.BytesIO(upload.getvalue())) as w:
            return w.getnframes() / float(w.getframerate())
    except wave.Error:
        return None


# ---------------------------------------------------------------------------
# 4. UI — Lab Part A: selector, uploader, preview
# ---------------------------------------------------------------------------
st.title("Multimodal Lab")
st.write("Upload an image or a short audio clip to get a summary, three key points, "
         "and an answer to your question.")

mode = st.radio("Input type", ["Image", "Audio"], horizontal=True)

if mode == "Image":
    formats = IMAGE_FORMATS
    upload = st.file_uploader("Upload an image (PNG/JPG/JPEG, max 5 MB)",
                              type=[s.lstrip(".") for s in formats])
else:
    formats = AUDIO_FORMATS
    upload = st.file_uploader("Upload audio (MP3/WAV/M4A, max 5 MB, ≤ 60 seconds)",
                              type=[s.lstrip(".") for s in formats])

error = None
if upload is not None:
    error = validate_upload(upload, formats)
    if error is None and mode == "Audio":
        seconds = wav_duration_seconds(upload)
        if seconds is not None and seconds > MAX_AUDIO_SECONDS:
            error = f"Audio is {seconds:.0f} seconds; the limit is {MAX_AUDIO_SECONDS}."

    if error:
        st.error(error)
    else:
        st.subheader("Preview")
        if mode == "Image":
            st.image(upload, use_container_width=True)
        else:
            st.audio(upload)

# ---------------------------------------------------------------------------
# 5. UI — Lab Part B: optional question, Analyze button, results
# ---------------------------------------------------------------------------
question = st.text_input("Optional question about the file",
                         placeholder="e.g., What is the main takeaway of this slide?")

prompt_choice = st.radio(
    "Prompt",
    ["Original prompt", "Revised prompt", "Compare both"],
    horizontal=True,
    help="Use 'Compare both' for the before/after screenshot.",
)

if st.button("Analyze", type="primary", disabled=(upload is None or error is not None)):
    try:
        if mode == "Image":
            with st.spinner("Preparing image..."):
                content = build_content(question, image_block=prepare_image(upload))
        else:
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(client, upload)
            with st.expander("Transcript (check for transcription errors)"):
                st.write(transcript if transcript else "(empty transcript)")
            content = build_content(question, transcript=transcript)

        if prompt_choice == "Compare both":
            with st.spinner("Analyzing with both prompts..."):
                before = analyze_content(client, content, SYSTEM_PROMPT)
                after = analyze_content(client, content, SYSTEM_PROMPT_V2)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Before: original prompt")
                st.markdown(before)
            with col2:
                st.subheader("After: revised prompt")
                st.markdown(after)
        else:
            with st.spinner("Analyzing..."):
                result = analyze_content(client, content, PROMPTS[prompt_choice])
            st.subheader(f"Result ({prompt_choice.lower()})")
            st.markdown(result)
    except Exception as e:
        st.error(f"Something went wrong: {e}")

with st.sidebar.expander("Show prompts"):
    st.markdown("**Original prompt**")
    st.code(SYSTEM_PROMPT, language=None)
    st.markdown("**Revised prompt**")
    st.code(SYSTEM_PROMPT_V2, language=None)

# ---------------------------------------------------------------------------
# 6. REFLECTION COMMENTS
# Check these against the exact reflection questions in the assignment.
#
# - Tests run: (1) a slide: [TODO], (2) a photo of a woman in a white coat at a
#   table with a notebook and pen, (3) a short audio clip: [TODO].
#
# - How the output compared with the source (photo): the visible details were
#   right (white coat, glasses, smile, notebook, pen, wood paneling). But the
#   model turned guesses into facts: "medical coat" became a "medical setting"
#   and then a "clinical consultation", none of which the image shows. The rule
#   "exactly three key points" pushed it to add a filler point about the wood
#   paneling. It flagged nothing as unclear or missing.
#   Slide: [TODO: what it read correctly, any misread text or numbers]
#   Audio: [TODO: transcription errors? tone or speaker cues lost?]
#
# - Prompt revision: I revised SYSTEM_PROMPT into SYSTEM_PROMPT_V2. It requires
#   every key point to be traceable to the source, marks inferences with
#   "(inferred)", allows "Not enough evidence for another point" instead of
#   filler, adds a required "Unclear or missing" section, and makes the answer
#   start by saying whether the source contains it. Result: [TODO: describe the
#   before/after from the "Compare both" screenshot].
#
# - Image vs. audio path: the image is sent directly as pixels (input_image);
#   audio is transcribed first and only text reaches the model (Path 1: LLM +
#   Tools), so tone, pauses and speaker cues are lost (the Telephone Problem),
#   and any transcription error carries into the answer.
# ---------------------------------------------------------------------------
