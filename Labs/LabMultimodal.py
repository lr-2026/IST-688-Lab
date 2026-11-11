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
**Answer:** Answer the user's question, which appears at the END of the user's message.
Only write "No question asked." if the message says "The user did not ask a question."

Rules:
- Give exactly three key points.
- If text, numbers, or details are blurry, cut off, inaudible, or missing, say so
  explicitly instead of guessing.
- Do not invent exact values (e.g., chart numbers or coordinates) you cannot read clearly.
"""

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
    question_text = (f"The user asked this question about the content above: {q}"
                     if q else "The user did not ask a question.")

    content = []
    if image_block is not None:
        content.append(image_block)
    if transcript is not None:
        content.append({"type": "input_text", "text": f"Audio transcript:\n{transcript}"})
    # Question goes LAST so the model sees the evidence first, then the question
    content.append({"type": "input_text", "text": question_text})
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
            # Embed the image as a data URL (same one prepare_image builds),
            # which avoids Streamlit's media server and the broken-image icon.
            data_url = prepare_image(upload)["image_url"]
            st.markdown(f'<img src="{data_url}" width="600">', unsafe_allow_html=True)
        else:
            st.audio(upload)

# ---------------------------------------------------------------------------
# 5. UI — Lab Part B: optional question, Analyze button, results
# ---------------------------------------------------------------------------
# A form submits the question and the button click together,
# so the typed question is never lost.
with st.form("analyze_form"):
    question = st.text_input("Optional question about the file",
                             placeholder="e.g., What is the main takeaway of this slide?")
    submitted = st.form_submit_button("Analyze", type="primary",
                                      disabled=(upload is None or error is not None))

if submitted:
    st.caption(f"Question sent: {question.strip() or '(none)'}")
    try:
        if mode == "Image":
            with st.spinner("Analyzing image..."):
                content = build_content(question, image_block=prepare_image(upload))
                result = analyze_content(client, content)
        else:
            with st.spinner("Transcribing audio (call 1 of 2)..."):
                transcript = transcribe_audio(client, upload)
            with st.expander("Transcript (check for transcription errors)"):
                st.write(transcript if transcript else "(empty transcript)")
            with st.spinner("Analyzing transcript (call 2 of 2)..."):
                content = build_content(question, transcript=transcript)
                result = analyze_content(client, content)

        st.subheader("Result")
        st.markdown(result)
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# ---------------------------------------------------------------------------
# 6. REFLECTION COMMENTS
# Replace these with the exact reflection questions from your assignment.
#
# - Tests run: (1) a slide, (2) another image, (3) a short audio clip.
# - How the output compared with the source: what was accurate, what was
#   missed or hallucinated?
# - Prompt revision: which prompt did I change, what did I change, and did
#   the output improve?
# - Image vs. audio path: the image is sent directly as pixels (input_image);
#   audio is transcribed first and only text reaches the model (Path 1),
#   so tone and speaker cues are lost.
# ---------------------------------------------------------------------------