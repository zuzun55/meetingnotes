import streamlit as st
from transcription import transcribe_audio
import tempfile
import os
import subprocess

# -----------------------
# Extractive Meeting Summary
# -----------------------
def generate_meeting_summary(transcript):
    """Classify sentences into Decisions, Action Items, and Key Points"""
    sentences = transcript.split(". ")
    decisions, actions, key_points = [], [], []

    decision_keywords = ["decided", "agreed", "approved", "resolution", "conclusion"]
    action_keywords = ["will", "need to", "plan to", "assign", "responsible", "action"]
    
    for sent in sentences:
        lowered = sent.lower()
        if any(k in lowered for k in decision_keywords):
            decisions.append(sent)
        elif any(k in lowered for k in action_keywords):
            actions.append(sent)
        else:
            key_points.append(sent)

    summary = {
        "Decisions": ". ".join(decisions) if decisions else "None",
        "Action Items": ". ".join(actions) if actions else "None",
        "Key Points": ". ".join(key_points) if key_points else "None",
    }
    return summary

# -----------------------
# Extract audio from video
# -----------------------
def extract_audio_from_video(video_path, audio_path):
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "mp3", audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# -----------------------
# Streamlit UI
# -----------------------
st.title("🎤 Meeting Notes Transcriber & Summarizer")

uploaded_file = st.file_uploader(
    "Upload an audio or video file",
    type=["mp3", "wav", "m4a", "mp4", "mov", "avi", "mkv"]
)

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Extract audio if video
    if suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
        audio_path = tmp_path.replace(suffix, ".mp3")
        extract_audio_from_video(tmp_path, audio_path)
    else:
        audio_path = tmp_path

    st.info("⏳ Transcribing...")
    result = transcribe_audio(audio_path, model_size="base")
    transcript = result["text"]

    st.subheader("📜 Transcript")
    st.text_area("Transcript", transcript, height=300)

    # Generate structured summary
    summary = generate_meeting_summary(transcript)
    st.subheader("📝 Meeting Minutes Summary")
    for section, content in summary.items():
        st.markdown(f"**{section}:** {content}")

    # Downloads
    st.download_button("⬇️ Download Transcript", transcript, "transcript.txt")
    st.download_button(
        "⬇️ Download Meeting Summary",
        "\n".join(f"{k}: {v}" for k, v in summary.items()),
        "meeting_summary.txt"
    )

    # Detailed segments
    with st.expander("🔎 Detailed Segments"):
        for seg in result["segments"]:
            st.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}")
