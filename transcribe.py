import streamlit as st
import whisper
from collections import Counter
import tempfile

# -----------------------
# Summarizer
# -----------------------
def summarize_text(text, sentence_count=5):
    sentences = text.split(". ")
    words = text.lower().split()
    word_freq = Counter(words)

    sentence_scores = {}
    for sent in sentences:
        for word in sent.split():
            if word.lower() in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word.lower()]

    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    return ". ".join(ranked_sentences[:sentence_count])

# -----------------------
# Streamlit UI
# -----------------------
st.title("Meeting Audio/Video Transcriber & Summarizer")
st.write(
    "Upload an audio or video file, and get the transcript and summary instantly. "
    "No installation needed—everything runs in your browser!"
)

uploaded_file = st.file_uploader(
    "Upload audio/video file", type=["mp3", "wav", "m4a", "mp4"]
)

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    with st.spinner("⏳ Loading Whisper model..."):
        model = whisper.load_model("small")  # you can use "tiny" or "base" for faster load

    with st.spinner("🎤 Transcribing... This may take a minute"):
        result = model.transcribe(temp_path, language="en")
        transcript = result["text"]

    st.subheader("📜 Transcript")
    st.text_area("Transcript", transcript, height=300)

    summary = summarize_text(transcript)
    st.subheader("📝 Summary")
    st.text_area("Summary", summary, height=200)

    st.download_button("Download Transcript", transcript, "transcript.txt")
    st.download_button("Download Summary", summary, "summary.txt")
