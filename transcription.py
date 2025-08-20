from faster_whisper import WhisperModel

def transcribe_audio(file_path: str, model_size: str = "base"):
    """
    Transcribes audio using faster-whisper.
    Args:
        file_path (str): Path to audio file.
        model_size (str): Model size ("tiny", "base", "small", "medium", "large-v2").
    Returns:
        dict: Transcription text and segments.
    """

    # Load model (CPU optimized with int8, switch to cuda if GPU available)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Run transcription
    segments, info = model.transcribe(file_path)

    # Collect results
    transcript_text = ""
    segment_list = []
    for segment in segments:
        transcript_text += segment.text + " "
        segment_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })

    return {
        "language": info.language,
        "text": transcript_text.strip(),
        "segments": segment_list
    }
