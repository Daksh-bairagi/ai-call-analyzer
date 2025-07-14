import whisper
from llama_cpp import Llama

# Load Whisper
whisper_model = whisper.load_model("base")

# Load LLaMA model
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    verbose=True
)

def transcribe_audio(audio_path):
    return whisper_model.transcribe(audio_path, word_timestamps=False)

def match_speakers(transcription, speaker_segments):
    aligned = []
    for seg in transcription["segments"]:
        seg_text = seg['text']
        seg_start = seg['start']
        speaker = "Unknown"

        for s in speaker_segments:
            if s["start"] <= seg_start <= s["end"]:
                speaker = s["speaker"]
                break
        aligned.append(f"{speaker}: {seg_text.strip()}")
    return "\n".join(aligned)

def analyze_with_local_llm(transcript_text):
    prompt = f"""
You are an AI trained to analyze customer support calls.

Given the transcript with speaker labels, return ONLY a valid JSON with:
- category (Billing, Technical, Cancellation, Other)
- summary
- tags (like Angry Customer, Escalation Needed)

Transcript:
\"\"\"
{transcript_text}
\"\"\"
only respond in the json format below without any additional text:
Respond only in this format:
{{
  "category": "...",
  "summary": "...",
  "tags": ["...", "..."]
}}
    """

    output = llm(prompt, max_tokens=512, echo=False)
    text = output["choices"][0]["text"]
    print("🔷 LLaMA raw output:\n", text)
    return text
