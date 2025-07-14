# diarization.py
from pyannote.audio import Pipeline

HUGGINGFACE_TOKEN = ""  # Replace with your token

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_TOKEN
)

def diarize_audio(audio_path):
    diarization = pipeline(audio_path)
    
    # Save RTTM (optional)
    with open("audio.rttm", "w") as rttm:
        diarization.write_rttm(rttm)

    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })
    return result
