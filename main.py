from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uuid, shutil, json, re
from analyze import transcribe_audio, match_speakers, analyze_with_local_llm
from diarization import diarize_audio
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_json(raw_output):
    if not raw_output or not isinstance(raw_output, str):
        print("❌ extract_json: Got None or non-string:", raw_output)
        return None

    # Try matching complete JSON
    match = re.search(r'\{.*?\}', raw_output.strip(), re.DOTALL)
    if match:
        print("✅ Matched full JSON")
        return match.group(0)

    # Try recovering incomplete JSON (missing })
    partial = re.search(r'\{.*', raw_output.strip(), re.DOTALL)
    if partial:
        print("⚠️ Matched partial JSON — adding missing '}'")
        return partial.group(0).strip() + "}"

    print("❌ No JSON detected in output")
    return None

@app.post("/analyze-call")
async def analyze_call(file: UploadFile = File(...)):
    filename = f"temp_{uuid.uuid4()}.wav"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"✅ File saved: {filename}")

    whisper_result = transcribe_audio(filename)
    print("🟡 Whisper result:", whisper_result)

    speaker_segments = diarize_audio(filename)
    print("🟡 Speaker segments:", speaker_segments)

    transcript_with_speakers = match_speakers(whisper_result, speaker_segments)
    print("🟡 Transcript with speakers:\n", transcript_with_speakers)

    llama_output = analyze_with_local_llm(transcript_with_speakers)
    print("🔷 LLaMA raw output:\n", llama_output)

    json_text = extract_json(llama_output)

    if json_text:
        try:
            analysis = json.loads(json_text)
        except Exception as e:
            print("❌ JSON parse failed:", e)
            print("🧾 Raw extracted JSON:", json_text)
            analysis = {"error": "LLaMA output could not be parsed"}
    else:
        analysis = {"error": "LLaMA gave no usable output"}

    analysis["speaker_transcript"] = transcript_with_speakers
    print("✅ Final Analysis:", analysis)

    return analysis

app.mount("/", StaticFiles(directory="static", html=True), name="static")