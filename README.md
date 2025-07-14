# 🎧 AI Call Analyzer

A powerful AI-driven tool that transcribes, diarizes, and analyzes customer support calls using Whisper + PyAnnote + LLaMA 7B locally!

![Banner](https://img.shields.io/badge/LLM-LLaMA%207B-purple?style=for-the-badge) ![Whisper](https://img.shields.io/badge/Transcription-Whisper-blue?style=for-the-badge) ![FastAPI](https://img.shields.io/badge/API-FastAPI-teal?style=for-the-badge)

---

## 🔍 Features

- 🎙️ **Speaker Diarization** – Identify who said what using PyAnnote.
- 📝 **Transcription** – Convert audio to text using Whisper.
- 🧠 **LLM Analysis** – Analyze conversation tone, category & generate summaries using Mistral-7B or LLaMA-7B.
- 🎧 **Beautiful Frontend** – Upload `.wav` files and see the analysis in an aesthetic UI.
- 🖥️ **Runs Locally** – No external APIs needed (perfect for private data analysis).

---

## 🚀 How It Works

1. Upload a `.wav` call recording.
2. Backend transcribes it using Whisper.
3. Speakers are diarized via PyAnnote.
4. Local LLaMA analyzes the transcript and categorizes the call.
5. Results are shown beautifully on the web interface.

---

## 💻 Tech Stack

- 🧠 LLaMA / Mistral (via `llama.cpp`)
- 🔊 Whisper (OpenAI)
- 🗣️ PyAnnote (Speaker Diarization)
- ⚡ FastAPI
- 🌐 HTML + CSS + JS (with Lottie animations!)

---

## 🛠️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Daksh-bairagi/ai-call-analyzer.git
   cd ai-call-analyzer
