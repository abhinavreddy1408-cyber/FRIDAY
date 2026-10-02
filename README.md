# F.R.I.D.A.Y. — Personal AI Voice Assistant

> *"F.R.I.D.A.Y. online. All systems nominal."*

An always-on, voice-first AI assistant inspired by Iron Man.  
Runs silently in your system tray, activates on wake word, executes commands instantly.

---

## Architecture

```
┌──────────────────────────────────────────────┐
│                 SYSTEM TRAY                  │
│        (tray.py — main thread)               │
│   Blue=Idle  Green=Listening  Amber=Thinking │
└────────────────────┬─────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │       ENGINE        │
          │  (engine.py — bg    │
          │    thread)          │
          │                     │
          │  State Machine:     │
          │  IDLE → LISTENING → │
          │  PROCESSING →       │
          │  SPEAKING →         │
          │  CONVERSATION       │
          └──┬────┬────┬────┬───┘
             │    │    │    │
     ┌───────┘    │    │    └────────┐
     ▼            ▼    ▼             ▼
 wake_word.py  stt.py  actions.py  llm.py
 (Vosk/Google)        (instant)   (NVIDIA)
                                  
                tts.py
        (Kokoro/EdgeTTS/pyttsx3)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
Edit `.env`:
```env
NVIDIA_API_KEY=your_nvidia_api_key
```

### 3. Voice Setup (First Time Only)
You must teach F.R.I.D.A.Y. your voice signature before she will respond.
```bash
python enroll.py
```
*Follow the wizard to read 5 phrases and generate your secure voice profile.*

### 4. Launch
```bash
python main.py          # With console output
pythonw main.py         # Silent (no console window)
```

A blue circle icon with **F** appears in your system tray.  
Say **"Friday"** to activate.

---

## Security & Audio Subsystems

### 1. Voice Identity (Speaker Verification)
F.R.I.D.A.Y. uses **SpeechBrain (ECAPA-TDNN)** to mathematically fingerprint your voice. She will **reject** commands from unauthorized speakers.
* **Emergency Override:** If your voice is unrecognizable (e.g., you are sick), use the passphrase by saying: *"Override protocol Alpha, [your command]"*.

### 2. Advanced Noise Cancellation
Raw microphone audio is automatically piped through **Noisereduce** spectral gating to eliminate background fans, echoes, and static in real-time, greatly improving voice transcription in loud environments.

---

## Voice Commands

### Instant (Local — no LLM)
| Command | Action |
|---|---|
| "Friday, open Chrome" | Launches Chrome |
| "Friday, open YouTube" | Opens youtube.com |
| "Friday, search machine learning" | Google search |
| "Friday, play lo-fi on YouTube" | YouTube search |
| "Friday, what time is it" | Tells current time |
| "Friday, what's the date" | Tells current date |
| "Friday, volume up / down / mute" | System volume |
| "Friday, next track / pause" | Media controls |
| "Friday, close Chrome" | Kills the process |

### Conversational (via LLM)
| Command | Action |
|---|---|
| "Friday, explain quantum computing" | LLM response |
| "Friday, help me write an email" | LLM response |
| Any complex question | LLM response |

### Follow-up Mode
After any command, FRIDAY stays in **conversation mode** for 15 seconds.  
You can ask follow-up questions without saying "Friday" again.

---

## Project Structure

```
F.R.I.D.A.Y/
├── main.py               # Entry point
├── requirements.txt      # Dependencies
├── .env                  # API keys
│
└── friday/               # Core package
    ├── config.py          # Settings from .env
    ├── personality.py     # System prompt & voice lines
    ├── sounds.py          # Audio indicators (beeps)
    ├── wake_word.py       # Wake word detection (Vosk / Google)
    ├── stt.py             # Speech-to-Text (command capture)
    ├── tts.py             # Text-to-Speech (Kokoro + EdgeTTS + pyttsx3)
    ├── llm.py             # LLM Brain (NVIDIA Nemotron)
    ├── actions.py         # Local action router (instant)
    ├── engine.py          # Core state-machine coordinator
    └── tray.py            # System tray icon
```

---

## System Tray States

| Colour | Meaning |
|---|---|
| 🔵 Blue | Idle — listening for wake word |
| 🟢 Green | Listening — capturing your command |
| 🟡 Amber | Processing — thinking / executing |
| 🟣 Purple | Speaking — playing response |
| 🔴 Red | Offline |

---

## NVIDIA NIM Ensembles

F.R.I.D.A.Y. uses NVIDIA NIM as the only LLM backend. The default
`NVIDIA_API_KEY` is used for every model unless a per-role key is provided.

Per-role keys use this format:
```env
FRIDAY_NIM_KEY_CODING_PRIMARY_CODER=your_optional_role_key
FRIDAY_NIM_KEY_REASONING_LOGIC_VERIFIER=your_optional_role_key
FRIDAY_NIM_KEY_SAFETY_CONTENT_SAFETY=your_optional_role_key
```

Registry aliases such as `_` and `_provided` mean "use the configured role key
or fall back to `NVIDIA_API_KEY`"; they are not literal API keys. Final ensemble
outputs are checked by both safety models before delivery. Missing or ambiguous
safety verdicts are blocked by default.

## Switching Default LLM Model

Change `NVIDIA_MODEL` in `.env`:
```env
NVIDIA_MODEL=nvidia/nemotron-3-super-120b-a12b    # Default
NVIDIA_MODEL=meta/llama-3.3-70b-instruct           # Llama
NVIDIA_MODEL=deepseek-ai/deepseek-v4-pro           # DeepSeek
```

---

## Optional: Offline Wake Word (Vosk)

For fully offline wake-word detection:
```bash
pip install vosk
mkdir models
# Download small model (~40MB):
# https://alphacephei.com/vosk/models → vosk-model-small-en-us-0.15
# Extract to models/vosk-model-small-en-us/
```

Without Vosk, wake-word detection uses Google STT (requires internet).

---

*"Shutting down, Sir. Try not to do anything reckless while I'm offline."*
