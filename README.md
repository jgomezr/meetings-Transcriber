Real-time audio transcription with speaker detection and AI-powered meeting summaries. Runs 100% locally on your machine.

## Features

- **Real-time transcription** using Faster-Whisper (CPU optimized)
- **Speaker detection** using Resemblyzer voice embeddings
- **AI meeting summaries** using Google Gemma 3n via WebGPU
- **Multi-language support** (English, Spanish, Auto-detect)
- **Export to TXT** with transcription and summary
- **100% offline** - no data leaves your computer

## Requirements

- Windows 10/11
- Python 3.10 or higher
- Modern browser with WebGPU support (Chrome 113+, Edge 113+)
- ~4GB RAM for transcription
- ~4GB VRAM for AI summaries (WebGPU)

## Installation

### 1. Install dependencies

Double-click `instalar.bat` or run:

```batch
instalar.bat
```

This will:
- Create a Python virtual environment
- Install Faster-Whisper for transcription
- Install Resemblyzer for speaker detection
- Install required dependencies

### 2. Download the Gemma 3n model (for AI summaries)

1. Go to: https://huggingface.co/google/gemma-3n-E2B-it-litert-lm
2. Download `gemma-3n-E2B-it-int4-Web.litertlm` (~3GB)
3. Place it in the `models/` folder

```
transcriber/
├── models/
│   └── gemma-3n-E2B-it-int4-Web.litertlm  <-- here
```

## Usage

### Start the application

Double-click `iniciar.vbs` (runs without CMD window)

Or use `iniciar.bat` if you want to see the console output.

### Recording

1. Select your **Audio Source** (microphone or Stereo Mix for system audio)
2. Select the **Language** (English, Spanish, or Auto)
3. Select the **Interval** (how often to process audio chunks)
4. Click **Start** to begin recording
5. Click **Stop** when finished

### Generate Summary

1. After recording, click **Generate Summary**
2. The AI will analyze the transcription and create a structured summary
3. Summary includes: Main Topics, Key Decisions, Action Items

### Export

Click **Export TXT** to download a file containing:
- Full transcription with speaker labels
- AI-generated summary (if generated)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                              │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │  Audio Capture  │  │         Gemma 3n (WebGPU)        │  │
│  │  (MediaRecorder)│  │         Meeting Summaries        │  │
│  └────────┬────────┘  └─────────────────────────────────┘  │
│           │                                                  │
└───────────┼──────────────────────────────────────────────────┘
            │ WebM audio
            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Python Server (localhost:8765)            │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │  Faster-Whisper │  │         Resemblyzer              │  │
│  │  Transcription  │  │      Speaker Detection           │  │
│  │  (CPU, int8)    │  │    (Voice Embeddings)            │  │
│  └─────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `app.py` | Main application (HTTP server + HTML/JS UI) |
| `iniciar.vbs` | Launch script (no CMD window) |
| `iniciar.bat` | Launch script (with CMD window) |
| `instalar.bat` | Installation script |
| `models/` | Folder for Gemma 3n model |

## Configuration

### Similarity Threshold

Speaker detection uses a similarity threshold of **0.85** (85%). Adjust in `app.py`:

```python
SIMILARITY_THRESHOLD = 0.85
```

- Higher value (0.90+): More strict, may create too many speakers
- Lower value (0.75-): More lenient, may merge different speakers

### Whisper Model

Default model is `base`. Change in `app.py`:

```python
MODEL_NAME = "base"
```

Available models:
- `tiny` - Fastest, least accurate
- `base` - Good balance (default)
- `small` - Better accuracy, slower
- `medium` - High accuracy, slow
- `large-v3` - Best accuracy, very slow

### Max Tokens for Summary

Default is 4096 tokens. Adjust in the JavaScript section of `app.py`:

```javascript
maxTokens: 4096
```

## Troubleshooting

### "No microphones found"
- Allow microphone access in browser settings
- Check Windows sound settings for enabled devices

### "WebGPU not available"
- Use Chrome 113+ or Edge 113+
- Enable WebGPU in browser flags if needed

### "Error: No model format matched"
- Make sure you downloaded `gemma-3n-E2B-it-int4-Web.litertlm` (Web version)
- Not the regular `.litertlm` file

### Speaker detection not working
- Check console for "Resemblyzer loaded" message
- Ensure `webrtcvad-wheels` is installed

### Summary too short/truncated
- Long transcriptions are truncated to ~6000 characters
- This is a limitation of the model's context window

## Capturing System Audio (Teams/Zoom)

To transcribe audio from Teams, Zoom, or other applications:

1. Enable **Stereo Mix** in Windows Sound settings:
   - Right-click speaker icon → Sounds → Recording tab
   - Right-click → Show Disabled Devices
   - Enable "Stereo Mix"

2. Or use a virtual audio cable like [VB-Audio](https://vb-audio.com/Cable/)

3. Select the audio device in the app's **Audio Source** dropdown

## Technologies

- **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** - CTranslate2-based Whisper implementation
- **[Resemblyzer](https://github.com/resemble-ai/Resemblyzer)** - Voice embeddings for speaker detection
- **[MediaPipe LLM Inference](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference)** - On-device LLM inference
- **[Gemma 3n](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm)** - Google's efficient language model
- **[Marked](https://marked.js.org/)** - Markdown parser for summary rendering

## License

MIT License

