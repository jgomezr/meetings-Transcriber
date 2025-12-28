"""
Whisper Transcriber - Aplicacion Local
Transcripcion en tiempo real con deteccion de hablantes
Usa servidor HTTP local + navegador del sistema para acceso completo al microfono
"""

import threading
import json
import tempfile
import os
import sys
import numpy as np
from pathlib import Path
import base64
import logging
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuracion
MODEL_NAME = "base"
SIMILARITY_THRESHOLD = 0.85
PORT = 8765

# Estado global
whisper_model = None
voice_encoder = None
speaker_embeddings = {}
current_speaker = "A"
is_ready = False
loading_status = "Starting..."


def load_models():
    """Cargar modelos en segundo plano"""
    global whisper_model, voice_encoder, is_ready, loading_status

    try:
        loading_status = "Loading Whisper..."
        logger.info("Loading Whisper model...")

        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
        logger.info("Whisper loaded")

        loading_status = "Loading Resemblyzer..."
        logger.info("Loading Resemblyzer...")

        try:
            from resemblyzer import VoiceEncoder
            voice_encoder = VoiceEncoder()
            logger.info("Resemblyzer loaded")
        except Exception as e:
            logger.warning(f"Resemblyzer not available: {e}")
            voice_encoder = None

        is_ready = True
        loading_status = "Ready"
        logger.info("System ready")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        loading_status = f"Error: {e}"


def load_audio_for_embedding(audio_path):
    """Cargar audio usando el decodificador de faster-whisper (av/ffmpeg)"""
    try:
        # Usar el mismo decoder que usa faster-whisper
        from faster_whisper.audio import decode_audio

        # decode_audio devuelve numpy array a 16kHz mono
        wav = decode_audio(audio_path, sampling_rate=16000)
        logger.info(f"Audio decodificado: {len(wav)} samples, {len(wav)/16000:.2f}s")
        return wav
    except Exception as e:
        logger.error(f"Error decodificando audio: {e}")
        return None


def get_voice_embedding(audio_path):
    """Obtener embedding de voz"""
    global voice_encoder

    if voice_encoder is None:
        logger.warning("VoiceEncoder no disponible")
        return None

    try:
        from resemblyzer import preprocess_wav

        # Cargar audio usando el decodificador de faster-whisper
        wav = load_audio_for_embedding(audio_path)

        if wav is None:
            return None

        if len(wav) < 1600:  # Menos de 0.1 segundos
            logger.warning(f"Audio muy corto: {len(wav)} samples")
            return None

        # Preprocesar para Resemblyzer
        wav = preprocess_wav(wav)
        logger.info(f"Audio preprocesado: {len(wav)} samples")

        if len(wav) < 1600:
            logger.warning("Audio muy corto despues de preprocesar")
            return None

        # Obtener embedding
        embedding = voice_encoder.embed_utterance(wav)
        logger.info(f"Embedding generado: shape {embedding.shape}")
        return embedding

    except Exception as e:
        logger.error(f"Error obteniendo embedding: {e}")
        import traceback
        traceback.print_exc()
        return None


def detect_speaker(embedding):
    """Detectar hablante"""
    global speaker_embeddings, current_speaker

    if embedding is None:
        logger.warning("No hay embedding, usando hablante actual: " + current_speaker)
        return current_speaker

    if not speaker_embeddings:
        speaker_embeddings["A"] = [embedding]
        current_speaker = "A"
        logger.info("Primer hablante registrado: A")
        return "A"

    best_speaker = None
    best_similarity = -1

    for speaker_id, embeddings_list in speaker_embeddings.items():
        similarities = [np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb))
                      for emb in embeddings_list]
        avg_similarity = np.mean(similarities)
        logger.info(f"Similitud con {speaker_id}: {avg_similarity:.3f} (de {len(embeddings_list)} muestras)")

        if avg_similarity > best_similarity:
            best_similarity = avg_similarity
            best_speaker = speaker_id

    logger.info(f"Mejor match: {best_speaker} con similitud {best_similarity:.3f} (umbral: {SIMILARITY_THRESHOLD})")

    if best_similarity >= SIMILARITY_THRESHOLD:
        if len(speaker_embeddings[best_speaker]) < 10:
            speaker_embeddings[best_speaker].append(embedding)
        current_speaker = best_speaker
        return best_speaker

    # Nuevo hablante detectado
    new_speaker = chr(ord('A') + len(speaker_embeddings))
    if len(speaker_embeddings) < 10:
        speaker_embeddings[new_speaker] = [embedding]
        current_speaker = new_speaker
        logger.info(f"Nuevo hablante detectado: {new_speaker}")
        return new_speaker

    return best_speaker


def reset_speakers():
    """Reiniciar deteccion de hablantes"""
    global speaker_embeddings, current_speaker
    speaker_embeddings = {}
    current_speaker = "A"


def transcribe_audio(audio_base64, language="es"):
    """Transcribir audio en base64"""
    global whisper_model, voice_encoder, speaker_embeddings

    if not is_ready:
        return {"error": "Modelo no cargado"}

    try:
        # Decodificar audio
        audio_data = base64.b64decode(audio_base64)

        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            # Detectar hablante
            speaker = None
            if voice_encoder:
                embedding = get_voice_embedding(tmp_path)
                speaker = detect_speaker(embedding)

            # Transcribir
            segments, info = whisper_model.transcribe(
                tmp_path,
                language=language if language != "auto" else None,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            text_parts = [segment.text.strip() for segment in segments]
            full_text = " ".join(text_parts)

            return {
                "text": full_text,
                "speaker": speaker,
                "language": info.language,
                "total_speakers": len(speaker_embeddings)
            }

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Error transcribiendo: {e}")
        return {"error": str(e)}


# HTML de la aplicacion
HTML_CONTENT = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meetings Transcriber</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        .container { max-width: 1400px; margin: 0 auto; }

        header { text-align: center; margin-bottom: 25px; }
        h1 {
            font-size: 2em;
            background: linear-gradient(135deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #888; margin-top: 5px; }

        .status-bar {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,255,255,0.1);
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.85em;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #666;
        }
        .status-dot.ready { background: #00ff88; }
        .status-dot.recording { background: #ff4757; animation: pulse 1s infinite; }
        .status-dot.loading { background: #ffa502; animation: pulse 1s infinite; }

        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        .settings {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .setting-item { display: flex; flex-direction: column; gap: 5px; flex: 1; min-width: 150px; }
        .setting-item label { font-size: 0.8em; color: #aaa; }
        .setting-item select {
            padding: 8px;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(0,0,0,0.3);
            color: white;
        }

        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        button {
            padding: 12px 30px;
            font-size: 14px;
            font-weight: 600;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        #startBtn {
            background: linear-gradient(135deg, #00d9ff, #00ff88);
            color: #1a1a2e;
        }
        #startBtn.recording {
            background: linear-gradient(135deg, #ff4757, #ff6b81);
            color: white;
        }
        .secondary-btn {
            background: rgba(255,255,255,0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.2);
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 15px;
        }

        @media (max-width: 900px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            min-height: 350px;
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .panel-header h2 { font-size: 1em; color: #00d9ff; }

        .panel-content {
            font-size: 1em;
            line-height: 1.7;
            color: #e0e0e0;
            min-height: 250px;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        .panel-content.empty { color: #666; font-style: italic; }

        .speaker-tag {
            font-weight: bold;
            margin-right: 5px;
        }
        .speaker-A { color: #00d9ff; }
        .speaker-B { color: #00ff88; }
        .speaker-C { color: #ff6b81; }
        .speaker-D { color: #ffa502; }
        .speaker-E { color: #a55eea; }

        #summaryBtn {
            background: linear-gradient(135deg, #a55eea, #7c3aed);
            color: white;
            padding: 8px 16px;
            font-size: 12px;
        }
        #summaryBtn:disabled { opacity: 0.5; }

        .summary-loading {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #a55eea;
        }
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid rgba(165, 94, 234, 0.3);
            border-top-color: #a55eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .model-status {
            font-size: 0.75em;
            color: #888;
            margin-top: 10px;
        }

        .stats {
            display: flex;
            justify-content: center;
            gap: 25px;
        }
        .stat { text-align: center; }
        .stat-value { font-size: 1.3em; font-weight: 600; color: #00d9ff; }
        .stat-label { font-size: 0.8em; color: #888; }

        .footer {
            text-align: center;
            margin-top: 20px;
            color: #666;
            font-size: 0.85em;
        }

        .error-msg {
            background: rgba(255,71,87,0.2);
            border: 1px solid #ff4757;
            color: #ff6b81;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
        }
        /* Markdown styles for summary */
        #summary h1, #summary h2, #summary h3 { color: #00d9ff; margin: 10px 0 5px 0; }
        #summary h1 { font-size: 1.3em; }
        #summary h2 { font-size: 1.1em; }
        #summary h3 { font-size: 1em; }
        #summary ul, #summary ol { margin: 5px 0 5px 20px; }
        #summary li { margin: 3px 0; }
        #summary p { margin: 8px 0; }
        #summary strong { color: #00ff88; }
        #summary code { background: rgba(0,0,0,0.3); padding: 2px 5px; border-radius: 3px; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script type="module">
        import { FilesetResolver, LlmInference } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai';
        window.FilesetResolver = FilesetResolver;
        window.LlmInference = LlmInference;
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>Meetings Transcriber</h1>
            <p class="subtitle">Real-time transcription with speaker detection</p>
        </header>

        <div id="errorContainer"></div>

        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
            <div class="status-item">
                <span id="speakersCount">Speakers: 0</span>
            </div>
        </div>

        <div class="settings">
            <div class="setting-item">
                <label>Audio Source</label>
                <select id="audioSource">
                    <option value="">Loading devices...</option>
                </select>
            </div>
            <div class="setting-item">
                <label>Language</label>
                <select id="language">
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="auto">Auto</option>
                </select>
            </div>
            <div class="setting-item">
                <label>Interval</label>
                <select id="interval">
                    <option value="3000">3 sec</option>
                    <option value="5000" selected>5 sec</option>
                    <option value="10000">10 sec</option>
                </select>
            </div>
        </div>

        <div class="controls">
            <button id="startBtn" disabled>
                <span id="startIcon">&#127908;</span>
                <span id="startText">Start</span>
            </button>
            <button class="secondary-btn" id="clearBtn">Clear</button>
            <button class="secondary-btn" id="copyBtn">Copy</button>
            <button class="secondary-btn" id="exportBtn">Export TXT</button>
        </div>

        <div class="main-content">
            <div class="panel">
                <div class="panel-header">
                    <h2>Transcription</h2>
                </div>
                <div id="transcription" class="panel-content empty">Press Start to begin...</div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <h2>Summary</h2>
                    <button id="summaryBtn" disabled>Generate Summary</button>
                </div>
                <div id="summary" class="panel-content empty">Summary will appear here after generating...</div>
                <div id="modelStatus" class="model-status"></div>
            </div>
        </div>

        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="wordCount">0</div>
                <div class="stat-label">Words</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="duration">00:00</div>
                <div class="stat-label">Duration</div>
            </div>
        </div>

        <div class="footer">
            Whisper + Resemblyzer + Gemma 3n | 100% Local | Close this window to stop the server
        </div>
    </div>

    <script>
        const API_URL = window.location.origin;

        let mediaRecorder = null;
        let isRecording = false;
        let transcriptionInterval = null;
        let startTime = null;
        let durationInterval = null;
        let fullText = '';
        let lastSpeaker = null;

        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const startBtn = document.getElementById('startBtn');
        const startIcon = document.getElementById('startIcon');
        const startTextEl = document.getElementById('startText');
        const transcription = document.getElementById('transcription');
        const speakersCount = document.getElementById('speakersCount');
        const audioSource = document.getElementById('audioSource');
        const errorContainer = document.getElementById('errorContainer');

        function showError(msg) {
            errorContainer.innerHTML = '<div class="error-msg">' + msg + '</div>';
        }

        function clearError() {
            errorContainer.innerHTML = '';
        }

        // Load audio devices
        async function loadDevices() {
            console.log('Loading audio devices...');

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                showError('Your browser does not support microphone access. Use Chrome or Edge.');
                audioSource.innerHTML = '<option value="">Not available</option>';
                return;
            }

            try {
                console.log('Requesting microphone permission...');
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                stream.getTracks().forEach(track => track.stop());

                console.log('Enumerating devices...');
                const devices = await navigator.mediaDevices.enumerateDevices();
                const inputs = devices.filter(d => d.kind === 'audioinput');

                console.log('Audio devices found:', inputs.length);

                if (inputs.length === 0) {
                    audioSource.innerHTML = '<option value="">No microphones found</option>';
                } else {
                    audioSource.innerHTML = inputs.map((d, i) =>
                        '<option value="' + d.deviceId + '">' + (d.label || 'Microphone ' + (i+1)) + '</option>'
                    ).join('');
                    clearError();
                }
            } catch (e) {
                console.error('Error loading devices:', e);
                if (e.name === 'NotAllowedError') {
                    showError('Microphone permission denied. Allow access in browser settings.');
                } else {
                    showError('Error accessing microphone: ' + e.message);
                }
                audioSource.innerHTML = '<option value="">Permission error</option>';
            }
        }

        // Check server status
        async function checkStatus() {
            try {
                const response = await fetch(API_URL + '/status');
                const status = await response.json();

                if (status.ready) {
                    statusDot.className = 'status-dot ready';
                    statusText.textContent = 'Ready';
                    startBtn.disabled = false;
                } else {
                    statusDot.className = 'status-dot loading';
                    statusText.textContent = status.status;
                }

                speakersCount.textContent = 'Speakers: ' + status.speakers;

            } catch (e) {
                statusDot.className = 'status-dot';
                statusText.textContent = 'Disconnected';
                startBtn.disabled = true;
            }
        }

        // Start recording
        async function startRecording() {
            try {
                const deviceId = audioSource.value;
                const constraints = {
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        sampleRate: 16000,
                        channelCount: 1
                    }
                };

                if (deviceId) {
                    constraints.audio.deviceId = { exact: deviceId };
                }

                const stream = await navigator.mediaDevices.getUserMedia(constraints);

                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                let chunks = [];

                mediaRecorder.ondataavailable = e => {
                    if (e.data.size > 0) chunks.push(e.data);
                };

                mediaRecorder.onstop = async () => {
                    if (chunks.length > 0) {
                        const blob = new Blob(chunks, { type: 'audio/webm' });
                        chunks = [];
                        await processAudio(blob);
                    }
                };

                mediaRecorder.start();
                isRecording = true;
                startTime = Date.now();

                startBtn.classList.add('recording');
                startIcon.innerHTML = '&#9209;';
                startTextEl.textContent = 'Stop';
                statusDot.className = 'status-dot recording';
                statusText.textContent = 'Recording...';
                clearError();

                const interval = parseInt(document.getElementById('interval').value);
                transcriptionInterval = setInterval(() => {
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                        setTimeout(() => {
                            if (isRecording) mediaRecorder.start();
                        }, 100);
                    }
                }, interval);

                durationInterval = setInterval(updateDuration, 1000);

            } catch (e) {
                console.error('Error starting recording:', e);
                showError('Error starting recording: ' + e.message);
            }
        }

        function stopRecording() {
            isRecording = false;

            if (transcriptionInterval) clearInterval(transcriptionInterval);
            if (durationInterval) clearInterval(durationInterval);

            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
            }

            startBtn.classList.remove('recording');
            startIcon.innerHTML = '&#127908;';
            startTextEl.textContent = 'Start';
            statusDot.className = 'status-dot ready';
            statusText.textContent = 'Ready';
        }

        async function processAudio(blob) {
            try {
                statusText.textContent = 'Processing...';

                const buffer = await blob.arrayBuffer();
                const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
                const lang = document.getElementById('language').value;

                const response = await fetch(API_URL + '/transcribe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ audio: base64, language: lang })
                });

                const result = await response.json();

                if (result.error) {
                    console.error(result.error);
                    return;
                }

                if (result.text && result.text.trim()) {
                    appendTranscription(result.text, result.speaker);
                    speakersCount.textContent = 'Speakers: ' + result.total_speakers;
                }

                if (isRecording) {
                    statusText.textContent = 'Recording...';
                }

            } catch (e) {
                console.error('Error processing:', e);
            }
        }

        function appendTranscription(text, speaker) {
            if (transcription.classList.contains('empty')) {
                transcription.classList.remove('empty');
                transcription.innerHTML = '';
            }

            if (speaker && speaker !== lastSpeaker) {
                const tag = document.createElement('span');
                tag.className = 'speaker-tag speaker-' + speaker;
                tag.textContent = '\\n[' + speaker + ']: ';
                transcription.appendChild(tag);
                fullText += '\\n[' + speaker + ']: ';
                lastSpeaker = speaker;
            }

            transcription.appendChild(document.createTextNode(text + ' '));
            fullText += text + ' ';

            document.getElementById('wordCount').textContent =
                fullText.trim().split(/\\s+/).filter(w => w).length;

            transcription.scrollTop = transcription.scrollHeight;

            // Enable summary button when there's text
            if (typeof updateSummaryButton === 'function') {
                updateSummaryButton();
            }
        }

        function updateDuration() {
            if (!startTime) return;
            const s = Math.floor((Date.now() - startTime) / 1000);
            document.getElementById('duration').textContent =
                String(Math.floor(s/60)).padStart(2,'0') + ':' + String(s%60).padStart(2,'0');
        }

        // Event listeners
        startBtn.onclick = () => isRecording ? stopRecording() : startRecording();

        document.getElementById('clearBtn').onclick = async () => {
            if (confirm('Clear transcription, summary and reset speakers?')) {
                fullText = '';
                lastSpeaker = null;
                transcription.innerHTML = 'Press Start to begin...';
                transcription.className = 'panel-content empty';
                document.getElementById('wordCount').textContent = '0';
                document.getElementById('duration').textContent = '00:00';
                startTime = null;

                // Clear summary
                currentSummary = '';
                summaryDiv.innerHTML = 'Summary will appear here after generating...';
                summaryDiv.className = 'panel-content empty';

                try {
                    await fetch(API_URL + '/reset', { method: 'POST' });
                    checkStatus();
                } catch (e) {
                    console.error('Error resetting:', e);
                }
            }
        };

        document.getElementById('copyBtn').onclick = () => {
            navigator.clipboard.writeText(fullText).then(() => {
                alert('Text copied to clipboard');
            }).catch(e => {
                // Fallback for browsers without clipboard API
                const textarea = document.createElement('textarea');
                textarea.value = fullText;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
                alert('Text copied');
            });
        };

        document.getElementById('exportBtn').onclick = () => {
            if (!fullText.trim()) {
                alert('No text to export');
                return;
            }

            // Create filename with date and time
            const now = new Date();
            const filename = 'transcription_' +
                now.getFullYear() +
                String(now.getMonth() + 1).padStart(2, '0') +
                String(now.getDate()).padStart(2, '0') + '_' +
                String(now.getHours()).padStart(2, '0') +
                String(now.getMinutes()).padStart(2, '0') + '.txt';

            // Include summary if available
            let exportContent = '=== TRANSCRIPTION ===\\n\\n' + fullText;
            if (currentSummary && currentSummary.trim()) {
                exportContent += '\\n\\n=== SUMMARY ===\\n\\n' + currentSummary;
            }

            // Create blob and download
            const blob = new Blob([exportContent], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        };

        // ===== Gemma 3n Summary Generation =====
        let llmInference = null;
        let isModelLoading = false;
        let currentSummary = '';
        const MODEL_URL = 'models/gemma-3n-E2B-it-int4-Web.litertlm';
        const summaryBtn = document.getElementById('summaryBtn');
        const summaryDiv = document.getElementById('summary');
        const modelStatus = document.getElementById('modelStatus');

        async function initializeLLM() {
            if (llmInference || isModelLoading) return;

            isModelLoading = true;
            modelStatus.textContent = 'Loading Gemma 3n model (first time may take a while)...';
            summaryBtn.disabled = true;

            try {
                const genai = await FilesetResolver.forGenAiTasks(
                    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm'
                );

                llmInference = await LlmInference.createFromOptions(genai, {
                    baseOptions: { modelAssetPath: MODEL_URL },
                    maxTokens: 4096,
                    topK: 40,
                    temperature: 0.7,
                    randomSeed: 1
                });
                modelStatus.textContent = 'Model loaded successfully';
                summaryBtn.disabled = false;
                isModelLoading = false;
            } catch (e) {
                console.error('Error loading model:', e);
                modelStatus.textContent = 'Error: ' + e.message;
                isModelLoading = false;
            }
        }

        async function generateSummary() {
            if (!fullText.trim()) {
                alert('No transcription to summarize');
                return;
            }

            if (!llmInference) {
                await initializeLLM();
                if (!llmInference) return;
            }

            summaryBtn.disabled = true;
            summaryBtn.textContent = 'Generating...';
            summaryDiv.classList.remove('empty');
            summaryDiv.textContent = '';
            currentSummary = '';

            try {
                // Truncate text if too long (roughly 3 chars per token, leave room for prompt and output)
                const maxChars = 6000;
                let textToSummarize = fullText;
                if (fullText.length > maxChars) {
                    textToSummarize = fullText.substring(0, maxChars) + '... [truncated]';
                    console.log('Transcription truncated from', fullText.length, 'to', maxChars, 'chars');
                }

                const prompt = `You are a meeting assistant. Summarize the following transcript using Markdown formatting.

Your response MUST use this exact format:

## Main Topics
- Topic 1
- Topic 2

## Key Decisions
- Decision 1
- Decision 2

## Action Items
- [ ] Action item 1
- [ ] Action item 2

## Summary
Brief summary paragraph here.

---

Transcript to summarize:
${textToSummarize}

Markdown Summary:`;

                // Use streaming for real-time output
                llmInference.generateResponse(prompt, (partialResult, done) => {
                    currentSummary += partialResult;
                    // Render Markdown in real-time
                    if (typeof marked !== 'undefined') {
                        summaryDiv.innerHTML = marked.parse(currentSummary);
                    } else {
                        summaryDiv.textContent = currentSummary;
                    }
                    summaryDiv.scrollTop = summaryDiv.scrollHeight;

                    if (done) {
                        summaryBtn.disabled = false;
                        summaryBtn.textContent = 'Generate Summary';
                    }
                });

            } catch (e) {
                console.error('Error generating summary:', e);
                summaryDiv.textContent = 'Error generating summary: ' + e.message;
                summaryBtn.disabled = false;
                summaryBtn.textContent = 'Generate Summary';
            }
        }

        summaryBtn.onclick = generateSummary;

        // Enable summary button when there's text
        function updateSummaryButton() {
            if (fullText.trim() && !isModelLoading) {
                summaryBtn.disabled = false;
            }
        }

        // Initialize
        loadDevices();
        setInterval(checkStatus, 2000);
        checkStatus();

        // Check if WebGPU is available
        if (!navigator.gpu) {
            modelStatus.textContent = 'WebGPU not available. Summary generation requires Chrome/Edge with WebGPU support.';
        } else {
            modelStatus.textContent = 'WebGPU available. Click "Generate Summary" to load the model.';
        }
    </script>
</body>
</html>
'''


class TranscriberHandler(BaseHTTPRequestHandler):
    """Manejador HTTP para la API"""

    def log_message(self, format, *args):
        # Solo loguear errores, no cada request
        pass

    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))

        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()

            status = {
                "ready": is_ready,
                "status": loading_status,
                "model": MODEL_NAME,
                "resemblyzer": voice_encoder is not None,
                "speakers": len(speaker_embeddings)
            }
            self.wfile.write(json.dumps(status).encode('utf-8'))

        elif self.path.startswith('/models/'):
            # Serve model files with streaming for large files
            file_path = os.path.join(os.path.dirname(__file__), self.path[1:])
            logger.info(f"Requesting model file: {file_path}")
            if os.path.exists(file_path) and os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"Serving model file: {file_size} bytes")
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', file_size)
                self.send_header('Cache-Control', 'public, max-age=31536000')
                self.send_cors_headers()
                self.end_headers()
                # Stream in chunks for large files
                with open(file_path, 'rb') as f:
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            else:
                logger.error(f"Model file not found: {file_path}")
                self.send_response(404)
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(b'Model file not found')

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        if self.path == '/transcribe':
            try:
                data = json.loads(body)
                audio_base64 = data.get('audio', '')
                language = data.get('language', 'es')

                result = transcribe_audio(audio_base64, language)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif self.path == '/reset':
            reset_speakers()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()


def main():
    global PORT

    print("=" * 50)
    print("Whisper Transcriber - Iniciando...")
    print("=" * 50)
    print()

    # Cargar modelos en segundo plano
    model_thread = threading.Thread(target=load_models)
    model_thread.daemon = True
    model_thread.start()

    # Iniciar servidor HTTP
    try:
        server = HTTPServer(('127.0.0.1', PORT), TranscriberHandler)
    except OSError:
        # Puerto en uso, intentar otro
        PORT = 8766
        try:
            server = HTTPServer(('127.0.0.1', PORT), TranscriberHandler)
        except OSError:
            PORT = 0  # Dejar que el sistema asigne uno
            server = HTTPServer(('127.0.0.1', PORT), TranscriberHandler)
            PORT = server.server_address[1]

    url = f'http://127.0.0.1:{PORT}'

    print(f"Servidor iniciado en: {url}")
    print()
    print("Abriendo navegador...")
    print()
    print("Presiona Ctrl+C para detener el servidor")
    print("=" * 50)

    # Abrir navegador
    webbrowser.open(url)

    # Ejecutar servidor
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDeteniendo servidor...")
        server.shutdown()


if __name__ == '__main__':
    main()
