/**
 * Audio Recording, 16kHz PCM WAV Encoding & Waveform Visualizer
 */

export const sampleDialogueChunks = [
  "Doctor: Hello Rahul, good to see you. What brings you in today?",
  "Patient: Hi Doctor. I have had fever for three days with headache and body aches. The fever is worse at night.",
  "Doctor: Any cough, chest pain, or shortness of breath?",
  "Patient: No, I deny cough, chest pain, and shortness of breath. No significant past medical history.",
  "Doctor: Are you taking any medications or have any allergies?",
  "Patient: No known drug allergies. I take paracetamol occasionally.",
  "Doctor: Your temperature is 101.2 Fahrenheit. Provisional diagnosis is viral fever. I advise CBC, increased fluids, and rest. Follow up in three days if symptoms persist."
];

// In-browser 16-bit PCM WAV Encoder
function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono channel
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // Byte rate
  view.setUint16(32, 2, true); // Block align
  view.setUint16(34, 16, true); // Bits per sample
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  // Write PCM float32 samples to int16
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

export class AudioTranscriptionManager {
  constructor({
    waveformCanvas,
    apiBase,
    onTranscriptChunk,
    onInterimSpeech,
    onStatusChange,
    onStateChange,
    onToast,
    onRecordingComplete
  }) {
    this.canvas = waveformCanvas;
    this.canvasCtx = waveformCanvas ? waveformCanvas.getContext('2d') : null;
    this.apiBase = apiBase || (window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8000");
    this.onTranscriptChunk = onTranscriptChunk || (() => {});
    this.onInterimSpeech = onInterimSpeech || (() => {});
    this.onStatusChange = onStatusChange || (() => {});
    this.onStateChange = onStateChange || (() => {});
    this.onToast = onToast || (() => {});
    this.onRecordingComplete = onRecordingComplete || (() => {});

    this.isRecording = false;
    this.isSimulated = false;
    this.simulationIndex = 0;
    this.simulationTimer = null;

    this.stream = null;
    this.audioContext = null;
    this.analyser = null;
    this.source = null;
    this.processor = null;

    this.sampleBuffer = [];
    this.chunkIntervalId = null;
    this.targetSampleRate = 16000;
    this.animationFrameId = null;
    this.inFlightUploadPromise = null;

    this.initCanvas();
  }

  toggleSimulationMode() {
    this.isSimulated = !this.isSimulated;
    if (this.isRecording) {
      this.stopRecording();
    }
    return this.isSimulated;
  }

  initCanvas() {
    if (!this.canvas) return;
    const resizeCanvas = () => {
      this.canvas.width = this.canvas.parentElement.clientWidth * window.devicePixelRatio;
      this.canvas.height = this.canvas.parentElement.clientHeight * window.devicePixelRatio;
      this.canvasCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
      if (!this.isRecording) {
        this.drawIdleWaveform();
      }
    };
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
  }

  drawIdleWaveform() {
    if (!this.canvas || !this.canvasCtx) return;
    const w = this.canvas.parentElement.clientWidth;
    const h = this.canvas.parentElement.clientHeight;
    this.canvasCtx.clearRect(0, 0, w, h);

    this.canvasCtx.beginPath();
    this.canvasCtx.strokeStyle = '#e2e8f0';
    this.canvasCtx.lineWidth = 1.5;
    this.canvasCtx.moveTo(0, h / 2);
    this.canvasCtx.lineTo(w, h / 2);
    this.canvasCtx.stroke();
  }

  drawActiveWaveform(dataArray) {
    if (!this.canvas || !this.canvasCtx) return;
    const w = this.canvas.parentElement.clientWidth;
    const h = this.canvas.parentElement.clientHeight;
    this.canvasCtx.clearRect(0, 0, w, h);

    this.canvasCtx.lineWidth = 2;
    this.canvasCtx.strokeStyle = '#6366f1';
    this.canvasCtx.beginPath();

    const sliceWidth = w / dataArray.length;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * h) / 2;

      if (i === 0) {
        this.canvasCtx.moveTo(x, y);
      } else {
        this.canvasCtx.lineTo(x, y);
      }
      x += sliceWidth;
    }
    this.canvasCtx.lineTo(w, h / 2);
    this.canvasCtx.stroke();
  }

  async startRecording() {
    if (this.isRecording) return;

    if (this.isSimulated) {
      this.startSimulatedRecording();
      return;
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false, // Let our backend preprocessing handle noise
          autoGainControl: true,
        },
        video: false,
      });

      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
      });
      const inputSampleRate = this.audioContext.sampleRate;

      this.source = this.audioContext.createMediaStreamSource(this.stream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.source.connect(this.analyser);

      // ScriptProcessor to collect PCM audio frames
      const bufferSize = 4096;
      this.processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

      this.sampleBuffer = [];
      this.processor.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        const inputData = e.inputBuffer.getChannelData(0);

        // Resample if AudioContext is not 16kHz
        if (inputSampleRate !== 16000) {
          const ratio = inputSampleRate / 16000;
          for (let i = 0; i < inputData.length; i += ratio) {
            this.sampleBuffer.push(inputData[Math.floor(i)]);
          }
        } else {
          for (let i = 0; i < inputData.length; i++) {
            this.sampleBuffer.push(inputData[i]);
          }
        }
      };

      this.source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);

      this.isRecording = true;
      this.onStateChange(true);
      this.onStatusChange("🎙️ Live Mic Active — Capturing speech chunks (every 3.5s)");

      // Draw real-time waveform
      const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      const renderWave = () => {
        if (!this.isRecording) return;
        this.analyser.getByteTimeDomainData(dataArray);
        this.drawActiveWaveform(dataArray);
        this.animationFrameId = requestAnimationFrame(renderWave);
      };
      renderWave();

      // Send 3.5-second audio chunks to /api/transcribe/
      this.chunkIntervalId = setInterval(() => {
        this.flushAudioChunk();
      }, 3500);

    } catch (err) {
      console.error("Microphone access error:", err);
      this.onToast("Microphone access denied or unavailable");
      this.onStatusChange("Microphone access error");
      this.stopRecording();
    }
  }

  startSimulatedRecording() {
    this.isRecording = true;
    this.simulationIndex = 0;
    this.onStateChange(true);
    this.onStatusChange("🎙️ Simulated Speech Mode Active");

    // Simulated waveform animation
    const fakeData = new Uint8Array(128);
    let phase = 0;
    const renderSimWave = () => {
      if (!this.isRecording) return;
      phase += 0.08;
      for (let i = 0; i < fakeData.length; i++) {
        fakeData[i] = 128 + Math.sin(phase + i * 0.15) * 35;
      }
      this.drawActiveWaveform(fakeData);
      this.animationFrameId = requestAnimationFrame(renderSimWave);
    };
    renderSimWave();

    // Emit dialogue chunks every 3.5 seconds
    this.simulationTimer = setInterval(() => {
      if (this.simulationIndex < sampleDialogueChunks.length) {
        const chunk = sampleDialogueChunks[this.simulationIndex++];
        this.onTranscriptChunk(chunk);
      } else {
        this.stopRecording();
      }
    }, 3500);
  }

  async flushAudioChunk(isFinal = false) {
    // Need at least ~0.6s of audio to process
    if (this.sampleBuffer.length < this.targetSampleRate * 0.6) {
      return;
    }

    const chunkSamples = new Float32Array(this.sampleBuffer);
    this.sampleBuffer = [];

    const wavBlob = encodeWAV(chunkSamples, this.targetSampleRate);
    this.inFlightUploadPromise = this.uploadAudioChunk(wavBlob, isFinal);
    await this.inFlightUploadPromise;
    this.inFlightUploadPromise = null;
  }

  async uploadAudioChunk(wavBlob, isFinal = false) {
    this.onInterimSpeech("Transcribing speech chunk...");

    try {
      const formData = new FormData();
      formData.append("file", wavBlob, "chunk.wav");
      formData.append("language", "en");

      const response = await fetch(`${this.apiBase}/api/transcribe/`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP status ${response.status}`);
      }

      const data = await response.json();
      if (data.text && data.text.trim()) {
        this.onTranscriptChunk(data.text.trim());
      }
      this.onInterimSpeech("");
    } catch (err) {
      console.error("Transcription chunk failed:", err);
      this.onInterimSpeech("");
      // Fail-forward: log/notify without retrying to prevent duplicate chunks
      this.onToast("Audio chunk transcription notice: skipped silent/corrupt chunk");
    }
  }

  async stopRecording() {
    if (!this.isRecording) return;
    this.isRecording = false;

    if (this.chunkIntervalId) {
      clearInterval(this.chunkIntervalId);
      this.chunkIntervalId = null;
    }

    if (this.simulationTimer) {
      clearInterval(this.simulationTimer);
      this.simulationTimer = null;
    }

    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    // Await any in-flight chunk upload, then flush remaining buffered audio
    if (this.inFlightUploadPromise) {
      try {
        await this.inFlightUploadPromise;
      } catch (e) {}
      this.inFlightUploadPromise = null;
    }

    if (!this.isSimulated) {
      await this.flushAudioChunk(true);

      if (this.processor) {
        this.processor.disconnect();
        this.processor = null;
      }
      if (this.source) {
        this.source.disconnect();
        this.source = null;
      }
      if (this.audioContext && this.audioContext.state !== "closed") {
        try {
          await this.audioContext.close();
        } catch (e) {}
      }
      if (this.stream) {
        this.stream.getTracks().forEach((track) => track.stop());
        this.stream = null;
      }
    }

    this.onStateChange(false);
    this.onStatusChange("VAD Idle — Start mic to capture audio");
    this.drawIdleWaveform();

    // Trigger completion handler (which initiates single LLM analysis on full transcript)
    this.onRecordingComplete();
  }
}
