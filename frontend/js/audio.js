export const sampleDialogueChunks = [
  "Doctor: Hello Rahul, good to see you. What brings you in today?",
  "Patient: Hi Doctor. I have had fever for three days with headache and body aches. The fever is worse at night.",
  "Doctor: Any cough, chest pain, or shortness of breath?",
  "Patient: No, I deny cough, chest pain, and shortness of breath. No significant past medical history.",
  "Doctor: Are you taking any medications or have any allergies?",
  "Patient: No known drug allergies. I take paracetamol occasionally.",
  "Doctor: Your temperature is 101.2 Fahrenheit. Provisional diagnosis is viral fever. I advise CBC, increased fluids, and rest. Follow up in three days if symptoms persist."
];

function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let index = 0; index < string.length; index += 1) {
      view.setUint8(offset + index, string.charCodeAt(index));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let index = 0; index < samples.length; index += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, Number(samples[index]) || 0));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
  }

  return new Blob([view], { type: "audio/wav" });
}

export class AudioTranscriptionManager {
  constructor({
    waveformCanvas,
    apiBase,
    language,
    onTranscriptChunk,
    onLanguageDetected,
    onInterimSpeech,
    onStatusChange,
    onStateChange,
    onToast,
    onRecordingComplete
  }) {
    this.canvas = waveformCanvas;
    this.canvasCtx = waveformCanvas ? waveformCanvas.getContext("2d") : null;
    this.apiBase = apiBase || (window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8000");
    this.language = language || "auto";
    this.onTranscriptChunk = onTranscriptChunk || (() => {});
    this.onLanguageDetected = onLanguageDetected || (() => {});
    this.onInterimSpeech = onInterimSpeech || (() => {});
    this.onStatusChange = onStatusChange || (() => {});
    this.onStateChange = onStateChange || (() => {});
    this.onToast = onToast || (() => {});
    this.onRecordingComplete = onRecordingComplete || (() => {});

    this.isRecording = false;
    this.isStarting = false;
    this.isStopping = false;
    this.isSimulated = false;
    this.simulationIndex = 0;
    this.simulationTimer = null;

    this.stream = null;
    this.audioContext = null;
    this.analyser = null;
    this.source = null;
    this.processor = null;
    this.muteGain = null;

    this.sampleBuffer = [];
    this.chunkIntervalId = null;
    this.targetSampleRate = 16000;
    this.chunkDurationMs = 7000;
    this.transcriptContext = "";
    this.animationFrameId = null;
    this.inFlightUploadPromise = null;
    this.uploadChain = Promise.resolve();
    this.pendingUploads = new Set();
    this.stopPromise = null;
    this.canvasSize = { width: 0, height: 0 };

    this.initCanvas();
  }

  setLanguage(language) {
    this.language = language || "auto";
    this.transcriptContext = "";
  }

  async toggleSimulationMode() {
    if (this.isStarting) return this.isSimulated;
    if (this.isRecording || this.isStopping) {
      await this.stopRecording();
    }
    this.isSimulated = !this.isSimulated;
    return this.isSimulated;
  }

  initCanvas() {
    if (!this.canvas || !this.canvasCtx) return;
    const resizeCanvas = () => {
      const parent = this.canvas.parentElement;
      const rect = this.canvas.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width || parent?.clientWidth || 300));
      const height = Math.max(1, Math.floor(rect.height || parent?.clientHeight || 80));
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      this.canvasSize = { width, height };
      this.canvas.width = Math.floor(width * ratio);
      this.canvas.height = Math.floor(height * ratio);
      this.canvasCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
      if (!this.isRecording) {
        this.drawIdleWaveform();
      }
    };
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
  }

  drawIdleWaveform() {
    if (!this.canvasCtx) return;
    const { width, height } = this.canvasSize;
    this.canvasCtx.clearRect(0, 0, width, height);
    this.canvasCtx.beginPath();
    this.canvasCtx.strokeStyle = "#e2e8f0";
    this.canvasCtx.lineWidth = 1.5;
    this.canvasCtx.moveTo(0, height / 2);
    this.canvasCtx.lineTo(width, height / 2);
    this.canvasCtx.stroke();
  }

  drawActiveWaveform(dataArray) {
    if (!this.canvasCtx || !dataArray.length) return;
    const { width, height } = this.canvasSize;
    this.canvasCtx.clearRect(0, 0, width, height);
    this.canvasCtx.lineWidth = 2;
    this.canvasCtx.strokeStyle = "#6366f1";
    this.canvasCtx.beginPath();

    const sliceWidth = width / dataArray.length;
    let x = 0;
    for (let index = 0; index < dataArray.length; index += 1) {
      const value = dataArray[index] / 128 - 1;
      const y = height / 2 + value * (height * 0.42);
      if (index === 0) {
        this.canvasCtx.moveTo(x, y);
      } else {
        this.canvasCtx.lineTo(x, y);
      }
      x += sliceWidth;
    }
    this.canvasCtx.lineTo(width, height / 2);
    this.canvasCtx.stroke();
  }

  async startRecording() {
    if (this.isRecording || this.isStarting || this.isStopping) return false;
    this.transcriptContext = "";

    if (this.isSimulated) {
      this.startSimulatedRecording();
      return true;
    }

    this.isStarting = true;
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Microphone capture is not supported by this browser");
      }
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: true
        },
        video: false
      });

      const AudioContextConstructor = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextConstructor) {
        throw new Error("Web Audio is not supported by this browser");
      }
      this.audioContext = new AudioContextConstructor({ sampleRate: this.targetSampleRate });
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

      const inputSampleRate = this.audioContext.sampleRate;
      this.source = this.audioContext.createMediaStreamSource(this.stream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.muteGain = this.audioContext.createGain();
      this.muteGain.gain.value = 0;
      this.source.connect(this.analyser);
      this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
      this.processor.onaudioprocess = (event) => {
        if (!this.isRecording) return;
        const inputData = event.inputBuffer.getChannelData(0);
        if (inputSampleRate !== this.targetSampleRate) {
          const step = inputSampleRate / this.targetSampleRate;
          for (let sourceIndex = 0; sourceIndex < inputData.length; sourceIndex += step) {
            this.sampleBuffer.push(inputData[Math.floor(sourceIndex)]);
          }
          return;
        }
        for (let index = 0; index < inputData.length; index += 1) {
          this.sampleBuffer.push(inputData[index]);
        }
      };
      this.source.connect(this.processor);
      this.processor.connect(this.muteGain);
      this.muteGain.connect(this.audioContext.destination);

      this.sampleBuffer = [];
      this.isRecording = true;
      this.onStateChange(true);
      this.onStatusChange(`Live mic active — capturing ${this.chunkDurationMs / 1000}-second chunks`);

      const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      const renderWave = () => {
        if (!this.isRecording || !this.analyser) return;
        this.analyser.getByteTimeDomainData(dataArray);
        this.drawActiveWaveform(dataArray);
        this.animationFrameId = requestAnimationFrame(renderWave);
      };
      renderWave();

      this.chunkIntervalId = setInterval(() => {
        void this.flushAudioChunk();
      }, this.chunkDurationMs);
      return true;
    } catch (error) {
      console.error("Microphone access error:", error);
      await this.cleanupAudioResources();
      this.isRecording = false;
      this.onStateChange(false);
      this.onToast("Microphone access denied or unavailable");
      this.onStatusChange("Microphone access error");
      return false;
    } finally {
      this.isStarting = false;
    }
  }

  startSimulatedRecording() {
    this.isRecording = true;
    this.simulationIndex = 0;
    this.onStateChange(true);
    this.onStatusChange("Simulated speech mode active");

    const fakeData = new Uint8Array(128);
    let phase = 0;
    const renderSimWave = () => {
      if (!this.isRecording) return;
      phase += 0.08;
      for (let index = 0; index < fakeData.length; index += 1) {
        fakeData[index] = 128 + Math.sin(phase + index * 0.15) * 35;
      }
      this.drawActiveWaveform(fakeData);
      this.animationFrameId = requestAnimationFrame(renderSimWave);
    };
    renderSimWave();

    this.simulationTimer = setInterval(() => {
      if (this.simulationIndex >= sampleDialogueChunks.length) {
        void this.stopRecording();
        return;
      }
      this.onTranscriptChunk(sampleDialogueChunks[this.simulationIndex]);
      this.simulationIndex += 1;
    }, 3500);
  }

  async flushAudioChunk(isFinal = false) {
    const minimumSamples = isFinal ? 1 : Math.floor(this.targetSampleRate * 0.6);
    if (this.sampleBuffer.length < minimumSamples) return;

    const chunkSamples = new Float32Array(this.sampleBuffer);
    this.sampleBuffer = [];

    try {
      const wavBlob = encodeWAV(chunkSamples, this.targetSampleRate);
      const uploadPromise = this.uploadChain.then(() => this.uploadAudioChunk(wavBlob));
      this.uploadChain = uploadPromise.catch(() => undefined);
      this.pendingUploads.add(uploadPromise);
      this.inFlightUploadPromise = uploadPromise;
      try {
        await uploadPromise;
      } finally {
        this.pendingUploads.delete(uploadPromise);
        if (this.inFlightUploadPromise === uploadPromise) {
          this.inFlightUploadPromise = null;
        }
      }
    } catch (error) {
      console.error("Audio chunk buffering failed:", error);
      this.onToast("Audio chunk could not be prepared");
    }
  }

  async uploadAudioChunk(wavBlob) {
    this.onInterimSpeech("Transcribing speech chunk...");
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000);

    try {
      const formData = new FormData();
      formData.append("file", wavBlob, "chunk.wav");
      formData.append("language", this.language);
      if (this.transcriptContext) formData.append("context", this.transcriptContext);

      const response = await fetch(`${this.apiBase}/api/transcribe/`, {
        method: "POST",
        body: formData,
        signal: controller.signal
      });
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const payload = await response.json();
          if (typeof payload?.detail === "string") detail = payload.detail;
        } catch (error) {
          void error;
        }
        throw new Error(detail);
      }

      const data = await response.json();
      const responseText = typeof data.text === "string" ? data.text.trim() : "";
      if (data.language && data.language !== "auto") {
        this.onLanguageDetected(data.language);
      }
      if (responseText) {
        this.transcriptContext = `${this.transcriptContext} ${responseText}`.trim().slice(-1200);
        this.onTranscriptChunk(responseText);
      }
      this.onInterimSpeech("");
      return true;
    } catch (error) {
      console.error("Transcription chunk failed:", error);
      this.onInterimSpeech("");
      const message = error?.name === "AbortError" ? "timed out" : "failed";
      this.onToast(`Audio chunk transcription ${message}`);
      return false;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async waitForUploads() {
    while (this.pendingUploads.size > 0) {
      await Promise.allSettled([...this.pendingUploads]);
    }
  }

  async cleanupAudioResources() {
    if (this.processor) {
      this.processor.onaudioprocess = null;
      try {
        this.processor.disconnect();
      } catch (error) {
        void error;
      }
      this.processor = null;
    }
    for (const node of [this.source, this.muteGain]) {
      if (node) {
        try {
          node.disconnect();
        } catch (error) {
          void error;
        }
      }
    }
    this.source = null;
    this.muteGain = null;
    this.analyser = null;
    if (this.audioContext && this.audioContext.state !== "closed") {
      try {
        await this.audioContext.close();
      } catch (error) {
        void error;
      }
    }
    this.audioContext = null;
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
  }

  async stopRecording() {
    if (this.stopPromise) return this.stopPromise;
    if (!this.isRecording) return;

    this.stopPromise = (async () => {
      this.isStopping = true;
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

      await this.waitForUploads();
      if (!this.isSimulated) {
        await this.flushAudioChunk(true);
        await this.waitForUploads();
        await this.cleanupAudioResources();
      }

      this.onStateChange(false);
      this.onStatusChange("VAD idle — start mic to capture audio");
      this.drawIdleWaveform();
      this.onRecordingComplete();
    })();

    try {
      await this.stopPromise;
    } finally {
      this.isStopping = false;
      this.stopPromise = null;
    }
  }
}
