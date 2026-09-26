/**
 * UI Interactions, Medical Extraction & Clinical Note Generation
 */

import { AudioTranscriptionManager, sampleDialogueChunks } from "./audio.js";

// DOM Element References
const btnStartMic = document.getElementById('btnStartMic');
const btnStopMic = document.getElementById('btnStopMic');
const btnClearTranscript = document.getElementById('btnClearTranscript');
const btnSaveTranscript = document.getElementById('btnSaveTranscript');
const btnProcessAI = document.getElementById('btnProcessAI');
const btnClearSummary = document.getElementById('btnClearSummary');
const btnCopyToPrescription = document.getElementById('btnCopyToPrescription');
const btnLoadDemo = document.getElementById('btnLoadDemo');
const btnToggleSimulateMic = document.getElementById('btnToggleSimulateMic');
const micModeLabel = document.getElementById('micModeLabel');
const consentDataProcessing = document.getElementById('consentDataProcessing');
const languageSelect = document.getElementById('languageSelect');
const detectedLanguageLabel = document.getElementById('detectedLanguageLabel');

const recordingBadge = document.getElementById('recordingBadge');
const wordCountEl = document.getElementById('wordCount');
const chunkCountEl = document.getElementById('chunkCount');
const transcriptContent = document.getElementById('transcriptContent');
const interimTranscript = document.getElementById('interimTranscript');
const vadStatusOverlay = document.getElementById('vadStatusOverlay');
const waveformCanvas = document.getElementById('waveformCanvas');

// 11 Clinical UI Section DOM elements
const sectionClinicalSummary = document.getElementById('sectionClinicalSummary');
const sectionChiefComplaints = document.getElementById('sectionChiefComplaints');
const sectionHPI = document.getElementById('sectionHPI');
const sectionSymptoms = document.getElementById('sectionSymptoms');
const sectionClinicalImpression = document.getElementById('sectionClinicalImpression');
const sectionMedications = document.getElementById('sectionMedications');
const sectionExaminations = document.getElementById('sectionExaminations');
const sectionAllergies = document.getElementById('sectionAllergies');
const sectionPastHistory = document.getElementById('sectionPastHistory');
const sectionFollowUp = document.getElementById('sectionFollowUp');

// Patient info header & modal elements
const patientNameDisplay = document.getElementById('patientNameDisplay');
const patientMetaDisplay = document.getElementById('patientMetaDisplay');
const btnEditPatient = document.getElementById('btnEditPatient');
const patientModal = document.getElementById('patientModal');
const btnCancelPatient = document.getElementById('btnCancelPatient');
const btnSavePatient = document.getElementById('btnSavePatient');
const inputPatientName = document.getElementById('inputPatientName');
const inputPatientAge = document.getElementById('inputPatientAge');
const inputPatientGender = document.getElementById('inputPatientGender');

// Toast element
const toast = document.getElementById('toast');
const toastMsg = document.getElementById('toastMsg');

// API Base Resolution
const apiBase = window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8000";

// State Variables
let audioChunksCount = 0;
let fullTranscriptText = "";
let analysisPromise = null;
let analysisAbortController = null;
let analysisTranscript = "";
let toastTimer = null;
let isRecording = false;
let transcriptLanguage = "auto";

function requireDataConsent() {
  if (!consentDataProcessing?.checked) {
    showToast("Confirm AI data processing consent before starting or analyzing audio");
    return false;
  }
  return true;
}

// Initialize Audio & Speech Manager
const audioManager = new AudioTranscriptionManager({
  waveformCanvas,
  apiBase,
  language: languageSelect?.value || "auto",
  onTranscriptChunk: (chunk) => appendTranscriptChunk(chunk),
  onLanguageDetected: (language) => {
    if (!language || language === "auto") return;
    transcriptLanguage = language;
    if (detectedLanguageLabel) detectedLanguageLabel.textContent = `Detected: ${language.toUpperCase()}`;
    document.documentElement.lang = language;
  },
  onInterimSpeech: (text) => {
    if (interimTranscript) interimTranscript.textContent = text;
  },
  onStatusChange: (status) => {
    if (vadStatusOverlay) vadStatusOverlay.textContent = status;
  },
  onStateChange: (recording) => {
    isRecording = recording;
    if (recording) {
      transcriptLanguage = languageSelect?.value || "auto";
      if (detectedLanguageLabel) detectedLanguageLabel.textContent = "Detected: —";
      document.documentElement.lang = transcriptLanguage === "auto" ? "en" : transcriptLanguage;
    }
    btnStartMic.disabled = recording;
    btnStartMic.classList.toggle('opacity-60', recording);
    btnStartMic.classList.toggle('cursor-not-allowed', recording);
    btnStopMic.disabled = !recording;
    btnStopMic.classList.toggle('opacity-60', !recording);
    btnStopMic.classList.toggle('cursor-not-allowed', !recording);
    btnClearTranscript.disabled = recording;
    btnSaveTranscript.disabled = recording;
    btnLoadDemo.disabled = recording;
    btnProcessAI.disabled = recording || Boolean(analysisPromise);
    btnToggleSimulateMic.disabled = recording;
    if (languageSelect) languageSelect.disabled = recording;
    recordingBadge.classList.toggle('hidden', !recording);
    recordingBadge.classList.toggle('inline-flex', recording);
  },
  onToast: (msg) => showToast(msg),
  onRecordingComplete: () => {
    if (fullTranscriptText.trim().length > 0) {
      showToast("Recording stopped. Analyzing consultation transcript...");
      void processTranscriptWithAI();
    }
  }
});

function cancelAnalysis() {
  const wasRunning = Boolean(analysisPromise);
  if (analysisAbortController) {
    analysisAbortController.abort();
    analysisAbortController = null;
  }
  analysisPromise = null;
  analysisTranscript = "";
  if (wasRunning) {
    btnProcessAI.disabled = isRecording;
    btnProcessAI.classList.remove('opacity-75');
    document.getElementById('aiButtonText').textContent = "Process Transcript with AI";
    document.getElementById('aiButtonIcon').classList.remove('animate-spin');
    document.getElementById('aiButtonIcon').innerHTML = `<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />`;
  }
}

// Transcript management
function appendTranscriptChunk(chunkText) {
  if (!chunkText) return;
  const placeholderText = document.getElementById('placeholderText');
  if (placeholderText) placeholderText.remove();

  audioChunksCount++;
  chunkCountEl.textContent = audioChunksCount;

  fullTranscriptText += (fullTranscriptText ? "\n\n" : "") + chunkText;
  updateTranscriptDisplay();
}

function updateTranscriptDisplay() {
  const words = fullTranscriptText.trim().length > 0 ? fullTranscriptText.trim().split(/\s+/).length : 0;
  wordCountEl.textContent = words;
  transcriptContent.textContent = fullTranscriptText;
  transcriptContent.scrollTop = transcriptContent.scrollHeight;
}

function clearTranscript() {
  if (isRecording) {
    showToast("Stop recording before clearing the transcript");
    return;
  }
  cancelAnalysis();
  fullTranscriptText = "";
  audioChunksCount = 0;
  wordCountEl.textContent = "0";
  chunkCountEl.textContent = "0";
  transcriptContent.innerHTML = `
    <span id="placeholderText" class="text-slate-400 flex items-start gap-2 select-none">
      <svg class="w-4 h-4 text-slate-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
      </svg>
      <span>Ready &mdash; click Start Mic and speak naturally. VAD will auto-chunk your speech.</span>
    </span>
  `;
  if (interimTranscript) interimTranscript.textContent = "";
  showToast("Transcript reset");
}

async function processTranscriptWithAI() {
  if (!requireDataConsent()) return;
  const transcript = fullTranscriptText.trim();
  if (!transcript) {
    showToast("Please record speech or load sample dialogue before processing!");
    return;
  }
  if (isRecording) {
    showToast("Stop recording before processing the transcript");
    return;
  }
  if (analysisPromise) {
    showToast("Clinical analysis is already in progress");
    return;
  }

  const aiBtn = btnProcessAI;
  const aiBtnText = document.getElementById('aiButtonText');
  const aiBtnIcon = document.getElementById('aiButtonIcon');
  const controller = new AbortController();
  analysisAbortController = controller;
  analysisTranscript = transcript;
  analysisPromise = (async () => {
    aiBtn.disabled = true;
    aiBtn.classList.add('opacity-75');
    aiBtnText.textContent = "Extracting Clinical Entities...";
    aiBtnIcon.innerHTML = `<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>`;
    aiBtnIcon.classList.add('animate-spin');

    try {
      const response = await fetch(`${apiBase}/api/analyze/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          transcript,
          language: transcriptLanguage === "auto" ? null : transcriptLanguage
        }),
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

      const note = await response.json();
      if (analysisTranscript === transcript && fullTranscriptText.trim() === transcript) {
        populateMedicalSummary(note);
        showToast("Clinical summary extracted successfully");
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        console.error("Clinical analysis error:", error);
        showToast(`Analysis failed: ${error.message}`);
      }
    } finally {
      if (analysisAbortController === controller) {
        analysisAbortController = null;
        analysisPromise = null;
        analysisTranscript = "";
        aiBtn.disabled = isRecording;
        aiBtn.classList.remove('opacity-75');
        aiBtnText.textContent = "Process Transcript with AI";
        aiBtnIcon.classList.remove('animate-spin');
        aiBtnIcon.innerHTML = `<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />`;
      }
    }
  })();

  await analysisPromise;
}

// Helper to render lists as visual badges
function renderBadgeList(container, items, colorClass = "bg-slate-50 text-slate-700 border-slate-200", emptyText = "None documented") {
  if (!container) return;
  const values = Array.isArray(items) ? items.filter(Boolean) : [];
  if (values.length === 0) {
    container.className = "text-center py-2 text-xs text-slate-400 min-h-[30px] flex items-center justify-center";
    container.textContent = emptyText;
    return;
  }
  container.className = "flex flex-wrap gap-1.5 py-1";
  container.innerHTML = values
    .map(
      (item) => `
      <span class="inline-flex items-center text-xs px-2.5 py-1 rounded-md border font-medium ${colorClass}">
        ${escapeHtml(item)}
      </span>
    `
    )
    .join('');
}

function escapeHtml(str) {
  if (typeof str !== 'string') return String(str || '');
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Populate the 11 UI sections derived from required clinical information
function populateMedicalSummary(data) {
  if (!data) return;

  // 1. Patient Demographics Header
  if (data.patient_details) {
    const p = data.patient_details;
    const name = typeof p.name === "string" && p.name.trim() ? p.name.trim() : "Not specified";
    patientNameDisplay.textContent = `Patient Name: ${name}`;
    inputPatientName.value = name === "Not specified" ? "" : name;
    const ageText = p.age !== null && p.age !== undefined ? String(p.age) : "N/A";
    const sexText = typeof p.sex === "string" && p.sex.trim() ? p.sex.trim() : "N/A";
    patientMetaDisplay.textContent = `Age: ${ageText}  Gender: ${sexText}`;
    inputPatientAge.value = p.age !== null && p.age !== undefined ? String(p.age) : "";
    inputPatientGender.value = ["Male", "Female", "Other"].includes(sexText) ? sexText : "";
  }

  // 2. Structured Clinical Summary
  sectionClinicalSummary.value = data.clinical_summary || "";

  // 3. Chief Complaint
  if (data.chief_complaint) {
    sectionChiefComplaints.className = "py-1 text-xs sm:text-sm text-slate-800 font-medium";
    sectionChiefComplaints.innerHTML = `
      <span class="inline-flex items-center text-xs px-2.5 py-1 rounded-md border font-medium bg-amber-50 text-amber-900 border-amber-200">
        ${escapeHtml(data.chief_complaint)}
      </span>
    `;
  } else {
    sectionChiefComplaints.className = "text-center py-2 text-xs text-slate-400 min-h-[30px] flex items-center justify-center";
    sectionChiefComplaints.textContent = "No chief complaint recorded";
  }

  // 4. History of Present Illness (HPI)
  if (sectionHPI) {
    if (data.history_of_present_illness) {
      sectionHPI.className = "py-1 text-xs sm:text-sm text-slate-700 leading-relaxed";
      sectionHPI.textContent = data.history_of_present_illness;
    } else {
      sectionHPI.className = "text-center py-2 text-xs text-slate-400 min-h-[30px] flex items-center justify-center";
      sectionHPI.textContent = "No HPI narrative extracted";
    }
  }

  // 5. Symptoms (Positive and Negative / Denied)
  const posSymptoms = Array.isArray(data.symptoms?.positive) ? data.symptoms.positive : [];
  const negSymptoms = Array.isArray(data.symptoms?.negative) ? data.symptoms.negative : [];
  if (posSymptoms.length === 0 && negSymptoms.length === 0) {
    sectionSymptoms.className = "text-center py-2 text-xs text-slate-400 min-h-[30px] flex items-center justify-center";
    sectionSymptoms.textContent = "No symptoms recorded";
  } else {
    sectionSymptoms.className = "flex flex-wrap gap-1.5 py-1";
    let html = "";
    posSymptoms.forEach((s) => {
      html += `
        <span class="inline-flex items-center text-xs px-2.5 py-1 rounded-md border font-medium bg-rose-50 text-rose-800 border-rose-200">
          ${escapeHtml(s)}
        </span>
      `;
    });
    negSymptoms.forEach((s) => {
      html += `
        <span class="inline-flex items-center text-xs px-2.5 py-1 rounded-md border font-medium bg-slate-100 text-slate-600 border-slate-300">
          Denies: ${escapeHtml(s)}
        </span>
      `;
    });
    sectionSymptoms.innerHTML = html;
  }

  // 6. Clinical Impression / Assessment
  renderBadgeList(
    sectionClinicalImpression,
    data.assessment,
    "bg-indigo-50 text-indigo-800 border-indigo-200 font-semibold",
    "No diagnosis or impression inferred"
  );

  // 7. Medications with Dosage & Adherence
  const meds = Array.isArray(data.medication_history) ? data.medication_history : [];
  if (meds.length === 0) {
    sectionMedications.className = "text-center py-2 text-xs text-slate-400 min-h-[30px] flex items-center justify-center";
    sectionMedications.textContent = "No medications reported";
  } else {
    sectionMedications.className = "space-y-1.5 py-1 text-left text-xs text-slate-700";
    sectionMedications.innerHTML = meds
      .map((m) => {
        const details = [m.dosage, m.adherence].filter(Boolean).join(" &bull; ");
        return `
          <div class="flex items-start gap-2 p-2 rounded-lg bg-emerald-50/70 border border-emerald-200 text-emerald-900 font-medium">
            <svg class="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
            </svg>
            <div>
              <div class="font-bold text-emerald-950">${escapeHtml(m.name)}</div>
              ${details ? `<div class="text-[11px] text-emerald-700 font-normal mt-0.5">${escapeHtml(details)}</div>` : ""}
            </div>
          </div>
        `;
      })
      .join('');
  }

  // 8. Clinical Observations (Vitals & Physical Exam)
  renderBadgeList(
    sectionExaminations,
    data.clinical_observations,
    "bg-sky-50 text-sky-800 border-sky-200",
    "No vitals or exam observations extracted"
  );

  // 9. Allergies
  renderBadgeList(
    sectionAllergies,
    data.allergies,
    "bg-purple-50 text-purple-800 border-purple-200",
    "No known allergies reported"
  );

  // 10. Past Medical History
  renderBadgeList(
    sectionPastHistory,
    data.past_medical_history,
    "bg-slate-50 text-slate-700 border-slate-200",
    "No past medical history recorded"
  );

  // 11. Plan & Follow-up
  renderBadgeList(
    sectionFollowUp,
    data.plan,
    "bg-blue-50 text-blue-800 border-blue-200",
    "No plan or follow-up noted"
  );
}

function clearMedicalSummary() {
  cancelAnalysis();
  sectionClinicalSummary.value = "";
  patientNameDisplay.textContent = "Patient Name: Not specified";
  patientMetaDisplay.textContent = "Age: N/A  Gender: N/A";
  inputPatientName.value = "";
  inputPatientAge.value = "";
  inputPatientGender.value = "";

  const resetPlaceholders = [
    { el: sectionChiefComplaints, text: "No complaints extracted yet" },
    { el: sectionHPI, text: "No HPI narrative extracted yet" },
    { el: sectionSymptoms, text: "No symptoms extracted yet" },
    { el: sectionClinicalImpression, text: "No diagnosis inferred" },
    { el: sectionMedications, text: "No medications extracted yet" },
    { el: sectionExaminations, text: "No examinations extracted yet" },
    { el: sectionAllergies, text: "No allergies extracted yet" },
    { el: sectionPastHistory, text: "No past medical history extracted" },
    { el: sectionFollowUp, text: "No follow-up notes extracted" }
  ];

  resetPlaceholders.forEach(({ el, text }) => {
    if (el) {
      el.className = "text-center py-2 text-xs text-slate-400 min-h-[30px] flex items-center justify-center";
      el.innerHTML = text;
    }
  });

  showToast("Summary fields cleared");
}

async function copyToPrescription() {
  const patientName = inputPatientName.value.trim() || "Not specified";
  const patientAge = inputPatientAge.value.trim() || "Not specified";
  const patientGender = inputPatientGender.value || "Not specified";
  const summaryVal = sectionClinicalSummary.value.trim() || "Not documented";

  function extractText(node) {
    return node && node.innerText.trim() ? node.innerText.trim() : "Not documented";
  }

  const prescriptionText = `================ CLINICAL PRESCRIPTION & ENCOUNTER NOTE ================
*** AI-GENERATED DRAFT — CLINICIAN REVIEW REQUIRED ***
Patient: ${patientName} | Age: ${patientAge} | Gender: ${patientGender}
Date: ${new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })}
Provider: Clinical AI Scribe System
------------------------------------------------------------------------
[CLINICAL SUMMARY]
${summaryVal}

[CHIEF COMPLAINTS]
${extractText(sectionChiefComplaints)}

[HISTORY OF PRESENT ILLNESS (HPI)]
${extractText(sectionHPI)}

[SYMPTOMS]
${extractText(sectionSymptoms)}

[CLINICAL IMPRESSION / ASSESSMENT]
${extractText(sectionClinicalImpression)}

[MEDICATIONS & RX]
${extractText(sectionMedications)}

[CLINICAL OBSERVATIONS & VITALS]
${extractText(sectionExaminations)}

[ALLERGIES]
${extractText(sectionAllergies)}

[PAST MEDICAL HISTORY]
${extractText(sectionPastHistory)}

[PLAN & FOLLOW-UP]
${extractText(sectionFollowUp)}
========================================================================`;

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(prescriptionText);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = prescriptionText;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      textarea.remove();
      if (!copied) throw new Error("Clipboard unavailable");
    }
    showToast("Encounter copied to prescription clipboard!");
  } catch (error) {
    showToast("Clipboard copy failed");
  }
}

function saveRecordingTranscript() {
  if (!fullTranscriptText.trim()) {
    showToast("No transcript available to save!");
    return;
  }
  const blob = new Blob([fullTranscriptText], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `Medical_Transcription_${new Date().toISOString().slice(0, 10)}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast("Transcript file downloaded (.txt)");
}

function showToast(message) {
  if (!toast || !toastMsg) return;
  if (toastTimer) clearTimeout(toastTimer);
  toastMsg.textContent = message;
  toast.classList.remove('opacity-0', 'pointer-events-none');
  toast.classList.add('opacity-100');
  toastTimer = setTimeout(() => {
    toast.classList.remove('opacity-100');
    toast.classList.add('opacity-0', 'pointer-events-none');
    toastTimer = null;
  }, 2400);
}

// Event Listeners
btnStartMic.addEventListener('click', async () => {
  if (!requireDataConsent()) return;
  if (analysisPromise) cancelAnalysis();
  await audioManager.startRecording();
});
btnStopMic.addEventListener('click', () => {
  void audioManager.stopRecording();
});
btnClearTranscript.addEventListener('click', clearTranscript);
btnSaveTranscript.addEventListener('click', saveRecordingTranscript);
btnProcessAI.addEventListener('click', () => {
  void processTranscriptWithAI();
});
btnClearSummary.addEventListener('click', clearMedicalSummary);
btnCopyToPrescription.addEventListener('click', () => {
  void copyToPrescription();
});

btnToggleSimulateMic.addEventListener('click', async () => {
  const isSimulated = await audioManager.toggleSimulationMode();
  micModeLabel.textContent = isSimulated ? "Simulated Speech" : "Live Mic (16kHz PCM)";
  showToast(isSimulated ? "Switched to simulated voice test" : "Switched to live microphone");
});

btnLoadDemo.addEventListener('click', () => {
  if (isRecording) {
    showToast("Stop recording before loading a sample dialogue");
    return;
  }
  cancelAnalysis();
  const placeholderText = document.getElementById('placeholderText');
  if (placeholderText) placeholderText.remove();

  fullTranscriptText = sampleDialogueChunks.join("\n\n");
  audioChunksCount = sampleDialogueChunks.length;
  chunkCountEl.textContent = audioChunksCount;
  updateTranscriptDisplay();
  showToast("Loaded sample consultation transcript");
});

languageSelect?.addEventListener("change", () => {
  if (isRecording) return;
  audioManager.setLanguage(languageSelect.value);
  transcriptLanguage = languageSelect.value;
  detectedLanguageLabel.textContent = "Detected: —";
  document.documentElement.lang = languageSelect.value === "auto" ? "en" : languageSelect.value;
  showToast(languageSelect.value === "auto" ? "Automatic language detection enabled" : `Language set to ${languageSelect.value.toUpperCase()}`);
});

function closePatientModal() {
  patientModal.classList.add('hidden');
}

btnEditPatient.addEventListener('click', () => {
  patientModal.classList.remove('hidden');
  inputPatientName.focus();
});
btnCancelPatient.addEventListener('click', closePatientModal);
patientModal.addEventListener('click', (event) => {
  if (event.target === patientModal) closePatientModal();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !patientModal.classList.contains('hidden')) {
    closePatientModal();
  }
});

btnSavePatient.addEventListener('click', () => {
  const name = inputPatientName.value.trim() || "Not specified";
  const ageValue = inputPatientAge.value.trim();
  const ageNumber = Number(ageValue);
  if (ageValue && (!Number.isInteger(ageNumber) || ageNumber < 0 || ageNumber > 150)) {
    showToast("Age must be a whole number between 0 and 150");
    return;
  }
  const age = ageValue || "N/A";
  const gender = inputPatientGender.value || "Not specified";

  patientNameDisplay.textContent = `Patient Name: ${name}`;
  patientMetaDisplay.textContent = `Age: ${age}  Gender: ${gender}`;
  closePatientModal();
  showToast("Patient information updated");
});

document.getElementById('floatingChatBtn').addEventListener('click', () => {
  showToast("Clinical Copilot is not connected in this local build");
});
