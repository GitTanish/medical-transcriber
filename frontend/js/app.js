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

// Initialize Audio & Speech Manager
const audioManager = new AudioTranscriptionManager({
  waveformCanvas,
  apiBase,
  onTranscriptChunk: (chunk) => appendTranscriptChunk(chunk),
  onInterimSpeech: (text) => {
    if (interimTranscript) interimTranscript.textContent = text;
  },
  onStatusChange: (status) => {
    if (vadStatusOverlay) vadStatusOverlay.textContent = status;
  },
  onStateChange: (isRecording) => {
    if (isRecording) {
      btnStartMic.disabled = true;
      btnStartMic.classList.add('opacity-60', 'cursor-not-allowed');
      btnStopMic.disabled = false;
      btnStopMic.classList.remove('opacity-60', 'cursor-not-allowed');
      recordingBadge.classList.remove('hidden');
      recordingBadge.classList.add('inline-flex');
    } else {
      btnStartMic.disabled = false;
      btnStartMic.classList.remove('opacity-60', 'cursor-not-allowed');
      btnStopMic.disabled = true;
      btnStopMic.classList.add('opacity-60', 'cursor-not-allowed');
      recordingBadge.classList.add('hidden');
      recordingBadge.classList.remove('inline-flex');
    }
  },
  onToast: (msg) => showToast(msg),
  onRecordingComplete: () => {
    // Session completed: Automatically trigger LLM analysis if transcript text exists
    if (fullTranscriptText.trim().length > 0) {
      showToast("Recording stopped. Analyzing consultation transcript...");
      processTranscriptWithAI();
    }
  }
});

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

// Single Clinical Analysis Request on Session Completion
async function processTranscriptWithAI() {
  const transcript = fullTranscriptText.trim();
  if (!transcript) {
    showToast("Please record speech or load sample dialogue before processing!");
    return;
  }

  const aiBtn = document.getElementById('btnProcessAI');
  const aiBtnText = document.getElementById('aiButtonText');
  const aiBtnIcon = document.getElementById('aiButtonIcon');

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
      body: JSON.stringify({ transcript })
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const note = await response.json();
    populateMedicalSummary(note);
    showToast("Clinical summary extracted successfully");

  } catch (err) {
    console.error("Clinical analysis error:", err);
    showToast(`Analysis failed: ${err.message}`);
  } finally {
    aiBtn.disabled = false;
    aiBtn.classList.remove('opacity-75');
    aiBtnText.textContent = "Process Transcript with AI";
    aiBtnIcon.classList.remove('animate-spin');
    aiBtnIcon.innerHTML = `<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />`;
  }
}

// Helper to render lists as visual badges
function renderBadgeList(container, items, colorClass = "bg-slate-50 text-slate-700 border-slate-200", emptyText = "None documented") {
  if (!container) return;
  if (!items || items.length === 0) {
    container.className = "text-center py-2 text-xs text-slate-400 min-h-[30px] flex items-center justify-center";
    container.textContent = emptyText;
    return;
  }
  container.className = "flex flex-wrap gap-1.5 py-1";
  container.innerHTML = items
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
    if (p.name) {
      patientNameDisplay.textContent = `Patient Name: ${p.name}`;
      inputPatientName.value = p.name;
    }
    const ageText = p.age !== null && p.age !== undefined ? p.age : "N/A";
    const sexText = p.sex || "N/A";
    patientMetaDisplay.innerHTML = `Age: ${ageText} &nbsp; Gender: ${sexText}`;
    if (p.age !== null && p.age !== undefined) inputPatientAge.value = p.age;
    if (p.sex) inputPatientGender.value = p.sex;
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
  const posSymptoms = data.symptoms?.positive || [];
  const negSymptoms = data.symptoms?.negative || [];
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
  const meds = data.medication_history || [];
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
  sectionClinicalSummary.value = "";

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

function copyToPrescription() {
  const patientName = inputPatientName.value || "Dr Tushar";
  const patientAge = inputPatientAge.value || "30";
  const patientGender = inputPatientGender.value || "Male";
  const summaryVal = sectionClinicalSummary.value || "N/A";

  function extractText(node) {
    return node ? node.innerText.trim() : "";
  }

  const prescriptionText = `================ CLINICAL PRESCRIPTION & ENCOUNTER NOTE ================
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

  const textarea = document.createElement('textarea');
  textarea.value = prescriptionText;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'absolute';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();

  try {
    document.execCommand('copy');
    showToast("Encounter copied to prescription clipboard!");
  } catch (err) {
    showToast("Clipboard copy failed");
  }
  document.body.removeChild(textarea);
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
  toastMsg.textContent = message;
  toast.classList.remove('opacity-0', 'pointer-events-none');
  toast.classList.add('opacity-100');
  setTimeout(() => {
    toast.classList.remove('opacity-100');
    toast.classList.add('opacity-0', 'pointer-events-none');
  }, 2400);
}

// Event Listeners
btnStartMic.addEventListener('click', () => audioManager.startRecording());
btnStopMic.addEventListener('click', () => audioManager.stopRecording());
btnClearTranscript.addEventListener('click', clearTranscript);
btnSaveTranscript.addEventListener('click', saveRecordingTranscript);
btnProcessAI.addEventListener('click', processTranscriptWithAI);
btnClearSummary.addEventListener('click', clearMedicalSummary);
btnCopyToPrescription.addEventListener('click', copyToPrescription);

// Toggle simulation mode
btnToggleSimulateMic.addEventListener('click', () => {
  const isSimulated = audioManager.toggleSimulationMode();
  micModeLabel.textContent = isSimulated ? "Simulated Speech" : "Live Mic (16kHz PCM)";
  showToast(isSimulated ? "Switched to simulated voice test" : "Switched to live microphone");
});

// Quick demo loader button
btnLoadDemo.addEventListener('click', () => {
  const placeholderText = document.getElementById('placeholderText');
  if (placeholderText) placeholderText.remove();

  fullTranscriptText = sampleDialogueChunks.join("\n\n");
  audioChunksCount = sampleDialogueChunks.length;
  chunkCountEl.textContent = audioChunksCount;
  updateTranscriptDisplay();
  showToast("Loaded sample consultation transcript");
});

// Patient info modal handlers
btnEditPatient.addEventListener('click', () => {
  patientModal.classList.remove('hidden');
});

btnCancelPatient.addEventListener('click', () => {
  patientModal.classList.add('hidden');
});

btnSavePatient.addEventListener('click', () => {
  const name = inputPatientName.value.trim() || "Dr Tushar";
  const age = inputPatientAge.value.trim() || "30";
  const gender = inputPatientGender.value || "Male";

  patientNameDisplay.textContent = `Patient Name: ${name}`;
  patientMetaDisplay.innerHTML = `Age: ${age} &nbsp; Gender: ${gender}`;
  patientModal.classList.add('hidden');
  showToast("Patient information updated");
});

// Floating chat button interaction
document.getElementById('floatingChatBtn').addEventListener('click', () => {
  showToast("Clinical Copilot ready: ask follow-up questions or drug interactions");
});
