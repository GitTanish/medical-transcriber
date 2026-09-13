"""LLM clinical analysis service using Groq."""

from groq import Groq
from backend.config import settings
from backend.schemas.clinical import ClinicalNote

client = Groq(api_key=settings.groq_api_key)

SYSTEM_PROMPT = """You are an expert clinical documentation AI and medical scribe.
Your task is to analyze the medical consultation transcript and extract a factual, structured clinical note.

CRITICAL CLINICAL EXTRACTION GUARDRAILS:
1. Strict Grounding & Zero-Hallucination:
   - Only extract information explicitly supported by the transcript.
   - Never infer medications, diagnoses, vitals, or patient details from context.
   - If a clinical entity is not directly stated in the transcript, DO NOT deduce, extrapolate, or assume it. Leave the field as null or an empty list [].
2. Patient Details:
   - Extract name, age, and sex ONLY if explicitly articulated in the consultation. Never infer sex from names or context, nor guess age.
3. Medications:
   - Include only medications explicitly mentioned by name by the clinician or patient.
   - Never infer, guess, or recommend medications based on complaints or symptoms (e.g. do NOT infer antipyretics or analgesics for fever/headache unless explicitly named).
4. Assessment & Diagnoses:
   - Include only diagnoses, differentials, or clinical impressions explicitly stated by the healthcare provider in the transcript.
   - Never formulate or extrapolate your own diagnosis or clinical impression based on symptom clusters.
5. Vitals & Observations:
   - Record vitals (temperature, blood pressure, heart rate, oxygen saturation) and physical examination findings ONLY if explicit measurements or observations are verbalized.
6. Pertinent Negatives:
   - Record symptoms or conditions explicitly denied by the patient or clinician under symptoms.negative. Do not assume denial if unmentioned.
7. Allergies & Past Medical History:
   - Record only explicitly stated allergies and past conditions. If unmentioned, return an empty list [].

Respond strictly with a JSON object matching this exact schema:
{
  "patient_details": {
    "name": string or null,
    "age": integer or null,
    "sex": string or null,
    "identifiers": list of strings
  },
  "chief_complaint": string or null,
  "history_of_present_illness": string or null,
  "symptoms": {
    "positive": list of strings (symptoms explicitly reported as present),
    "negative": list of strings (pertinent negatives explicitly denied)
  },
  "allergies": list of strings (known allergies explicitly stated),
  "past_medical_history": list of strings (prior conditions explicitly stated),
  "medication_history": [
    {
      "name": string,
      "dosage": string or null,
      "adherence": string or null
    }
  ],
  "clinical_observations": list of strings (vitals and exam findings explicitly verbalized),
  "assessment": list of strings (diagnoses or clinical impressions explicitly stated by provider),
  "plan": list of strings (investigations, prescriptions, advice, and follow-up explicitly stated),
  "clinical_summary": string or null
}

Output format: Return valid JSON ONLY. No preamble, no explanation."""


def _clean_json_payload(raw: str) -> str:
    """Strip markdown code blocks or surrounding whitespace from LLM output."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


async def analyze_clinical_transcript(transcript: str) -> ClinicalNote:
    """Analyze consultation transcript and return a validated Pydantic ClinicalNote."""
    if not transcript or not transcript.strip():
        return ClinicalNote(clinical_summary="No transcript provided for analysis.")

    import time
    last_error = None

    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Medical Consultation Transcript:\n\n{transcript}"},
                ],
                response_format={"type": "json_object"},
                reasoning_effort="none",
                temperature=0.0,
            )

            raw_json = completion.choices[0].message.content or "{}"
            cleaned = _clean_json_payload(raw_json)

            try:
                return ClinicalNote.model_validate_json(cleaned)
            except Exception:
                import json
                parsed = json.loads(cleaned)
                return ClinicalNote.model_validate(parsed)

        except Exception as e:
            last_error = e
            print(f"[LLM Retry Notice] Attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(1.0)

    raise RuntimeError(f"Clinical analysis LLM service failed after retries: {last_error}")
