"""Tests for Clinical schemas and analysis service."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.schemas.clinical import (
    ClinicalNote,
    Medication,
    PatientDetails,
    Symptoms,
)
from backend.services.llm import _clean_json_payload, analyze_clinical_transcript
from tests.helpers import make_llm_completion


def test_clinical_note_schema_instantiation():
    """Verify ClinicalNote schema initializes with valid types and default values."""
    note = ClinicalNote(
        patient_details=PatientDetails(name="Rahul Sharma", age=32, sex="Male"),
        chief_complaint="Fever and headache for 3 days",
        history_of_present_illness="Patient reports fever worsening at night.",
        symptoms=Symptoms(positive=["fever", "headache"], negative=["cough", "chest pain"]),
        allergies=["No known drug allergies"],
        past_medical_history=[],
        medication_history=[Medication(name="Paracetamol", dosage="650mg SOS", adherence="Adherent")],
        clinical_observations=["Temperature 101.2 F"],
        assessment=["Viral fever"],
        plan=["CBC test", "Adequate rest and fluids", "Follow up in 3 days"],
        clinical_summary="32-year-old male with 3-day viral fever; CBC advised.",
    )

    assert note.patient_details.name == "Rahul Sharma"
    assert note.patient_details.age == 32
    assert "fever" in note.symptoms.positive
    assert "cough" in note.symptoms.negative
    assert len(note.assessment) > 0
    assert len(note.plan) > 0
    assert note.clinical_summary is not None


def test_empty_transcript_handling():
    """Verify analyze_clinical_transcript handles empty input gracefully without calling external LLM."""
    response = asyncio.run(analyze_clinical_transcript(""))
    assert isinstance(response, ClinicalNote)
    assert response.clinical_summary == "No transcript provided for analysis."
    assert response.assessment == []
    assert response.plan == []


LLM_RESPONSE = {
    "patient_details": {
        "name": "Rahul",
        "age": 30,
        "sex": "Male",
        "identifiers": [],
    },
    "chief_complaint": "Persistent cough and mild fever",
    "history_of_present_illness": "Cough and fever have persisted for three days.",
    "symptoms": {
        "positive": ["cough", "fever"],
        "negative": ["chest pain", "shortness of breath"],
    },
    "allergies": [],
    "past_medical_history": [],
    "medication_history": [{"name": "amoxicillin", "dosage": "500mg", "adherence": None}],
    "clinical_observations": [],
    "assessment": ["Acute bronchitis"],
    "plan": ["Follow up in one week"],
    "clinical_summary": "Thirty-year-old male with three days of cough and fever.",
}


def test_analyze_clinical_transcript():
    sample_text = (
        "Patient Rahul, 30-year-old male, complains of persistent cough and mild fever for 3 days. "
        "He denies chest pain or shortness of breath. "
        "Doctor's provisional assessment is acute bronchitis. "
        "Plan is to take amoxicillin 500mg and follow up in one week."
    )
    with patch(
        "backend.services.llm.client.chat.completions.create",
        return_value=make_llm_completion(json.dumps(LLM_RESPONSE)),
    ):
        response = asyncio.run(analyze_clinical_transcript(sample_text))

    assert isinstance(response, ClinicalNote)
    assert response.chief_complaint is not None
    assert response.assessment == ["Acute bronchitis"]
    assert response.symptoms.negative == ["chest pain", "shortness of breath"]
    assert response.medication_history[0].name == "amoxicillin"


def test_clean_json_payload_handles_fenced_and_wrapped_output():
    payload = _clean_json_payload("prefix\n```json\n{\"clinical_summary\": \"ok\"}\n```\nsuffix")
    assert json.loads(payload) == {"clinical_summary": "ok"}


if __name__ == "__main__":
    test_clinical_note_schema_instantiation()
    test_empty_transcript_handling()
    test_analyze_clinical_transcript()
    print("All clinical schema tests passed successfully!")
