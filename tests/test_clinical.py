"""Tests for Clinical schemas and analysis service."""

import asyncio
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pytest
from backend.schemas.clinical import (
    ClinicalAnalysisRequest,
    ClinicalAnalysisResponse,
    ClinicalNote,
    Medication,
    PatientDetails,
    Symptoms,
)
from backend.services.llm import analyze_clinical_transcript


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


def test_analyze_clinical_transcript():
    """Verify analyze_clinical_transcript extracts structured clinical fields adhering to ClinicalNote schema."""
    sample_text = (
        "Patient Rahul, 30-year-old male, complains of persistent cough and mild fever for 3 days. "
        "He denies chest pain or shortness of breath. "
        "Doctor's provisional assessment is acute bronchitis. "
        "Plan is to take amoxicillin 500mg and follow up in one week."
    )
    response = asyncio.run(analyze_clinical_transcript(sample_text))

    # Validate against actual ClinicalNote / ClinicalAnalysisResponse schema fields
    assert isinstance(response, ClinicalNote)
    assert response.chief_complaint is not None or response.clinical_summary is not None
    assert isinstance(response.assessment, list)
    assert isinstance(response.plan, list)
    assert isinstance(response.symptoms.positive, list)
    assert isinstance(response.symptoms.negative, list)
    assert isinstance(response.medication_history, list)


if __name__ == "__main__":
    test_clinical_note_schema_instantiation()
    test_empty_transcript_handling()
    test_analyze_clinical_transcript()
    print("All clinical schema tests passed successfully!")
