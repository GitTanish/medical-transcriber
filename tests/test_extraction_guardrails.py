import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.services.llm import analyze_clinical_transcript
from tests.helpers import make_llm_completion

# Transcript deliberately omitting medications, vitals, diagnosis, and patient demographics
AMBIGUOUS_TRANSCRIPT = """
Patient came in feeling tired and unwell.
Says: "I have had a mild sore throat for two days, but no fever and no runny nose."
Doctor: "Understood. Get plenty of rest and drink water."
"""


LLM_RESPONSE = {
    "patient_details": {"name": None, "age": None, "sex": None, "identifiers": []},
    "chief_complaint": "Feeling tired and unwell with a mild sore throat",
    "history_of_present_illness": "Mild sore throat for two days.",
    "symptoms": {"positive": ["sore throat"], "negative": ["fever", "runny nose"]},
    "allergies": [],
    "past_medical_history": [],
    "medication_history": [],
    "clinical_observations": [],
    "assessment": [],
    "plan": ["Rest", "Drink water"],
    "clinical_summary": "Patient reports mild sore throat and denies fever or runny nose.",
}


def test_guardrails():
    with patch(
        "backend.services.llm.client.chat.completions.create",
        return_value=make_llm_completion(json.dumps(LLM_RESPONSE)),
    ):
        note = asyncio.run(analyze_clinical_transcript(AMBIGUOUS_TRANSCRIPT))
    data = note.model_dump()

    # Guardrail checks:
    # 1. No inferred patient name, age, or sex
    assert data["patient_details"]["name"] is None, "Inferred patient name when not mentioned!"
    assert data["patient_details"]["age"] is None, "Inferred patient age when not mentioned!"
    assert data["patient_details"]["sex"] is None, "Inferred patient sex when not mentioned!"

    # 2. No inferred medications
    assert len(data["medication_history"]) == 0, "Inferred medications when none were stated!"

    # 3. No inferred diagnoses/assessments (e.g. pharyngitis, viral syndrome)
    assert len(data["assessment"]) == 0, f"Inferred assessment when doctor made none: {data['assessment']}"

    # 4. No inferred vitals (temperature, BP)
    assert len(data["clinical_observations"]) == 0, f"Inferred vitals when none were measured: {data['clinical_observations']}"

    # 5. Symptoms properly captured
    assert any("sore throat" in s.lower() for s in data["symptoms"]["positive"]), f"Sore throat missing from {data['symptoms']['positive']}"
    assert any("fever" in s.lower() for s in data["symptoms"]["negative"]), f"Fever missing from {data['symptoms']['negative']}"


if __name__ == "__main__":
    test_guardrails()
    print("All clinical extraction guardrails passed successfully!")
