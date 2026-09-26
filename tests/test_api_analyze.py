import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from fastapi.testclient import TestClient

from backend.main import app
from tests.helpers import make_llm_completion

client = TestClient(app)

SAMPLE_TRANSCRIPT = """Patient name is Rahul Sharma, 32-year-old male.
He has had fever for three days with headache and body aches.
The fever is worse at night.
He denies cough, chest pain, and shortness of breath.
No known drug allergies.
No significant past medical history.
He takes paracetamol occasionally.
Temperature is 101.2 Fahrenheit.
The provisional diagnosis is viral fever.
CBC is advised, along with increased fluids and rest.
Follow up in three days if symptoms persist."""

LLM_RESPONSE = {
    "patient_details": {
        "name": "Rahul Sharma",
        "age": 32,
        "sex": "Male",
        "identifiers": [],
    },
    "chief_complaint": "Fever, headache, and body aches for three days",
    "history_of_present_illness": "Fever has been present for three days and worsens at night.",
    "symptoms": {
        "positive": ["fever", "headache", "body aches"],
        "negative": ["cough", "chest pain", "shortness of breath"],
    },
    "allergies": ["No known drug allergies"],
    "past_medical_history": [],
    "medication_history": [{"name": "paracetamol", "dosage": None, "adherence": "occasionally"}],
    "clinical_observations": ["Temperature 101.2 Fahrenheit"],
    "assessment": ["Provisional viral fever"],
    "plan": ["CBC", "Increased fluids", "Rest", "Follow up in three days"],
    "clinical_summary": "32-year-old male with three days of fever and associated symptoms.",
}


def test_analyze_endpoint():
    with patch(
        "backend.services.llm.client.chat.completions.create",
        return_value=make_llm_completion(json.dumps(LLM_RESPONSE)),
    ):
        response = client.post(
            "/api/analyze/",
            json={"transcript": SAMPLE_TRANSCRIPT},
        )

    assert response.status_code == 200, f"Analysis failed: {response.text}"
    data = response.json()
    assert data["patient_details"]["name"] == "Rahul Sharma"
    assert data["patient_details"]["age"] == 32
    assert data["patient_details"]["sex"].lower() == "male"
    assert len(data["symptoms"]["positive"]) > 0
    assert len(data["symptoms"]["negative"]) > 0
    assert len(data["assessment"]) > 0


def test_analyze_endpoint_rejects_blank_transcript():
    response = client.post("/api/analyze/", json={"transcript": "   "})
    assert response.status_code == 422


def test_analyze_endpoint_rejects_oversized_transcript():
    response = client.post("/api/analyze/", json={"transcript": "x" * 100_001})
    assert response.status_code == 422


def test_analyze_endpoint_hides_provider_errors():
    with patch(
        "backend.services.llm.client.chat.completions.create",
        side_effect=RuntimeError("provider secret details"),
    ):
        response = client.post("/api/analyze/", json={"transcript": SAMPLE_TRANSCRIPT})
    assert response.status_code == 503
    assert "secret details" not in response.text
    assert response.json()["detail"] == "Clinical analysis service unavailable"


def test_health_disables_caching_and_blocks_untrusted_origin():
    response = client.get("/health", headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in response.headers


if __name__ == "__main__":
    test_analyze_endpoint()
