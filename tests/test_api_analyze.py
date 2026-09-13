import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from fastapi.testclient import TestClient
from backend.main import app

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


def test_analyze_endpoint():
    print("Sending consultation transcript to POST /api/analyze/ ...\n")
    response = client.post(
        "/api/analyze/",
        json={"transcript": SAMPLE_TRANSCRIPT},
    )

    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200, f"Analysis failed: {response.text}"

    data = response.json()
    print("\nStructured ClinicalNote JSON Response:")
    print(json.dumps(data, indent=2))

    # Basic assertions
    assert data["patient_details"]["name"] == "Rahul Sharma"
    assert data["patient_details"]["age"] == 32
    assert data["patient_details"]["sex"].lower() == "male"
    assert len(data["symptoms"]["positive"]) > 0
    assert len(data["symptoms"]["negative"]) > 0
    assert len(data["assessment"]) > 0


if __name__ == "__main__":
    test_analyze_endpoint()
