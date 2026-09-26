from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientDetails(BaseModel):
    name: Annotated[
        str | None,
        Field(default=None, description="Patient's name if mentioned in the transcript.")
    ] = None

    age: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            le=150,
            description="Patient's age if explicitly mentioned."
        )
    ] = None

    sex: Annotated[
        str | None,
        Field(default=None, description="Patient's sex if explicitly mentioned.")
    ] = None

    identifiers: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Patient identifiers explicitly mentioned in the transcript"
        )
    ] = Field(default_factory=list)


class Symptoms(BaseModel):
    positive: Annotated[
        list[str],
        Field(default_factory=list, description="Symptoms explicitly reported as present.")
    ] = Field(default_factory=list)

    negative: Annotated[
        list[str],
        Field(default_factory=list, description="Symptoms explicitly denied.")
    ] = Field(default_factory=list)


class Medication(BaseModel):
    name: Annotated[
        str, 
        Field(description="Medication name")
    ]

    dosage: Annotated[
        str | None,
        Field(default=None, description="Dosage and frequency if mentioned")
    ] = None

    adherence: Annotated[
        str | None,
        Field(default=None, description="Adherence information if mentioned")
    ] = None


class ClinicalNote(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        str_strip_whitespace=True,
    )

    patient_details: Annotated[
        PatientDetails,
        Field(default_factory=PatientDetails, description="Patient details and identifiers")
    ] = Field(default_factory=PatientDetails)

    language: str | None = Field(
        default=None,
        max_length=16,
        description="Primary language of the source transcript when identifiable.",
    )
    script: str | None = Field(
        default=None,
        max_length=32,
        description="Primary script used by the source transcript when identifiable.",
    )

    chief_complaint: Annotated[
        str | None,
        Field(default=None, description="Primary reason for the visit, in the patient's own terms")
    ] = None

    history_of_present_illness: Annotated[
        str | None,
        Field(
            default=None,
            description="Chronological account of symptoms including onset, duration, progression, aggravating and relieving factors" 
        )
    ] = None

    symptoms: Annotated[
        Symptoms,
        Field(default_factory=Symptoms, description="Positive and explicitly stated negative symptoms.")
    ] = Field(default_factory=Symptoms)

    allergies: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Known allergies or adverse drug reactions explicitly mentioned"
        )
    ] = Field(default_factory=list)

    past_medical_history: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Prior conditions, surgeries, and hospitalisations"
        )
    ] = Field(default_factory=list)

    medication_history: Annotated[
        list[Medication],
        Field(
            default_factory=list,
            description="Current medicines, dosages, adherence, and related information"
        )
    ] = Field(default_factory=list)

    clinical_observations: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Vitals and examination findings stated during the consultation."
        )
    ] = Field(default_factory=list)

    assessment: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="The healthcare provider's assessment, diagnoses, or differentials."
        )
    ] = Field(default_factory=list)

    plan: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Investigations ordered, prescriptions, advice, and follow-ups"
        )
    ] = Field(default_factory=list)

    clinical_summary: Annotated[
        str | None,
        Field(
            default=None,
            description="Concise, structured clinical summary suitable for a medical record"
        )
    ] = None


class ClinicalAnalysisRequest(BaseModel):
    transcript: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100_000,
            description="Full consultation transcript to analyze.",
        )
    ]
    language: str | None = Field(
        default=None,
        max_length=16,
        description="Optional detected transcript language code.",
    )

    @field_validator("transcript")
    @classmethod
    def validate_transcript(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transcript must contain non-whitespace content")
        return value.strip()

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        return value or None


class ClinicalAnalysisResponse(ClinicalNote):
    """Clinical analysis response matching the structured ClinicalNote model."""