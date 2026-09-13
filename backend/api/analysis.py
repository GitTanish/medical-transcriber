"""LLM clinical analysis endpoints."""

from fastapi import APIRouter, HTTPException
from backend.schemas.clinical import ClinicalAnalysisRequest, ClinicalAnalysisResponse
from backend.services.llm import analyze_clinical_transcript

router = APIRouter()


@router.post("/", response_model=ClinicalAnalysisResponse)
async def analyze_transcript(request: ClinicalAnalysisRequest):
    """Analyze a transcript to extract clinical findings, SOAP note, and entities."""
    try:
        analysis = await analyze_clinical_transcript(request.transcript)
        return analysis
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
