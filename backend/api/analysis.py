import logging

from fastapi import APIRouter, HTTPException

from backend.schemas.clinical import ClinicalAnalysisRequest, ClinicalAnalysisResponse
from backend.services.llm import ClinicalAnalysisError, analyze_clinical_transcript

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=ClinicalAnalysisResponse)
async def analyze_transcript(request: ClinicalAnalysisRequest):
    """Analyze a transcript and return a validated clinical note."""
    try:
        return await analyze_clinical_transcript(request.transcript, request.language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ClinicalAnalysisError as exc:
        logger.error(
            "Clinical analysis provider unavailable",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(status_code=503, detail="Clinical analysis service unavailable") from exc
    except Exception as exc:
        logger.error(
            "Unexpected clinical analysis failure",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(status_code=500, detail="Clinical analysis failed") from exc
