from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.transcription import router as transcription_router
from backend.api.analysis import router as analysis_router

app = FastAPI(
    title="Medical Transcriber",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcription_router, prefix="/api/transcribe", tags=["transcription"])
app.include_router(analysis_router, prefix="/api/analyze", tags=["analysis"])


@app.get("/health")
def health_check():
    return {"status": "ok"}


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")