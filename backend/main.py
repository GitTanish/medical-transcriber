from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.analysis import router as analysis_router
from backend.api.transcription import router as transcription_router
from backend.config import settings

app = FastAPI(
    title="Medical Transcriber",
    version="1.0.0",
)

allowed_origins = settings.allowed_cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.middleware("http")
async def add_response_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/api/") or request.url.path == "/health":
        response.headers.setdefault("Cache-Control", "no-store")
    return response


app.include_router(transcription_router, prefix="/api/transcribe", tags=["transcription"])
app.include_router(analysis_router, prefix="/api/analyze", tags=["analysis"])


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
