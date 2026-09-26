"""Transcript-related schemas."""

from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class Segment(BaseModel):
    id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(max_length=20_000)
    language: str | None = Field(default=None, max_length=16)


class TranscriptionRequest(BaseModel):
    language: str | None = Field(default="auto", description="Spoken language code or auto")


class TranscriptionResponse(BaseModel):
    text: str = Field(max_length=200_000)
    filename: str | None = Field(default=None, max_length=255)
    language: str | None = Field(default=None, max_length=32)
    detected_language: str | None = Field(default=None, max_length=32)
    duration: float | None = Field(default=None, ge=0)
    segments: list[Segment] = Field(default_factory=list, max_length=500)
